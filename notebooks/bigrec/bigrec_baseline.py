"""Entrenamiento y evaluación del baseline BIGRec de LatentR3 (lo usan notebooks/bigrec/baseline_bigrec.ipynb y notebooks/d3/baseline_d3.ipynb;
notebooks/base_cot/ y notebooks/latentr3/ reutilizan sus funciones de datos y evaluación).

BIGRec como lo corrieron los autores de LatentR3: Qwen2.5-1.5B con ajuste completo y decodificación
restringida al catálogo, es decir, el pipeline de LatentR3 sin el token latente. El código de LatentR3
deriva del de D3 (SAI990323/DecodingMatters), que es BIGRec reimplementado con Qwen por sus autores.
De D3 (commit 585c842) se usan sin modificar `D3Dataset` (prompt y etiquetas) y
`CFEnhancedLogitsProcessor` (decodificación restringida). El entrenamiento copia los argumentos de
`latent_attention_train.py` de LatentR3, que son los de `code/train.py` de D3. La generación sigue
`latent_attention_eval.py`. Las decisiones están en notebooks/bigrec/README.md.
"""
import os
import io
import re
import json
import math
import time
import glob
import shutil
import contextlib
import numpy as np
import torch

from baseline_comun import SEED, buscar_archivo, verificar_sha256, cargar_modulo

BASE_MODEL = "Qwen/Qwen2.5-1.5B"   # paper, App. B
# Entrenamiento: latent_attention_train.py L31-35 y L116-139 (= D3 code/train.py L60-64 y L134-156)
BATCH_SIZE = 128        # batch efectivo; el autor lo confirma en el issue #2
MICRO_BATCH = 4         # filas por paso de la GPU; acumulación = BATCH_SIZE // MICRO_BATCH
EPOCAS = 3
LR = 3e-4               # el autor lo confirma en el issue #2; la grilla del paper es {3e-3, 3e-4, 3e-5}
CUTOFF_LEN = 512        # D3Dataset trunca por la izquierda a este largo
WARMUP = 20             # scheduler lineal (default de HF)
PACIENCIA = 1           # early stopping sobre eval_loss, por época (paper, App. B)
SAVE_TOTAL_LIMIT = 2    # LatentR3 L129 (D3 usa 1)
# Generación: latent_attention_eval.py L65-117 y latent_train.sh L447
NUM_BEAMS = 10
MAX_NEW_TOKENS = 128    # D3 usa 64 (evaluate.py L115)
BATCH_GENERACION = 24
LENGTH_PENALTY = {"bigrec": 1.0,   # latent_attention_eval.py L32
                  "d3": 0.0}       # D3 sin ensemble: evaluate2.sh L8 (sin --guidance_scale, así que sin TFA)
FILAS_POR_PARTE = 256   # la evaluación se guarda por partes para poder retomarla
# Palabra de cada categoría en el prompt: latent_attention_train.py L49 (D3 code/train.py L83 es igual, sin yelp)
CATEGORY_DICT = {"Office_Products": "office products", "Books": "books", "steam": "games",
                 "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games",
                 "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
                 "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie",
                 "yelp": "resturant"}
# sha256 de los archivos de D3 que se cargan, en el commit 585c8427dd327a33155b5566f32bc22e21b5236c
COMMIT_D3 = "585c8427dd327a33155b5566f32bc22e21b5236c"
SHA256 = {os.path.join("code", "dataset.py"): "2b0ea46ef8cab6f9f97516e050c219a009405a4698c1e5da0ffef8aaf9f8b706",
          os.path.join("code", "LogitProcesser.py"): "f0b52d7accfca944641274bd67ad4e3b52ce4cd36224659cd2a94ed5ad73bae6"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D3Dataset = None                  # se cargan del código de D3 en cargar_d3
CFEnhancedLogitsProcessor = None
get_hash = None


# ---------- Código de D3 ----------

def verificar_codigo(codigo):
    """Comprueba que los archivos de D3 sean los del commit revisado, antes de importarlos."""
    verificar_sha256(codigo, SHA256, COMMIT_D3, "SAI990323/DecodingMatters")


def cargar_d3(codigo):
    """Importa D3Dataset y CFEnhancedLogitsProcessor de D3 sin tocar sys.path."""
    global D3Dataset, CFEnhancedLogitsProcessor, get_hash
    verificar_codigo(codigo)
    D3Dataset = cargar_modulo("d3_dataset", os.path.join(codigo, "code", "dataset.py")).D3Dataset
    logits = cargar_modulo("d3_logits", os.path.join(codigo, "code", "LogitProcesser.py"))
    CFEnhancedLogitsProcessor, get_hash = logits.CFEnhancedLogitsProcessor, logits.get_hash


# ---------- Datos ----------

def cargar_tokenizer(ruta=BASE_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ruta)


def construir_dataset(tokenizer, archivo, categoria, sample=-1, test=False, seed=0):
    """D3Dataset con los argumentos de LatentR3 (L91-92; eval L54). Con test=True devuelve solo el prompt."""
    return D3Dataset(train_file=archivo, tokenizer=tokenizer, max_len=2560 if test else CUTOFF_LEN,
                     sample=sample, test=test, seed=seed, category=CATEGORY_DICT[categoria])


def a_hf(dataset):
    # Igual que LatentR3 L105-108 y D3 L125-127
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({k: [v[k] for v in dataset] for k in dataset[0].keys()})


def largos_tokens(dataset):
    largos = np.array([len(dataset[i]["input_ids"]) for i in range(len(dataset))])
    return {"filas": len(largos), "media": largos.mean(), "p99": np.percentile(largos, 99), "máximo": largos.max()}


# ---------- Entrenamiento ----------

def entrenado(salida):
    registro = os.path.join(salida, "entrenamiento.json")
    if not os.path.isfile(registro):
        return False
    with open(registro) as f:
        return json.load(f).get("terminado", False)


def entrenar(categoria, base, salida, sample_train=-1, sample_valid=-1, micro_batch=MICRO_BATCH,
             checkpointing=True, optim="adamw_torch", borrar_intermedios=True, modelo_base=BASE_MODEL):
    """SFT de BIGRec en una categoría. Retoma desde el último checkpoint si lo hay y se salta si ya terminó.

    Guarda en `salida`: `trainer/` (checkpoints por época, se borran al final), `modelo/` (la época con menor
    eval_loss) y `entrenamiento.json`. micro_batch, checkpointing y optim solo cambian la memoria y la
    velocidad, no el batch efectivo ni el método (notebooks/bigrec/README.md, Adaptaciones).
    """
    from transformers import (AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback,
                              EarlyStoppingCallback, DataCollatorForSeq2Seq)
    import transformers
    dir_trainer, dir_modelo = os.path.join(salida, "trainer"), os.path.join(salida, "modelo")
    if entrenado(salida):
        print(f"{categoria}: ya entrenado en {dir_modelo}; se salta.")
        with open(os.path.join(salida, "entrenamiento.json")) as f:
            return json.load(f)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Hace falta una GPU con bf16 (los autores entrenan en bf16; una T4 no sirve).")
    assert BATCH_SIZE % micro_batch == 0

    tokenizer = cargar_tokenizer(modelo_base)
    train = construir_dataset(tokenizer, buscar_archivo(base, "train", categoria), categoria, sample_train)
    valid = construir_dataset(tokenizer, buscar_archivo(base, "valid", categoria), categoria, sample_valid)
    modelo = AutoModelForCausalLM.from_pretrained(modelo_base, torch_dtype=torch.bfloat16)   # L64-69

    class Cronometro(TrainerCallback):
        # Tiempo de cada paso de optimización, para estimar la duración sin contar la evaluación en valid
        def __init__(self):
            self.marcas = []
        def on_train_begin(self, args, state, control, **kwargs):
            self.marcas.append(time.time())
        def on_step_end(self, args, state, control, **kwargs):
            self.marcas.append(time.time())

    cronometro = Cronometro()
    args = TrainingArguments(
        output_dir=dir_trainer,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=MICRO_BATCH,   # 4 como los autores: eval_loss es una media de medias por lote
        gradient_accumulation_steps=BATCH_SIZE // micro_batch,
        warmup_steps=WARMUP,
        num_train_epochs=EPOCAS,
        learning_rate=LR,
        bf16=True,
        logging_steps=1,
        optim=optim,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,              # por eval_loss (default de HF)
        group_by_length=False,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,                                # 42, el default de HF que usaron los autores
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None,
    )
    trainer = Trainer(
        model=modelo, args=args, train_dataset=a_hf(train), eval_dataset=a_hf(valid), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PACIENCIA), cronometro],
    )
    modelo.config.use_cache = False   # L141

    checkpoints = sorted(glob.glob(os.path.join(dir_trainer, "checkpoint-*")), key=lambda r: int(r.rsplit("-", 1)[1]))
    paso_inicial = int(checkpoints[-1].rsplit("-", 1)[1]) if checkpoints else 0
    if checkpoints:
        print(f"{categoria}: se retoma desde {checkpoints[-1]}")
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    trainer.train(resume_from_checkpoint=checkpoints[-1] if checkpoints else None)
    segundos = time.time() - inicio

    modelo.save_pretrained(dir_modelo)    # la mejor época, cargada por load_best_model_at_end
    tokenizer.save_pretrained(dir_modelo)
    pasos = np.diff(cronometro.marcas)
    evaluaciones = [h for h in trainer.state.log_history if "eval_loss" in h]
    registro = {
        "categoria": categoria, "terminado": True,
        "mejor_checkpoint": trainer.state.best_model_checkpoint, "mejor_eval_loss": trainer.state.best_metric,
        "evaluaciones": [{"epoca": h["epoch"], "eval_loss": h["eval_loss"], "eval_runtime": h["eval_runtime"]}
                         for h in evaluaciones],
        "pasos_totales": trainer.state.global_step, "pasos_esta_sesion": trainer.state.global_step - paso_inicial,
        "filas_train": len(train), "filas_valid": len(valid),
        "segundos_esta_sesion": segundos,
        "s_por_paso_mediana": float(np.median(pasos[1:])) if len(pasos) > 1 else None,   # sin el primer paso
        "s_valid_por_fila": (float(np.mean([h["eval_runtime"] for h in evaluaciones])) / len(valid)
                             if evaluaciones else None),
        "memoria_max_gb": torch.cuda.max_memory_allocated() / 2**30,
        "memoria_reservada_max_gb": torch.cuda.max_memory_reserved() / 2**30,
        "micro_batch": micro_batch, "acumulacion": BATCH_SIZE // micro_batch,
        "gradient_checkpointing": checkpointing, "optim": optim,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(salida, "entrenamiento.json"), "w") as f:   # se escribe al final: marca el fin
        json.dump(registro, f, indent=1)
    del trainer, modelo
    torch.cuda.empty_cache()
    if borrar_intermedios:
        shutil.rmtree(dir_trainer, ignore_errors=True)
    return registro


# ---------- Decodificación restringida ----------

def tokenizar_titulos(info_file, tokenizer):
    # Igual que D3 evaluate.py L44-55: cada título como '### Response: \n"título"\n', más EOS (L60)
    with open(info_file, "r", encoding="utf-8") as f:
        info = f.readlines()
    info = ["\"" + linea.split("\t")[0].strip(" ") + "\"\n" for linea in info]
    info = ["### Response: \n" + t for t in info]   # mismo texto que el f-string de D3 (sin espacios extra)
    prefijos = [tokenizer(t).input_ids for t in info]
    for ids in prefijos:
        ids.append(tokenizer.eos_token_id)
    return prefijos


def construir_trie(info_file, tokenizer):
    """Trie de los títulos del catálogo como D3 evaluate.py L57-77 (sin el token latente que inserta LatentR3).

    La primera clave son los 4 tokens de '### Response: \\n'; las siguientes, los tokens ya generados. Cada
    clave apunta a los tokens permitidos a continuación.
    """
    trie = {}
    for ids in tokenizar_titulos(info_file, tokenizer):
        for i in range(4, len(ids)):
            clave = get_hash(ids[:i]) if i == 4 else get_hash(ids[4:i])
            trie.setdefault(clave, set()).add(ids[i])
    return {clave: list(tokens) for clave, tokens in trie.items()}


@torch.no_grad()
def generar(modelo, tokenizer, encodings, trie, length_penalty, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS):
    """Top-`num_beams` títulos para un lote de prompts, como evaluate() de latent_attention_eval.py L65-131.

    Padding a la izquierda con attention_mask. No se piden los puntajes por paso (output_scores), que solo
    ocupan memoria: el orden de los beams es el mismo.
    """
    from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper
    largo = max(len(e["input_ids"]) for e in encodings)
    ids = [[tokenizer.pad_token_id] * (largo - len(e["input_ids"])) + e["input_ids"] for e in encodings]
    mascara = [[0] * (largo - len(e["input_ids"])) + e["attention_mask"] for e in encodings]
    config = GenerationConfig(num_beams=num_beams, length_penalty=length_penalty, num_return_sequences=num_beams,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                              max_new_tokens=max_new_tokens)
    restriccion = CFEnhancedLogitsProcessor(
        guidance_scale=1.0, cf_logits=None, cf_dict=None, unconditional_ids=None, model=modelo, tokenizer=tokenizer,
        prefix_allowed_tokens_fn=lambda _, prefijo: trie.get(get_hash(prefijo), []), num_beams=num_beams)
    salida = modelo.generate(torch.tensor(ids, device=modelo.device),
                             attention_mask=torch.tensor(mascara, device=modelo.device),
                             generation_config=config, return_dict_in_generate=True,
                             logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), restriccion]))
    textos = tokenizer.batch_decode(salida.sequences[:, largo:], skip_special_tokens=True)
    return [textos[i * num_beams:(i + 1) * num_beams] for i in range(len(encodings))]


def cargar_modelo(dir_modelo, tokenizer):
    from transformers import AutoModelForCausalLM
    modelo = AutoModelForCausalLM.from_pretrained(dir_modelo, torch_dtype=torch.bfloat16).to(device)
    modelo.config.pad_token_id = modelo.config.eos_token_id = tokenizer.eos_token_id   # eval L60-61
    modelo.config.bos_token_id = tokenizer.bos_token_id
    modelo.config.use_cache = True   # se guardó en False para entrenar; solo cambia la velocidad
    return modelo.eval()


def evaluar_categoria(categoria, base, salida, variante="bigrec", batch_size=BATCH_GENERACION, sample=-1,
                      filas_por_parte=FILAS_POR_PARTE, carpeta=None):
    """Genera el top-10 de cada fila de test con el modelo de `salida/modelo` y lo guarda por partes.

    `variante` fija el length_penalty (BIGRec 1.0, D3 0.0). Se salta las partes ya escritas, así que se puede
    retomar. Al final une las partes en `final_result.json`, con el formato que lee calc.py de los autores.
    """
    dir_modelo = os.path.join(salida, "modelo")
    dir_pred = os.path.join(salida, "predicciones", carpeta or variante)
    final = os.path.join(dir_pred, "final_result.json")
    if os.path.isfile(final):
        print(f"{categoria} ({carpeta or variante}): ya evaluado en {final}; se salta.")
        return final
    os.makedirs(dir_pred, exist_ok=True)
    tokenizer = cargar_tokenizer(dir_modelo)
    tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id   # eval L51-53
    tokenizer.padding_side = "left"
    test = construir_dataset(tokenizer, buscar_archivo(base, "test", categoria), categoria, sample=sample, test=True)
    encodings = [test[i] for i in range(len(test))]
    filas = test.get_all()
    trie = construir_trie(buscar_archivo(base, "info", categoria), tokenizer)
    ruta_registro = os.path.join(dir_pred, "registro.json")
    registro = {"filas": 0, "segundos": 0.0, "batch_size": batch_size, "length_penalty": LENGTH_PENALTY[variante]}
    if os.path.isfile(ruta_registro):
        with open(ruta_registro) as f:
            registro = json.load(f)

    modelo = None
    n_partes = math.ceil(len(encodings) / filas_por_parte)
    for p in range(n_partes):
        ruta = os.path.join(dir_pred, f"parte_{p:03d}.json")
        if os.path.isfile(ruta):
            continue
        if modelo is None:
            modelo = cargar_modelo(dir_modelo, tokenizer)
            torch.cuda.reset_peak_memory_stats()
        ini, fin = p * filas_por_parte, min((p + 1) * filas_por_parte, len(encodings))
        inicio = time.time()
        predicciones = []
        for i in range(ini, fin, batch_size):
            predicciones += generar(modelo, tokenizer, encodings[i:min(i + batch_size, fin)], trie,
                                    LENGTH_PENALTY[variante])
        segundos = time.time() - inicio
        parte = []
        for fila, prediccion in zip(filas[ini:fin], predicciones):
            fila = {k: v for k, v in fila.items() if k != "dedup"}   # eval L157-159
            fila["predict"] = prediccion
            parte.append(fila)
        with open(ruta + ".tmp", "w") as f:
            json.dump(parte, f, indent=1)
        os.replace(ruta + ".tmp", ruta)   # una parte a medio escribir nunca queda con el nombre final
        registro["filas"] += fin - ini
        registro["segundos"] += segundos
        registro["memoria_max_gb"] = torch.cuda.max_memory_allocated() / 2**30
        with open(ruta_registro, "w") as f:
            json.dump(registro, f, indent=1)
        print(f"{categoria} ({carpeta or variante}): parte {p + 1}/{n_partes}, {segundos / (fin - ini):.2f} s por fila")

    datos = []
    for p in range(n_partes):   # como merge.py de los autores
        with open(os.path.join(dir_pred, f"parte_{p:03d}.json")) as f:
            datos.extend(json.load(f))
    assert len(datos) == len(encodings)
    with open(final, "w") as f:
        json.dump(datos, f, indent=1)
    if modelo is not None:
        del modelo
        torch.cuda.empty_cache()
    return final


# ---------- Métricas ----------

def titulos_catalogo(info_file):
    # Normalización de calc.py L356
    with open(info_file, encoding="utf-8") as f:
        return [linea.split("\t")[0].strip("\"").strip(" ").strip("\n").strip("\"") for linea in f]


def metricas(final_result, info_file, ks=(1, 3, 5, 10, 20)):
    """HR y NDCG por título con el criterio de calc.py; cuenta títulos fuera del catálogo y repetidos."""
    with open(final_result) as f:
        datos = json.load(f)
    catalogo = set(titulos_catalogo(info_file))
    hr, ndcg = np.zeros(len(ks)), np.zeros(len(ks))
    fuera = repetidos = 0
    for fila in datos:
        predicciones = [p.strip(" \n").strip("\"").strip(" ") for p in fila["predict"]]   # calc.py L380
        objetivo = fila["output"].strip(" \n\"")                                           # calc.py L387
        fuera += sum(p not in catalogo for p in predicciones)
        repetidos += len(predicciones) - len(set(predicciones))
        # Como calc.py L395-399: si el objetivo aparece más de una vez, cuenta la última posición
        posiciones = [i for i, p in enumerate(predicciones) if p == objetivo]
        if posiciones:
            for j, k in enumerate(ks):
                if posiciones[-1] < k:
                    hr[j] += 1
                    ndcg[j] += 1 / math.log2(posiciones[-1] + 2)
    resultado = {f"HR@{k}": v / len(datos) for k, v in zip(ks, hr)}
    resultado.update({f"NDCG@{k}": v / len(datos) for k, v in zip(ks, ndcg)})
    resultado.update({"fuera del catálogo": fuera, "repetidos": repetidos, "filas": len(datos)})
    return resultado


def verificar_con_calc(final_result, info_file, calc_py):
    """Corre gao() de calc.py de los autores sobre el mismo archivo y comprueba que dé lo mismo que metricas()."""
    calc = cargar_modulo("latentr3_calc", calc_py)
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(io.StringIO()):
        calc.gao(final_result, info_file)
    texto = salida.getvalue()
    # gao imprime las rutas, NDCG@[1,3,5,10,20], HR@[1,3,5,10,20] y el número de títulos fuera del catálogo
    ndcg_calc, hr_calc = [np.array(g.split(), dtype=float) for g in re.findall(r"\[([^\]]*)\]", texto)[-2:]]
    fuera_calc = int(texto.strip().splitlines()[-1])
    propias = metricas(final_result, info_file)
    ks = (1, 3, 5, 10, 20)
    assert np.allclose(ndcg_calc, [propias[f"NDCG@{k}"] for k in ks], atol=1e-6), (ndcg_calc, propias)
    assert np.allclose(hr_calc, [propias[f"HR@{k}"] for k in ks], atol=1e-6), (hr_calc, propias)
    assert fuera_calc == propias["fuera del catálogo"], (fuera_calc, propias)
    return {"NDCG calc.py": ndcg_calc, "HR calc.py": hr_calc, "fuera calc.py": fuera_calc}


# ---------- Verificaciones ----------

def verificar_prompt(base, categoria, latentr3, filas=200, tokenizer=None):
    """Los input_ids, labels y attention_mask de D3Dataset deben ser los de LatentRDataset sin el token latente.

    Agrega <|Thought|> a una copia del tokenizer como LatentR3 (L78-86) y compara fila a fila en train, valid
    (con labels) y test (solo prompt).
    """
    tokenizer = tokenizer or cargar_tokenizer()
    latente = cargar_tokenizer()
    especiales = latente.additional_special_tokens
    especiales.append("<|Thought|>")
    latente.add_special_tokens({"additional_special_tokens": especiales})
    id_latente = latente.convert_tokens_to_ids("<|Thought|>")
    LatentRDataset = cargar_modulo("latentr3_dataset",
                                   os.path.join(latentr3, "src", "latent", "reasoning_dataset.py")).LatentRDataset
    for split, test in [("train", False), ("valid", False), ("test", True)]:
        archivo = buscar_archivo(base, split, categoria)
        d3 = construir_dataset(tokenizer, archivo, categoria, sample=filas, test=test)
        lr3 = LatentRDataset(train_file=archivo, tokenizer=latente, max_len=2560 if test else CUTOFF_LEN,
                             sample=filas, test=test, seed=0, category=CATEGORY_DICT[categoria], K=0)
        for i in range(len(d3)):
            a, b = d3[i], lr3[i]
            quedan = [j for j, t in enumerate(b["input_ids"]) if t != id_latente]
            assert len(quedan) == len(b["input_ids"]) - 1, "LatentRDataset debería tener un solo token latente"
            for clave in a:
                assert a[clave] == [b[clave][j] for j in quedan], (split, i, clave)
    ejemplo = construir_dataset(tokenizer, buscar_archivo(base, "train", categoria), categoria, sample=1)[0]
    return tokenizer.decode(ejemplo["input_ids"])


def verificar_trie(info_file, tokenizer, encodings_test):
    """Comprueba que el trie solo genere títulos del catálogo y que todos los prompts terminen en su raíz."""
    prefijos = tokenizar_titulos(info_file, tokenizer)
    raices = {tuple(ids[:4]) for ids in prefijos}
    assert len(raices) == 1, f"'### Response: \\n' se tokeniza de {len(raices)} formas distintas"
    raiz = list(raices.pop())
    finales = {tuple(e["input_ids"][-4:]) for e in encodings_test}
    assert finales == {tuple(raiz)}, "algún prompt no termina en los 4 tokens de la raíz del trie"
    # Recorre el trie desde la raíz: cada camino que termina en EOS es un título generable
    trie = construir_trie(info_file, tokenizer)
    generables, pila = [], [[]]
    while pila:
        camino = pila.pop()
        for token in trie.get(get_hash(raiz) if not camino else get_hash(camino), []):
            if token == tokenizer.eos_token_id:
                generables.append(tokenizer.decode(camino).strip(" \n").strip("\"").strip(" "))
            else:
                pila.append(camino + [token])
    # Normalizados como calc.py, los títulos generables son exactamente los del catálogo: nada queda fuera
    catalogo = set(titulos_catalogo(info_file))
    assert set(generables) == catalogo, (len(set(generables) - catalogo), len(catalogo - set(generables)))
    tokens = [len(ids) - 4 for ids in prefijos]   # título con comillas, salto de línea y EOS
    return {"ítems": len(prefijos), "títulos únicos": len(catalogo), "caminos del trie": len(generables),
            "tokens del título más largo": max(tokens),
            "títulos con más de 64 tokens (corte de D3)": sum(t > 64 for t in tokens),
            f"títulos con más de {MAX_NEW_TOKENS} tokens": sum(t > MAX_NEW_TOKENS for t in tokens),
            "títulos con 'Response:'": sum("Response:" in t for t in catalogo)}


def estimar(n_train, n_valid, n_test, tokens_por_fila, medicion):
    """Horas de SFT y de evaluación de un dataset, escalando lo medido en la prueba rápida por los tokens por fila.

    medicion: {"s_por_paso", "s_valid_por_fila", "tokens_por_fila", "s_test_por_fila"} de la prueba rápida.
    La generación no se escala: depende más del largo de los títulos que del prompt.
    """
    escala = tokens_por_fila / medicion["tokens_por_fila"]
    pasos = math.ceil(n_train / BATCH_SIZE)
    epoca = pasos * medicion["s_por_paso"] * escala + n_valid * medicion["s_valid_por_fila"] * escala
    return {"pasos por época": pasos, "h por época": epoca / 3600, f"h SFT ({EPOCAS} épocas)": EPOCAS * epoca / 3600,
            "h evaluación": n_test * medicion["s_test_por_fila"] / 3600}
