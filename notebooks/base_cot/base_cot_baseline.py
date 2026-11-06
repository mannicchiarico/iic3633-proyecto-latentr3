"""Evaluación de los baselines Base y COT de LatentR3 (Tabla 1) con Qwen2.5-1.5B sin ajustar (lo usa base_cot.ipynb).

- Base: el prompt de D3/LatentR3 sin token latente (el mismo de BIGRec) sobre el modelo base, con la misma
  decodificación restringida al catálogo que BIGRec (10 beams, trie).
- COT: versión zero-shot de Rec-SAVER adaptada a la recomendación del siguiente ítem. Mismo prompt de D3 más el
  rating de cada ítem, las frases de razonamiento de la Tabla 12 de Rec-SAVER y una sección "### Reason: ".
  Etapa 1: razonamiento greedy (beam 1). Etapa 2: título restringido al catálogo con 10 beams, condicionado al
  prompt más el razonamiento.

Reutiliza sin modificar el código de bigrec_baseline.py (trie, generación, métricas y verificación con calc.py).
Las decisiones y sus fuentes están en README.md.
"""
import os
import sys
import ast
import json
import math
import time
import numpy as np
import pandas as pd
import torch

_aqui = os.path.dirname(os.path.abspath(__file__))
for _ruta in [os.path.join(_aqui, '..'), os.path.join(_aqui, '..', 'bigrec')]:   # baseline_comun y bigrec_baseline
    if os.path.abspath(_ruta) not in sys.path:
        sys.path.insert(0, os.path.abspath(_ruta))
import bigrec_baseline as bb   # noqa: E402

BASE_MODEL = bb.BASE_MODEL          # Qwen/Qwen2.5-1.5B base (paper, App. B)
LENGTH_PENALTY = bb.LENGTH_PENALTY["bigrec"]   # 1.0, como BIGRec
FILAS_POR_PARTE = bb.FILAS_POR_PARTE
BATCH_BASE = bb.BATCH_GENERACION    # 24, como BIGRec
BATCH_RAZON = 64                    # etapa 1 (greedy); fijo para que el razonamiento no dependa del lote
BATCH_TITULO = 12                   # etapa 2 (10 beams); prompts de ~410 tokens con el razonamiento, por eso 12 y no 24
MAX_TOKENS_RAZON = 256              # tope del razonamiento (decisión del proyecto)
MARCA = "###"                       # el razonamiento termina cuando el modelo abre otra sección
FILAS_TEST = {"CDs_and_Vinyl": 6157, "Musical_Instruments": 8313, "Toys_and_Games": 6738, "Video_Games": 9397}

# Texto de D3 (DecodingMatters@585c842, code/dataset.py), copiado tal cual, con sus erratas
PREAMBULO = ("Below is an instruction that describes a task, paired with an input that provides further context. "
             "Write a response that appropriately completes the request. \n\n### Instruction:\n")
# Frases de razonamiento: Tabla 12 de Rec-SAVER adaptada
RAZONAMIENTO = ("User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. "
                "What information can you infer about the user's preferences? "
                "Your reasoning explanation should be based on any commonalities in the user history items and "
                "inferred user tastes or preferences. After your reasoning, write the title of the new item.")
SECCION_RAZON = "### Reason: \n"
SECCION_RESPUESTA = "### Response: \n"   # raíz del trie de BIGRec (4 tokens)


# ---------- Prompts ----------

def instruccion_d3(categoria):
    return f"Given a list of {bb.CATEGORY_DICT[categoria]} the user recetenly enjoy, please write a new " \
           f"{bb.CATEGORY_DICT[categoria]} that the user may bought"


def formatear_rating(r):
    r = float(r)
    return str(int(r)) if r.is_integer() else str(r)


def textos_cot(fila, categoria, razonar=True, ratings=True):
    """Instrucción y entrada del prompt de COT para una fila del CSV. Usa solo las columnas history_*.

    Con razonar=False y ratings=False devuelve exactamente el prompt de D3 (lo comprueba verificar_alineacion).
    """
    palabra = bb.CATEGORY_DICT[categoria]
    titulos = ast.literal_eval(fila["history_item_title"])
    if ratings:
        valores = ast.literal_eval(fila["history_rating"])
        assert len(valores) == len(titulos)
        historial = ", ".join(f"\"{t}\" (rating: {formatear_rating(r)})" for t, r in zip(titulos, valores))
    else:
        historial = ", ".join(f"\"{t}\"" for t in titulos)
    instruccion = PREAMBULO + instruccion_d3(categoria) + "\n" + (RAZONAMIENTO + "\n" if razonar else "")
    entrada = f"The user has palyed the following {palabra}s before: {historial}"
    prompt = "### User Input: \n" + entrada + "\n\n" + (SECCION_RAZON if razonar else SECCION_RESPUESTA)
    return instruccion, entrada, prompt


def codificar(tokenizer, texto):
    # Como Tokenizer.encode de D3 con bos=False, eos=False (Qwen no agrega tokens especiales)
    ids = tokenizer.encode(texto)
    while ids and tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    while ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    return ids


def leer_filas(archivo, sample=-1, seed=0):
    # Mismas filas y mismo orden que D3Dataset
    datos = pd.read_csv(archivo)
    if sample > 0:
        datos = datos.sample(sample, random_state=seed)
    return datos


def construir_cot(tokenizer, archivo, categoria, sample=-1, razonar=True, ratings=True):
    """Encodings de la etapa 1 y filas con el formato de D3Dataset.get_all() (input, output)."""
    encodings, filas = [], []
    for _, fila in leer_filas(archivo, sample).iterrows():
        instruccion, entrada, prompt = textos_cot(fila, categoria, razonar, ratings)
        ids = codificar(tokenizer, instruccion) + codificar(tokenizer, prompt)   # dos partes, como D3
        encodings.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        filas.append({"input": entrada, "output": "\"" + str(fila["item_title"]) + "\"\n"})
    return encodings, filas


def construir_base(tokenizer, archivo, categoria, sample=-1):
    """Encodings y filas de Base: D3Dataset sin modificar, como BIGRec."""
    test = bb.construir_dataset(tokenizer, archivo, categoria, sample=sample, test=True)
    filas = [{k: v for k, v in f.items() if k != "dedup"} for f in test.get_all()]
    return [test[i] for i in range(len(test))], filas


def raiz_trie(tokenizer, info_file):
    raices = {tuple(ids[:4]) for ids in bb.tokenizar_titulos(info_file, tokenizer)}
    assert len(raices) == 1
    return list(raices.pop())


def ids_etapa2(prompt_ids, razonamiento, tokenizer, raiz):
    """Prompt de la etapa 2: prompt de COT + razonamiento + '\\n\\n### Response: \\n'.

    El razonamiento y el salto de línea se tokenizan juntos y la sección de respuesta aparte, así que los últimos
    4 tokens son siempre la raíz del trie.
    """
    respuesta = codificar(tokenizer, SECCION_RESPUESTA)
    assert respuesta == raiz, (respuesta, raiz)
    ids = prompt_ids + codificar(tokenizer, razonamiento.strip() + "\n\n") + respuesta
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


# ---------- Generación ----------

def preparar_tokenizer(ruta=BASE_MODEL):
    tokenizer = bb.cargar_tokenizer(ruta)
    tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


@torch.no_grad()
def razonar(modelo, tokenizer, encodings, max_new_tokens=MAX_TOKENS_RAZON):
    """Etapa 1 de COT: greedy desde '### Reason: \\n' hasta '###', EOS o max_new_tokens.

    Devuelve por fila el razonamiento recortado antes de '###', los tokens generados antes del EOS (incluye el
    token que contiene la marca) y el motivo de fin: 'marca', 'eos' o 'tope'.
    """
    from transformers import GenerationConfig
    largo = max(len(e["input_ids"]) for e in encodings)
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    ids = [[pad] * (largo - len(e["input_ids"])) + e["input_ids"] for e in encodings]
    mascara = [[0] * (largo - len(e["input_ids"])) + e["attention_mask"] for e in encodings]
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, pad_token_id=pad,
                              eos_token_id=eos, stop_strings=[MARCA])
    salida = modelo.generate(torch.tensor(ids, device=modelo.device),
                             attention_mask=torch.tensor(mascara, device=modelo.device),
                             generation_config=config, tokenizer=tokenizer)
    resultados = []
    for generados in salida[:, largo:].tolist():
        # Al terminar, generate rellena con pad (= EOS), así que lo generado llega hasta el primer EOS
        con_eos = eos in generados
        if con_eos:
            generados = generados[:generados.index(eos)]
        texto = tokenizer.decode(generados, skip_special_tokens=False)
        if MARCA in texto:
            texto, fin = texto[:texto.index(MARCA)], "marca"
        else:
            fin = "eos" if con_eos else "tope"
        resultados.append({"reasoning": texto.strip(), "tokens_generados": len(generados), "fin": fin})
    return resultados


def por_lotes(funcion, items, batch, registro, nombre):
    """Aplica `funcion` por lotes. Si un lote no cabe en la GPU, lo divide a la mitad y lo anota en el registro."""
    salida, i = [], 0
    while i < len(items):
        lote = items[i:i + batch]
        try:
            salida += funcion(lote)
            i += len(lote)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            registro.setdefault("lotes_reducidos", []).append({"etapa": nombre, "fila": i, "nuevo_lote": batch})
            print(f"{nombre}: sin memoria; lote reducido a {batch} desde la fila {i}")
    return salida


# ---------- Evaluación ----------

def evaluar(categoria, base, salida, metodo, sample=-1, max_tokens_razon=MAX_TOKENS_RAZON, batch_base=BATCH_BASE,
            batch_razon=BATCH_RAZON, batch_titulo=BATCH_TITULO, filas_por_parte=FILAS_POR_PARTE, carpeta=None,
            modelo=None, tokenizer=None):
    """Top-10 de cada fila de test con Base o COT, guardado por partes en `salida/predicciones/<carpeta>/`.

    Se salta las partes ya escritas, así que se puede retomar; falla si la configuración guardada es otra. Al final
    une las partes en `final_result.json`, con el formato que lee calc.py. Se puede pasar un modelo ya cargado.
    """
    assert metodo in ("base", "cot")
    dir_pred = os.path.join(salida, "predicciones", carpeta or metodo)
    final = os.path.join(dir_pred, "final_result.json")
    if os.path.isfile(final):
        print(f"{categoria} ({carpeta or metodo}): ya evaluado en {final}; se salta.")
        return final
    os.makedirs(dir_pred, exist_ok=True)
    tokenizer = tokenizer or preparar_tokenizer()
    archivo = bb.buscar_archivo(base, "test", categoria)
    info = bb.buscar_archivo(base, "info", categoria)
    if metodo == "base":
        encodings, filas = construir_base(tokenizer, archivo, categoria, sample)
    else:
        encodings, filas = construir_cot(tokenizer, archivo, categoria, sample)
    trie, raiz = bb.construir_trie(info, tokenizer), raiz_trie(tokenizer, info)

    config = {"metodo": metodo, "sample": sample, "filas_por_parte": filas_por_parte, "modelo": BASE_MODEL,
              "length_penalty": LENGTH_PENALTY, "num_beams": bb.NUM_BEAMS, "max_new_tokens": bb.MAX_NEW_TOKENS}
    config.update({"batch": batch_base} if metodo == "base" else
                  {"batch_razon": batch_razon, "batch_titulo": batch_titulo, "max_tokens_razon": max_tokens_razon})
    ruta_registro = os.path.join(dir_pred, "registro.json")
    registro = {"config": config, "filas": 0, "segundos": 0.0, "segundos_razon": 0.0, "segundos_titulo": 0.0}
    if os.path.isfile(ruta_registro):
        with open(ruta_registro) as f:
            registro = json.load(f)
        if registro["config"] != config:
            raise ValueError(f"{dir_pred} se generó con otra configuración: {registro['config']}. "
                             "Usar otra `carpeta` o borrar esa.")

    propio = modelo is None
    n_partes = math.ceil(len(encodings) / filas_por_parte)
    for p in range(n_partes):
        ruta = os.path.join(dir_pred, f"parte_{p:03d}.json")
        if os.path.isfile(ruta):
            continue
        if modelo is None:
            modelo = bb.cargar_modelo(BASE_MODEL, tokenizer)
        torch.cuda.reset_peak_memory_stats()
        ini, fin = p * filas_por_parte, min((p + 1) * filas_por_parte, len(encodings))
        parte = [dict(f) for f in filas[ini:fin]]
        inicio = time.time()
        if metodo == "base":
            predicciones = por_lotes(lambda lote: bb.generar(modelo, tokenizer, lote, trie, LENGTH_PENALTY),
                                     encodings[ini:fin], batch_base, registro, "base")
            s_razon = 0.0
        else:
            razones = por_lotes(lambda lote: razonar(modelo, tokenizer, lote, max_tokens_razon),
                                encodings[ini:fin], batch_razon, registro, "razon")
            s_razon = time.time() - inicio
            etapa2 = [ids_etapa2(e["input_ids"], r["reasoning"], tokenizer, raiz)
                      for e, r in zip(encodings[ini:fin], razones)]
            predicciones = por_lotes(lambda lote: bb.generar(modelo, tokenizer, lote, trie, LENGTH_PENALTY),
                                     etapa2, batch_titulo, registro, "titulo")
            for fila, r, e2 in zip(parte, razones, etapa2):
                fila.update(r)
                fila["tokens_etapa2"] = len(e2["input_ids"])
        segundos = time.time() - inicio
        for fila, prediccion in zip(parte, predicciones):
            fila["predict"] = prediccion
        with open(ruta + ".tmp", "w") as f:
            json.dump(parte, f, indent=1)
        os.replace(ruta + ".tmp", ruta)   # una parte a medio escribir nunca queda con el nombre final
        registro["filas"] += fin - ini
        registro["segundos"] += segundos
        registro["segundos_razon"] += s_razon
        registro["segundos_titulo"] += segundos - s_razon
        registro["memoria_max_gb"] = max(registro.get("memoria_max_gb", 0), torch.cuda.max_memory_allocated() / 2**30)
        with open(ruta_registro, "w") as f:
            json.dump(registro, f, indent=1)
        print(f"{categoria} ({carpeta or metodo}): parte {p + 1}/{n_partes}, {segundos / (fin - ini):.2f} s por fila")

    datos = []
    for p in range(n_partes):   # como merge.py de los autores
        with open(os.path.join(dir_pred, f"parte_{p:03d}.json")) as f:
            datos.extend(json.load(f))
    assert len(datos) == len(encodings)
    with open(final, "w") as f:
        json.dump(datos, f, indent=1)
    if propio and modelo is not None:
        del modelo
        torch.cuda.empty_cache()
    return final


def leer_registro(final_result):
    with open(os.path.join(os.path.dirname(final_result), "registro.json")) as f:
        return json.load(f)


# ---------- Estadísticas del razonamiento ----------

def estadisticas_razon(final_result, topes=(128, 256, 512)):
    """Cómo terminan los razonamientos y cuántos se habrían cortado con cada tope.

    Con greedy, el razonamiento con un tope menor es el prefijo del de uno mayor (V6), así que una fila se habría
    cortado con tope t si generó más de t tokens antes de la marca o del EOS.
    """
    with open(final_result) as f:
        datos = json.load(f)
    tokens = np.array([d["tokens_generados"] for d in datos])
    fines = pd.Series([d["fin"] for d in datos]).value_counts(normalize=True)
    tope_usado = leer_registro(final_result)["config"]["max_tokens_razon"]
    resultado = {f"fin: {k}": fines.get(k, 0.0) for k in ("marca", "eos", "tope")}
    resultado.update({"vacíos": np.mean([d["reasoning"] == "" for d in datos]),
                      "tokens generados (media)": tokens.mean(), "tokens generados (mediana)": np.median(tokens),
                      "tokens generados (p90)": np.percentile(tokens, 90),
                      "tokens etapa 2 (media)": np.mean([d["tokens_etapa2"] for d in datos])})
    for t in topes:
        if t <= tope_usado:
            resultado[f"cortados con tope {t}"] = float(np.mean(tokens > t) if t < tope_usado
                                                        else np.mean([d["fin"] == "tope" for d in datos]))
    return resultado


def ejemplos(final_result, n=3, seed=0):
    """n filas al azar con el historial, el razonamiento, el top-3 y el objetivo."""
    with open(final_result) as f:
        datos = json.load(f)
    rng = np.random.default_rng(seed)
    salida = []
    for i in rng.choice(len(datos), size=min(n, len(datos)), replace=False):
        d = datos[int(i)]
        salida.append(f"--- Fila {int(i)} ({d['fin']}, {d['tokens_generados']} tokens)\n"
                      f"Historial: {d['input']}\nRazonamiento:\n{d['reasoning']}\n"
                      f"Top-3: {[p.strip() for p in d['predict'][:3]]}\nObjetivo: {d['output'].strip()}")
    return "\n\n".join(salida)


def estimar_horas(s_por_fila, filas=FILAS_TEST):
    return {cat: n * s_por_fila / 3600 for cat, n in filas.items()}


# ---------- Verificaciones sin GPU ----------

def verificar_alineacion(base, categoria, tokenizer, filas=200):
    """V2: sin razonamiento ni ratings, el prompt de COT debe dar los mismos tokens que D3Dataset.

    Se comprueba en `filas` filas de train, valid y test. Así, lo único que COT agrega a Base es lo marcado.
    """
    for split in ("train", "valid", "test"):
        archivo = bb.buscar_archivo(base, split, categoria)
        d3, _ = construir_base(tokenizer, archivo, categoria, sample=filas)
        cot, _ = construir_cot(tokenizer, archivo, categoria, sample=filas, razonar=False, ratings=False)
        assert len(d3) == len(cot)
        for i, (a, b) in enumerate(zip(d3, cot)):
            assert a["input_ids"] == b["input_ids"], (split, i)
    return True


def verificar_etapa2(base, categoria, tokenizer, filas=200):
    """V3: el prompt de la etapa 2 termina en la raíz del trie con razonamientos de distintas formas, y el trie
    recorrido desde esa raíz da exactamente el catálogo (bb.verificar_trie)."""
    info = bb.buscar_archivo(base, "info", categoria)
    raiz = raiz_trie(tokenizer, info)
    encodings, _ = construir_cot(tokenizer, bb.buscar_archivo(base, "test", categoria), categoria, sample=filas)
    sinteticos = ["", "The user likes puzzles.", "The user likes puzzles.\n", "Line one.\n\nLine two\n\n",
                  "ends with space ", "\"A quoted title\"", "  leading spaces"]
    etapa2 = [ids_etapa2(e["input_ids"], r, tokenizer, raiz) for e in encodings for r in sinteticos]
    assert all(e["input_ids"][-4:] == raiz for e in etapa2)
    return bb.verificar_trie(info, tokenizer, etapa2)


def contar_recompras(base, categoria):
    """V4: filas de test cuyo título objetivo ya está en el historial (recompras, legítimas también en Base), y
    filas sin recompra en las que el título objetivo entre comillas aparece igual en el prompt de COT (debe ser 0;
    solo podría pasar si un título del historial contiene al objetivo entre comillas)."""
    datos = leer_filas(bb.buscar_archivo(base, "test", categoria))
    recompras = apariciones = 0
    for _, fila in datos.iterrows():
        objetivo = str(fila["item_title"])
        if objetivo in ast.literal_eval(fila["history_item_title"]):
            recompras += 1
            continue
        instruccion, _, prompt = textos_cot(fila, categoria)
        apariciones += f"\"{objetivo}\"" in instruccion + prompt
    return {"filas": len(datos), "objetivo en el historial": recompras, "porcentaje": recompras / len(datos),
            "objetivo en el prompt sin recompra": apariciones}
