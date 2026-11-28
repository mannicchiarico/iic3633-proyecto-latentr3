"""LatentR3-SFT + SigLIP en Toys (M5; lo usa latentr3_mm.ipynb). Extensión propia, no es parte del paper.

Mismo LatentR3 que ../latentr3/latentr3.py (mismo SFT, misma evaluación y, opcional, el mismo RL), con una imagen por
ítem del historial:
- Prompt: después de cada título del historial va el marcador `<|image_pad|>`, un token que ya existe en el
  vocabulario de Qwen2.5 (id 151655). No se agrega un token nuevo, porque el código de los autores identifica el token
  latente como el último del vocabulario (autores/model.py `generate_embs`; LogitProcesser.py `get_prefix_data`).
- Modelo: el embedding de cada marcador se reemplaza por `proyector(SigLIP_imagen(ítem))` (lineal 1152 → 1536) o, si
  el ítem no tiene imagen, por un vector aprendido `sin_imagen`, compartido. Se hace en las dos pasadas de LatentModel,
  así que LatentRATT también ve las imágenes. Es el mecanismo que los autores usan para `<|Thought|>` y el que usa
  Qwen2-VL con `<|image_pad|>` (masked_scatter de las features visuales).
- Embeddings: `siglip.npz` de Toys (SigLIP so400m-patch14-384, 6299 × 1152, norma 1), calculado en Colab por M
  (multimodal.ipynb); sus filas coinciden con `item_id`.

Decisiones, fuentes y verificaciones: README.md (sección "LatentR3-SFT + SigLIP").
"""
import os
import sys
import json
import shutil
import hashlib
import functools
import numpy as np
import torch

import latentr3 as lr3
from bigrec_baseline import CUTOFF_LEN, CATEGORY_DICT, buscar_archivo

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CATEGORIA = "Toys_and_Games"
TOKEN_IMAGEN = "<|image_pad|>"     # vocabulario de Qwen2.5 (convención de Qwen2-VL); sin uso en el modelo de texto
ID_IMAGEN = 151655
DIM_SIGLIP = 1152
MAX_HISTORIA = 10                   # largo máximo del historial en los CSV del repo
RUTA_NPZ = os.path.join(RAIZ, "checkpoints", "multimodal", CATEGORIA, "siglip.npz")
SHA256_NPZ = "e0bed3e088e0ad1bfb5bbf6fd2bdd003878fb5bd6d4ffb96b7424ab7bd9573f1"   # manifiesto_embeddings.json
SALIDA = os.path.join(RAIZ, "checkpoints", "latentr3_mm", CATEGORIA)
RESULTADOS = os.path.join(RAIZ, "resultados", "multimodal", CATEGORIA, "latentr3_mm")
UMBRAL_POPULARIDAD = 5              # el mismo de sasrec_mm.py

LatentModelMM = None
_CLASES_DATOS = {}


# ---------- Embeddings SigLIP ----------

@functools.lru_cache(maxsize=None)
def cargar_features(ruta=RUTA_NPZ):
    """(imagen [n, 1152] fp32, tiene_imagen [n] bool), verificando el sha256 del npz."""
    with open(ruta, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != SHA256_NPZ:
        raise ValueError(f"{ruta} no es el siglip.npz revisado (sha256 {sha}).")
    with np.load(ruta) as datos:
        imagen, tiene = datos["imagen"].astype(np.float32), datos["tiene_imagen"].astype(bool)
    assert imagen.shape == (len(tiene), DIM_SIGLIP)
    return imagen, tiene


# ---------- Datos ----------

class _MixinMM:
    """Cambia el prompt de LatentRDataset (SFT) y de D3Dataset (RL), que tienen el mismo get_history y pre: inserta
    ` <|image_pad|>` después de cada título del historial y agrega `item_ids` (ids del historial, rellenados con −1)."""

    def get_history(self, row):
        # get_history original (reasoning_dataset.py y latent_grpo_dataset.py) con el marcador después de cada título
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title'])
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\" " + TOKEN_IMAGEN
            else:
                history += ", \"" + row['history_item_title'][i] + "\" " + TOKEN_IMAGEN
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}

    def pre(self, idx):
        salida = super().pre(idx)
        ids = [int(i) for i in eval(self.data.iloc[idx]["history_item_id"])]
        if len(ids) > MAX_HISTORIA:
            raise ValueError(f"historial de {len(ids)} ítems (máximo {MAX_HISTORIA})")
        if not self.test and len(salida["input_ids"]) >= self.max_len:
            # pre trunca por la izquierda: se perderían marcadores y quedarían descuadrados con los ids
            raise ValueError(f"fila {idx}: {len(salida['input_ids'])} tokens, se truncaría a {self.max_len}")
        marcadores = sum(t == ID_IMAGEN for t in salida["input_ids"])
        if marcadores != len(ids):
            raise ValueError(f"fila {idx}: {marcadores} marcadores de imagen y {len(ids)} ítems en el historial")
        salida["item_ids"] = ids + [-1] * (MAX_HISTORIA - len(ids))
        return salida


def _clase_datos(nombre):
    if nombre not in _CLASES_DATOS:
        if lr3.LatentRDataset is None:
            raise RuntimeError("Llamar antes a lr3.cargar_autores(latentr3).")
        padre = {"sft": lr3.LatentRDataset, "rl": lr3.D3DatasetRL}[nombre]
        _CLASES_DATOS[nombre] = type(f"{padre.__name__}MM", (_MixinMM, padre), {})
    return _CLASES_DATOS[nombre]


def dataset_mm(tokenizer, archivo, categoria, sample=-1, test=False, seed=0):
    """Como lr3.dataset_sft, con el prompt multimodal."""
    return _clase_datos("sft")(train_file=archivo, tokenizer=tokenizer, max_len=2560 if test else CUTOFF_LEN,
                               sample=sample, test=test, seed=seed, category=CATEGORY_DICT[categoria], K=0)


def dataset_rl_mm(tokenizer, archivo, categoria, sample=-1, seed=lr3.SEED, test=False):
    """Como lr3.dataset_rl, con el prompt multimodal."""
    return _clase_datos("rl")(train_file=archivo, tokenizer=tokenizer, max_len=2560 if test else CUTOFF_LEN,
                              sample=sample, seed=seed, category=CATEGORY_DICT[categoria], test=test)


# ---------- Modelo ----------

def clase_mm():
    """LatentModel (copia parcheada de los autores) con el proyector, el vector `sin_imagen` y los embeddings SigLIP."""
    global LatentModelMM
    if lr3.LatentModel is None:
        raise RuntimeError("Llamar antes a lr3.cargar_autores(latentr3).")
    if LatentModelMM is not None and LatentModelMM.__bases__[0] is lr3.LatentModel:
        return LatentModelMM
    import torch.nn as nn
    from transformers import Qwen2ForCausalLM

    class _LatentModelMM(lr3.LatentModel):
        def __init__(self, config):
            super().__init__(config)
            # Lineal como LLaVA y como W·f de SASRec + SigLIP; nn.Embedding para que _init_weights de Qwen2 lo
            # inicialice (normal, std 0.02) al cargar desde el modelo base
            self.proyector = nn.Linear(config.mm_dim, config.hidden_size)
            self.sin_imagen = nn.Embedding(1, config.hidden_size)
            # Persistentes: quedan guardados con el modelo. Se vuelven a llenar desde el npz al cargar (asignar_features)
            self.register_buffer("feats", torch.zeros(config.mm_items, config.mm_dim, dtype=torch.float32))
            self.register_buffer("tiene", torch.zeros(config.mm_items, dtype=torch.bool))
            self._item_ids = None   # lo fija el trainer del RL antes de llamar a generate_embs

        def vectores_imagen(self, ids):
            """Vector de entrada de cada ítem de `ids` (1-D): proyector(SigLIP) o `sin_imagen`."""
            v = self.proyector(self.feats[ids].to(self.proyector.weight.dtype))
            return torch.where(self.tiene[ids].unsqueeze(-1), v, self.sin_imagen.weight[0].to(v.dtype))

        def embeddings_con_imagenes(self, input_ids, item_ids):
            embs = self.model.embed_tokens(input_ids)
            pos = torch.nonzero(input_ids == self.config.mm_token_id)   # en orden por filas, como el historial
            ids = item_ids.to(input_ids.device)
            ids = ids[ids >= 0]                                         # también en orden por filas
            if pos.size(0) != ids.numel():
                raise ValueError(f"{pos.size(0)} marcadores de imagen y {ids.numel()} item_ids")
            if pos.size(0) == 0:
                return embs
            return embs.index_put((pos[:, 0], pos[:, 1]), self.vectores_imagen(ids).to(embs.dtype))

        def generate_embs(self, input_ids, attention_mask, item_ids=None):
            # generate_embs de autores/model.py (con P2), con las imágenes también en la pasada 1. El latente se escribe
            # fuera de lugar: los embeddings de la pasada 1 están en el grafo y no se pueden modificar in situ
            if item_ids is None:
                item_ids = self._item_ids
            if item_ids is None:
                raise ValueError("Faltan los item_ids del historial.")
            embs = self.embeddings_con_imagenes(input_ids, item_ids)
            salida = Qwen2ForCausalLM.forward(self, inputs_embeds=embs, attention_mask=attention_mask,
                                              output_hidden_states=True, logits_to_keep=1)
            latente_id = self.model.embed_tokens.num_embeddings - 1
            donde = torch.nonzero(input_ids == latente_id)
            latente = self.attention(hidden_states=salida["hidden_states"][-1], attention_mask=attention_mask,
                                     thought_id_idx=donde[:, 1]).to(embs.dtype)
            return embs.index_put((donde[:, 0], donde[:, 1]), latente)

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                    output_hidden_states=None, return_dict=None, cache_position=None, logits_to_keep=0,
                    item_ids=None, **kwargs):
            # Igual que LatentModel.forward: solo al procesar el prompt completo (en la generación, el primer paso);
            # los pasos siguientes usan la caché y solo agregan tokens del título
            if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
                inputs_embeds = self.generate_embs(input_ids, attention_mask, item_ids)
                input_ids = None
            return Qwen2ForCausalLM.forward(
                self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict, cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)

    _LatentModelMM.__name__ = _LatentModelMM.__qualname__ = "LatentModelMM"
    LatentModelMM = _LatentModelMM
    return LatentModelMM


def asignar_features(modelo, ruta_npz=RUTA_NPZ, sin_imagenes=False):
    """Llena los buffers desde el npz (fp32). Con sin_imagenes=True todos los ítems usan `sin_imagen` (ablación)."""
    imagen, tiene = cargar_features(ruta_npz)
    dispositivo = modelo.proyector.weight.device
    modelo.feats = torch.from_numpy(imagen).to(dispositivo)
    modelo.tiene = torch.zeros(len(tiene), dtype=torch.bool, device=dispositivo) if sin_imagenes \
        else torch.from_numpy(tiene).to(dispositivo)
    return modelo


def modelo_base_mm(tokenizer, ruta=lr3.BASE_MODEL, ruta_npz=RUTA_NPZ):
    """Qwen2.5-1.5B + LatentRATT + proyector y `sin_imagen` recién inicializados (como lr3.modelo_base_latente)."""
    from transformers import AutoConfig
    imagen, _ = cargar_features(ruta_npz)
    config = AutoConfig.from_pretrained(ruta)
    config.mm_dim, config.mm_items, config.mm_token_id = DIM_SIGLIP, imagen.shape[0], ID_IMAGEN
    modelo = clase_mm().from_pretrained(ruta, config=config, torch_dtype=torch.bfloat16)
    modelo.attention.end_k = -1
    modelo.config.loss_type = "ce"
    modelo.resize_token_embeddings(len(tokenizer))
    assert modelo.model.embed_tokens.num_embeddings - 1 == tokenizer.convert_tokens_to_ids(lr3.TOKEN_LATENTE)
    assert tokenizer.convert_tokens_to_ids(TOKEN_IMAGEN) == ID_IMAGEN
    for nombre, p in [("proyector", modelo.proyector.weight), ("sin_imagen", modelo.sin_imagen.weight)]:
        std = p.float().std().item()
        assert torch.isfinite(p).all() and 0.01 < std < 0.03, (nombre, std)   # _init_weights de Qwen2: std 0.02
    return asignar_features(modelo, ruta_npz)


def _revisar_buffer(modelo, ruta_npz):
    # El buffer guardado debe ser el del npz (tolerancia de bf16 por si se convirtió al cargar)
    imagen, tiene = cargar_features(ruta_npz)
    assert torch.allclose(modelo.feats.float().cpu(), torch.from_numpy(imagen), atol=1e-2)
    assert torch.equal(modelo.tiene.cpu(), torch.from_numpy(tiene))


def cargar_modelo_mm(dir_modelo, tokenizer, sin_imagenes=False, ruta_npz=RUTA_NPZ):
    """Modelo entrenado listo para generar (como lr3.cargar_modelo_latente)."""
    modelo = clase_mm().from_pretrained(dir_modelo, torch_dtype=torch.bfloat16).to(lr3.device)
    _revisar_buffer(modelo, ruta_npz)
    asignar_features(modelo, ruta_npz, sin_imagenes)
    modelo.attention.end_k = -1
    modelo.config.pad_token_id = modelo.config.eos_token_id = tokenizer.eos_token_id
    modelo.config.bos_token_id = tokenizer.bos_token_id
    modelo.config.use_cache = True
    return modelo.eval()


def modelo_rl_mm(dir_sft, ruta_npz=RUTA_NPZ):
    """Modelo del SFT multimodal para el RL (como lr3._modelo_para_rl)."""
    modelo = clase_mm().from_pretrained(dir_sft, torch_dtype=torch.bfloat16)
    _revisar_buffer(modelo, ruta_npz)
    asignar_features(modelo, ruta_npz)
    modelo.attention.end_k = -1
    return modelo


def congelar_mm(modelo):
    """Como lr3.congelar_llm: el RL solo entrena LatentRATT (el proyector y `sin_imagen` también quedan fijos)."""
    for nombre, p in modelo.named_parameters():
        p.requires_grad = nombre.startswith("attention.")
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def trainer_mm():
    """NoiseGRPORecTrainer (copia parcheada) que pasa los item_ids de cada grupo al modelo.

    El trainer llama a self.model.generate_embs(ids, mask) en _puntuar_grupo (L164) y compute_loss (L243), sin
    item_ids: se dejan en model._item_ids antes de llamar a esos métodos sin modificarlos. Cada grupo son 8 copias del
    mismo prompt, así que sus item_ids son 8 filas iguales; swap_adjacent_blocks (P3) solo reordena las recompensas.
    """
    class NoiseGRPORecTrainerMM(lr3.NoiseGRPORecTrainer):
        def _puntuar_grupo(self, prompt_inputs, device):
            self.model._item_ids = prompt_inputs["item_ids"]
            try:
                salida, recompensas = super()._puntuar_grupo(prompt_inputs, device)
            finally:
                self.model._item_ids = None
            salida["item_ids"] = prompt_inputs["item_ids"]
            return salida, recompensas

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self.model._item_ids = inputs["item_ids"]
            try:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            finally:
                self.model._item_ids = None

    return NoiseGRPORecTrainerMM


# ---------- Etapas (mismas funciones de latentr3.py) ----------

def rutas(salida=SALIDA, lr_rl=lr3.LR_RL):
    r = lr3.rutas(salida, lr_rl)
    r["pred_sin_imagenes"] = os.path.join(r["sft"], "predicciones", "test_sin_imagenes")
    return r


def entrenar_sft(base, salida_sft, ruta_npz=RUTA_NPZ, **kwargs):
    return lr3.entrenar_sft(CATEGORIA, base, salida_sft, crear_dataset=dataset_mm,
                            crear_modelo=lambda tok, ruta: modelo_base_mm(tok, ruta, ruta_npz), **kwargs)


def evaluar(base, dir_modelo, dir_pred, batch_size, length_penalty, sin_imagenes=False, ruta_npz=RUTA_NPZ, **kwargs):
    return lr3.evaluar_categoria(CATEGORIA, base, dir_modelo, dir_pred, batch_size, length_penalty,
                                 crear_dataset=dataset_mm,
                                 cargar_modelo=lambda d, tok: cargar_modelo_mm(d, tok, sin_imagenes, ruta_npz), **kwargs)


def entrenar_rl(base, salida_rl, dir_sft, ruta_npz=RUTA_NPZ, **kwargs):
    return lr3.entrenar_rl(CATEGORIA, base, salida_rl, dir_sft, crear_dataset=dataset_rl_mm,
                           cargar_modelo=lambda d: modelo_rl_mm(d, ruta_npz), congelar=congelar_mm,
                           clase_trainer=trainer_mm(), **kwargs)


# ---------- Verificaciones ----------

def verificar_token(base, tokenizer):
    """VM1: el marcador es un token existente y no altera el token latente ni el trie."""
    assert tokenizer(TOKEN_IMAGEN)["input_ids"] == [ID_IMAGEN]
    latente = tokenizer.convert_tokens_to_ids(lr3.TOKEN_LATENTE)
    assert latente == len(tokenizer) - 1 and ID_IMAGEN < latente
    info = buscar_archivo(base, "info", CATEGORIA)
    with open(info, encoding="utf-8") as f:
        titulos = [linea.split("\t")[0] for linea in f]
    assert not any("<|" in t for t in titulos), "algún título contiene un token especial"
    trie = lr3.get_prefix_data(info, tokenizer)
    assert not any(ID_IMAGEN in v for v in trie.values())
    return {"id marcador": ID_IMAGEN, "id latente": latente, "títulos": len(titulos), "prefijos del trie": len(trie)}


def verificar_dataset(base, tokenizer, filas=200):
    """VM2: sin los marcadores, el prompt es el de LatentRDataset y el de D3Dataset del RL; marcadores = item_ids."""
    imagen, _ = cargar_features()
    salida = {}
    for split in ["train", "valid", "test"]:
        archivo = buscar_archivo(base, split, CATEGORIA)
        test = split == "test"
        for nombre, mm, original in [
                ("SFT", dataset_mm(tokenizer, archivo, CATEGORIA, sample=filas, test=test),
                 lr3.dataset_sft(tokenizer, archivo, CATEGORIA, sample=filas, test=test)),
                ("RL", dataset_rl_mm(tokenizer, archivo, CATEGORIA, sample=filas, test=test),
                 lr3.dataset_rl(tokenizer, archivo, CATEGORIA, sample=filas, test=test))]:
            largos = []
            for i in range(len(mm)):
                a, b = mm[i], original[i]
                texto = tokenizer.decode(a["input_ids"]).replace(" " + TOKEN_IMAGEN, "")
                assert texto == tokenizer.decode(b["input_ids"]), (split, nombre, i)
                validos = [j for j in a["item_ids"] if j >= 0]
                assert len(a["item_ids"]) == MAX_HISTORIA and max(validos) < imagen.shape[0]
                assert sum(t == ID_IMAGEN for t in a["input_ids"]) == len(validos)
                largos.append(len(a["input_ids"]))
            salida[(split, nombre)] = {"filas": len(mm), "tokens máx.": max(largos),
                                       "tokens extra por fila (media)": float(np.mean(largos) - np.mean(
                                           [len(original[i]["input_ids"]) for i in range(len(original))]))}
    return salida


def _diminuto_mm(vocab, d=64, n_items=50, dim=16, token=7, semilla=lr3.SEED):
    """LatentModelMM con un Qwen2 de 2 capas en fp32 y CPU (como lr3._diminuto), con features aleatorias."""
    from transformers import Qwen2Config
    torch.manual_seed(semilla)
    config = Qwen2Config(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         tie_word_embeddings=True, attn_implementation="eager")
    config.mm_dim, config.mm_items, config.mm_token_id = dim, n_items, token
    modelo = clase_mm()(config).float().eval()
    modelo.attention.end_k = -1
    f = torch.randn(n_items, dim)
    modelo.feats = f / f.norm(dim=-1, keepdim=True)
    modelo.tiene = torch.rand(n_items) > 0.2
    modelo.tiene[0] = False   # al menos un ítem sin imagen
    return modelo


def _lote_diminuto(vocab, token, n_items, filas=3, largo=30, semilla=1):
    """Prompts sintéticos: 2-4 marcadores de imagen, el token latente en la posición 22 y 'objetivo' después."""
    g = torch.Generator().manual_seed(semilla)
    ids = torch.randint(10, vocab - 2, (filas, largo), generator=g)
    item_ids = torch.full((filas, MAX_HISTORIA), -1)
    for i in range(filas):
        n = 2 + i % 3
        ids[i, torch.arange(n) * 4 + 2] = token
        item_ids[i, :n] = torch.randint(0, n_items, (n,), generator=g)
        item_ids[i, 0] = 0   # el ítem 0 no tiene imagen
    ids[:, 22] = vocab - 1
    return ids, torch.ones_like(ids), item_ids


def verificar_modelo(vocab=1000, d=64):
    """VM3: (a) si cada imagen es el embedding del propio marcador, LatentModelMM da los logits de LatentModel;
    (b) el gradiente llega al proyector y a `sin_imagen`, sin error por operaciones in situ."""
    mm = _diminuto_mm(vocab, d)
    token, n_items = mm.config.mm_token_id, mm.config.mm_items
    ids, mascara, item_ids = _lote_diminuto(vocab, token, n_items)
    texto = lr3.LatentModel(mm.config).float().eval()
    faltan = texto.load_state_dict(mm.state_dict(), strict=False)
    assert not faltan.missing_keys and set(faltan.unexpected_keys) == {"proyector.weight", "proyector.bias",
                                                                       "sin_imagen.weight", "feats", "tiene"}
    texto.attention.end_k = -1
    with torch.no_grad():
        original = mm.vectores_imagen
        mm.vectores_imagen = lambda x: mm.model.embed_tokens.weight[token].expand(x.numel(), -1)
        a = mm(input_ids=ids, attention_mask=mascara, item_ids=item_ids).logits
        mm.vectores_imagen = original
        b = texto(input_ids=ids, attention_mask=mascara).logits
        c = mm(input_ids=ids, attention_mask=mascara, item_ids=item_ids).logits
    dif = (a - b).abs().max().item()
    assert dif < 1e-5, dif
    assert (c - b).abs().max().item() > 1e-4   # con las imágenes de verdad, la salida cambia
    mm.train()
    perdida = mm(input_ids=ids, attention_mask=mascara, item_ids=item_ids).logits[:, 23:].logsumexp(-1).mean()
    perdida.backward()
    grads = {n: mm.get_parameter(n).grad for n in ["proyector.weight", "sin_imagen.weight", "attention.query.weight"]}
    assert all(g is not None and g.abs().sum() > 0 for g in grads.values())
    return {"logits con marcador = LatentModel (dif. máx.)": dif,
            "gradiente no nulo": list(grads)}


def verificar_generacion(dir_tmp, vocab=1000, d=64, beams=3):
    """VM4: generate entrega los item_ids replicados por beam al procesar el prompt; guardar y cargar conserva todo."""
    mm = _diminuto_mm(vocab, d)
    token, n_items = mm.config.mm_token_id, mm.config.mm_items
    ids, mascara, item_ids = _lote_diminuto(vocab, token, n_items, filas=2)
    ids, mascara = ids[:, :23], mascara[:, :23]   # el prompt termina en el token latente
    vistos = []
    original = mm.generate_embs
    mm.generate_embs = lambda i, m, it=None: (vistos.append(tuple(it.shape)), original(i, m, it))[1]
    with torch.no_grad():
        mm.generate(ids, attention_mask=mascara, item_ids=item_ids, num_beams=beams, num_return_sequences=beams,
                    max_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=1)
    mm.generate_embs = original
    assert vistos == [(2 * beams, MAX_HISTORIA)], vistos   # una sola vez, en el primer paso
    shutil.rmtree(dir_tmp, ignore_errors=True)
    mm.save_pretrained(dir_tmp)
    otro = clase_mm().from_pretrained(dir_tmp).float().eval()
    otro.attention.end_k = -1
    with torch.no_grad():
        a = mm(input_ids=ids, attention_mask=mascara, item_ids=item_ids).logits
        b = otro(input_ids=ids, attention_mask=mascara, item_ids=item_ids).logits
    assert torch.equal(otro.feats, mm.feats) and torch.equal(otro.tiene, mm.tiene)
    dif = (a - b).abs().max().item()
    assert dif < 1e-6, dif
    shutil.rmtree(dir_tmp, ignore_errors=True)
    return {"llamadas con item_ids (forma)": vistos, "logits tras guardar y cargar (dif. máx.)": dif}


def verificar_rl(base, tokenizer, salida_tmp, dim=16):
    """VM6: el trainer del RL multimodal en CPU, sobre un modelo diminuto y 2 filas reales de train.

    (a) los item_ids llegan a generate_embs en _puntuar_grupo y en compute_loss, 8 filas iguales por grupo;
    (b) tras un paso de AdamW solo cambian los tensores de attention (proyector, `sin_imagen` y LLM fijos);
    (c) el backward no falla por operaciones in situ.
    """
    lr3._reiniciar_accelerate()
    try:
        return _verificar_rl(base, tokenizer, salida_tmp, dim)
    finally:
        lr3._reiniciar_accelerate()
        shutil.rmtree(salida_tmp, ignore_errors=True)


def _verificar_rl(base, tokenizer, salida_tmp, dim):
    tok = lr3.para_generar(tokenizer)
    imagen, _ = cargar_features()
    modelo = _diminuto_mm(len(tok), n_items=imagen.shape[0], dim=dim, token=ID_IMAGEN).train()
    congelar_mm(modelo)
    datos = dataset_rl_mm(tok, buscar_archivo(base, "train", CATEGORIA), CATEGORIA, sample=2, seed=lr3.SEED)
    n = lr3.NUM_GENERATIONS
    filas = [datos[0]] * n + [datos[1]] * n
    trainer = trainer_mm()(
        prefix_allowed_tokens_fn=lambda _, ids: [], model=modelo, reward_funcs=[lr3.recompensa_exact_match],
        args=lr3.config_rl(salida_tmp, bf16=False, use_cpu=True, max_steps=1, save_strategy="no"),
        train_dataset=lr3.a_hf(datos), eval_dataset=None, processing_class=tok)
    vistos = []
    original = modelo.generate_embs
    modelo.generate_embs = lambda i, m, it=None: (vistos.append(modelo._item_ids.clone()), original(i, m, it))[1]
    lote = trainer.data_collator(filas)
    salidas = trainer._generate_and_score_completions(lote)["grupos"]
    esperados = [torch.tensor(datos[g]["item_ids"]).repeat(n, 1) for g in range(2)]
    assert len(vistos) == 2 and all(torch.equal(v.cpu(), e) for v, e in zip(vistos, esperados))
    antes = {nombre: p.detach().clone() for nombre, p in modelo.named_parameters()}
    trainer._prepare_inputs = lambda _: {"grupos": salidas}
    modelo.zero_grad()
    trainer.training_step(modelo, None)
    assert len(vistos) == 4 and all(torch.equal(v.cpu(), e) for v, e in zip(vistos[2:], esperados))
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    modelo.generate_embs = original
    cambiados = [nombre for nombre, p in modelo.named_parameters() if not torch.equal(p, antes[nombre])]
    assert cambiados and all(nombre.startswith("attention.") for nombre in cambiados), cambiados
    return {"llamadas a generate_embs con item_ids": len(vistos), "tensores cambiados tras un paso": cambiados}


# ---------- Análisis ----------

def _cargar_modulo(nombre, ruta):
    import importlib.util
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def analizar(base, predicciones, pares, semilla=0):
    """Métricas, comparación pareada (McNemar y bootstrap de analisis_pareado.py), popularidad del objetivo (≤ 5
    interacciones en train, definición de sasrec_mm.py) y fracción del historial con imagen.

    `predicciones`: {nombre: ruta de final_result.json} (todas sobre el test completo de Toys, en el orden del CSV).
    `pares`: [(nombre a, nombre b)] para la comparación pareada.
    """
    import pandas as pd
    import bigrec_baseline as bb
    carpeta = os.path.dirname(os.path.abspath(__file__))
    pareado = _cargar_modulo("analisis_pareado_mm", os.path.join(carpeta, "..", "latentr3", "analisis_pareado.py"))
    sasrec = os.path.abspath(os.path.join(carpeta, "..", "sasrec"))
    if sasrec not in sys.path:
        sys.path.insert(0, sasrec)
    import sasrec_mm
    info = buscar_archivo(base, "info", CATEGORIA)
    datos = {}
    for nombre, ruta in predicciones.items():
        with open(ruta) as f:
            datos[nombre] = json.load(f)
    test = pd.read_csv(buscar_archivo(base, "test", CATEGORIA))
    train = pd.read_csv(buscar_archivo(base, "train", CATEGORIA))
    assert all(len(d) == len(test) for d in datos.values())
    for d in datos.values():   # mismas filas y en el mismo orden que el CSV
        assert all(fila["output"].strip(" \n\"") == str(t).strip(" \n\"") for fila, t in zip(d, test["item_title"]))
    resultado = {"métricas": {nombre: bb.metricas(ruta, info) for nombre, ruta in predicciones.items()}}
    rng = np.random.default_rng(semilla)
    resultado["pareado"] = {f"{a} → {b}": pareado.comparar(a, b, datos, rng) for a, b in pares}
    # Grupos de filas
    train = train.assign(history_item_id=train["history_item_id"].map(eval))
    conteo = sasrec_mm.interacciones_train(train)
    poco = np.array([conteo.get(i, 0) <= UMBRAL_POPULARIDAD for i in test["item_id"]])
    _, tiene = cargar_features()
    historia = test["history_item_id"].map(eval)
    toda_con_imagen = np.array([all(tiene[i] for i in h) for h in historia])
    grupos = {f"objetivo con ≤{UMBRAL_POPULARIDAD} interacciones": poco,
              f"objetivo con >{UMBRAL_POPULARIDAD}": ~poco,
              "historial completo con imagen": toda_con_imagen,
              "historial con algún ítem sin imagen": ~toda_con_imagen}
    resultado["grupos"] = {}
    for grupo, mascara in grupos.items():
        fila = {"filas": int(mascara.sum())}
        for nombre, d in datos.items():
            for k in (5, 10):
                hr, ndcg = pareado.aciertos(d, k)
                fila[f"{nombre} HR@{k}"] = float(hr[mascara].mean())
                fila[f"{nombre} NDCG@{k}"] = float(ndcg[mascara].mean())
        resultado["grupos"][grupo] = fila
    return resultado
