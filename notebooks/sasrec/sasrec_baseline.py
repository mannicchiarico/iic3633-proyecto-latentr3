"""Entrenamiento y evaluación del baseline SASRec de LatentR3 (lo usa baseline_sasrec.ipynb).

Usa el modelo `SASRec` del port en PyTorch de pmixer (commit 9f6a02b), sin modificarlo. El protocolo
común (datos, semillas, métricas) está en baseline_comun.py y las decisiones en README.md.
"""
import os
import time
import functools
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

from baseline_comun import (SEED, MAX_LEN, MAX_EPOCAS, PACIENCIA, device, fijar_semillas, verificar_sha256,
                            cargar_modulo, cargar, cargar_catalogo, evaluar)

# Valores por defecto del SASRec original en TensorFlow (main.py L20-27 y model.py L103)
BATCH_SIZE = 128
HIDDEN = 50
BLOQUES = 2
CABEZAS = 1
DROPOUT = 0.5         # el port usa 0.2; se sigue al original
BETAS = (0.9, 0.98)
# sha256 de python/model.py en el commit 9f6a02b86a95e616fe815a0edc92dafa9911773d del port
COMMIT_OFICIAL = "9f6a02b86a95e616fe815a0edc92dafa9911773d"
SHA256_MODELO = "eefdc8211841d6fbb7b8e16ec68e76af87e148156daaefb45ad0cd3c94d1e8e9"
SASRec = None  # se carga del port en inicializar_proceso


def ruta_modelo(codigo):
    return os.path.join(codigo, "python", "model.py")


def verificar_codigo(codigo):
    """Comprueba que python/model.py sea el del commit revisado, antes de importarlo."""
    verificar_sha256(codigo, {os.path.join("python", "model.py"): SHA256_MODELO}, COMMIT_OFICIAL,
                     "pmixer/SASRec.pytorch")


def inicializar_proceso(codigo):
    """Prepara un proceso: importa el modelo del port y usa un hilo de CPU, para que los procesos no compitan."""
    global SASRec
    torch.set_num_threads(1)
    verificar_codigo(codigo)
    # Se carga solo model.py, sin agregar la carpeta a sys.path: el port trae módulos llamados utils y main
    SASRec = cargar_modulo("sasrec_oficial", ruta_modelo(codigo)).SASRec


# ---------- Datos ----------

def preparar(df):
    """Arreglos de un split, con padding a la izquierda hasta MAX_LEN, como el sampler original.

    S: historial de cada fila. P: ítem siguiente de cada posición del historial (el último es el
    objetivo); solo lo usan las variantes "todas las posiciones". En S y P los ids van +1, porque el
    port reserva el índice 0 para el padding. y: objetivo de cada fila, con su id original.
    """
    n = len(df)
    S = np.zeros((n, MAX_LEN), dtype=np.int64)
    P = np.zeros((n, MAX_LEN), dtype=np.int64)
    y = df["item_id"].to_numpy()
    for fila, (historial, objetivo) in enumerate(zip(df["history_item_id"], y)):
        largo = len(historial)
        S[fila, MAX_LEN - largo:] = np.asarray(historial) + 1
        P[fila, MAX_LEN - largo:] = np.asarray(list(historial[1:]) + [objetivo]) + 1
    return {"S": S, "P": P, "y": torch.tensor(y, device=device)}


@functools.lru_cache(maxsize=None)
def cargar_categoria(base, categoria):
    """Datos de una categoría, cargados una sola vez por proceso: (dataframes, títulos, arreglos por split)."""
    datos = {split: cargar(base, split, categoria) for split in ["train", "valid", "test"]}
    titulos = cargar_catalogo(base, categoria)
    preparados = {split: preparar(df) for split, df in datos.items()}
    return datos, titulos, preparados


def lotes(prep, batch_size, rng):
    # Una época: cada fila de train una vez, en orden aleatorio. S y P quedan en numpy (CPU),
    # porque log2feats y forward del port los convierten con torch.LongTensor
    perm = rng.permutation(len(prep["S"]))
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        yield prep["S"][idx], prep["P"][idx], prep["y"][torch.from_numpy(idx).to(device)]


# ---------- Modelo ----------

def nuevo_modelo(n_items):
    args = SimpleNamespace(device=device, hidden_units=HIDDEN, num_blocks=BLOQUES, num_heads=CABEZAS,
                           dropout_rate=DROPOUT, maxlen=MAX_LEN)
    modelo = SASRec(0, n_items, args).to(device)   # el modelo no usa el número de usuarios
    # Inicialización de main.py del port (L58-65): xavier normal en todo lo que lo admite y padding en cero
    for _, param in modelo.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except ValueError:
            pass   # xavier no está definido para tensores 1D (sesgos y LayerNorm)
    modelo.pos_emb.weight.data[0, :] = 0
    modelo.item_emb.weight.data[0, :] = 0
    return modelo


def codificar(modelo, S):
    # Salida de la última posición, la del ítem más reciente, como predict del port (model.py L110)
    return modelo.log2feats(S)[:, -1, :]


def puntajes(modelo, h):
    # Puntaje de cada ítem del catálogo: (filas, n_items). Se salta la fila 0 (padding), así que la
    # columna j corresponde al id j
    return h @ modelo.item_emb.weight[1:].T


def perdida_bce(modelo, S, P, y):
    # Paper (Apéndice B): BCE del objetivo de la fila contra un negativo uniforme del catálogo
    h = codificar(modelo, S)
    negativos = torch.randint(modelo.item_num, y.shape).to(device)

    def logit(items):
        # Misma fórmula que forward del port (model.py L99-100), solo en la última posición
        return (h * modelo.item_emb(items + 1)).sum(1)

    logits = torch.cat([logit(y), logit(negativos)])
    etiquetas = torch.cat([torch.ones(len(y), device=device), torch.zeros(len(y), device=device)])
    return F.binary_cross_entropy_with_logits(logits, etiquetas, reduction="sum") / len(y)


def perdida_ce(modelo, S, P, y):
    # Variante de referencia: softmax sobre todo el catálogo en la última posición, sin muestrear negativos
    return F.cross_entropy(puntajes(modelo, codificar(modelo, S)), y)


def perdida_bce_todas(modelo, S, P, y):
    # Variante de referencia: BCE en todas las posiciones del historial, como el SASRec original.
    # Usa forward del port y promedia sobre las posiciones que no son padding (main.py L108-110)
    negativos = torch.randint(1, modelo.item_num + 1, P.shape).numpy()
    pos_logits, neg_logits = modelo(None, S, P, negativos)
    m = torch.from_numpy(P != 0).to(device)
    return (F.binary_cross_entropy_with_logits(pos_logits[m], torch.ones_like(pos_logits[m]))
            + F.binary_cross_entropy_with_logits(neg_logits[m], torch.zeros_like(neg_logits[m])))


def perdida_ce_todas(modelo, S, P, y):
    # Variante de referencia: combina las dos anteriores. Softmax sobre todo el catálogo en cada posición
    # del historial, con su contexto causal, y media sobre las posiciones que no son padding. Los puntajes
    # son los de `puntajes` (sin la fila de padding), así que el objetivo de cada posición es su id original
    m = torch.from_numpy(P != 0).to(device)
    feats = modelo.log2feats(S)[m]                       # (posiciones supervisadas, d)
    objetivos = torch.from_numpy(P).to(device)[m] - 1
    return F.cross_entropy(puntajes(modelo, feats), objetivos)


PERDIDAS = {"BCE (paper)": perdida_bce, "CE catálogo": perdida_ce,
            "BCE todas las posiciones": perdida_bce_todas, "CE todas las posiciones": perdida_ce_todas}


@torch.no_grad()
def recomendar(modelo, prep, k=50, bloque=1024):
    # Top-k ids por fila, en el orden de las filas. Se piden 50 para que, al quitar títulos repetidos,
    # queden al menos 10
    modelo.eval()
    recs = []
    for i in range(0, len(prep["S"]), bloque):
        h = codificar(modelo, prep["S"][i:i + bloque])
        recs.extend(puntajes(modelo, h).topk(k, dim=1).indices.cpu().tolist())
    return recs


def entrenar(perdida, lr, wd, p_train, p_valid, df_valid, n_items, max_epocas=MAX_EPOCAS):
    """Devuelve (modelo en su mejor época, mejor época, NDCG@10 en valid, épocas corridas)."""
    rng = np.random.default_rng(SEED)
    modelo = nuevo_modelo(n_items)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, betas=BETAS, weight_decay=wd)
    mejor, sin_mejora = {"ndcg": -1}, 0
    for epoca in range(1, max_epocas + 1):
        modelo.train()   # recomendar deja el modelo en modo eval; el dropout solo va al entrenar
        for S, P, y in lotes(p_train, BATCH_SIZE, rng):
            opt.zero_grad()
            perdida(modelo, S, P, y).backward()
            opt.step()
        ndcg10 = evaluar(recomendar(modelo, p_valid), df_valid)["NDCG@10 id"]
        if ndcg10 > mejor["ndcg"]:
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            mejor, sin_mejora = {"ndcg": ndcg10, "epoca": epoca, "estado": estado}, 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(mejor["estado"])
    return modelo, mejor["epoca"], mejor["ndcg"], epoca


def correr_combinacion(base, categoria, variante, lr, wd, max_epocas=MAX_EPOCAS):
    """Entrena una combinación y devuelve su resultado en valid y en test.

    Las métricas de test se calculan para todas las combinaciones porque cada una corre en otro
    proceso, pero la elección de la mejor configuración usa solo `NDCG@10 valid`.
    """
    datos, titulos, prep = cargar_categoria(base, categoria)
    fijar_semillas()
    inicio = time.time()
    modelo, epoca, ndcg_valid, corridas = entrenar(PERDIDAS[variante], lr, wd, prep["train"], prep["valid"],
                                                   datos["valid"], len(titulos), max_epocas)
    segundos = time.time() - inicio
    recs = recomendar(modelo, prep["test"])
    return {"variante": variante, "categoria": categoria, "lr": lr, "weight_decay": wd, "época": epoca,
            "épocas corridas": corridas, "segundos": segundos, "NDCG@10 valid": ndcg_valid,
            **evaluar(recs, datos["test"], titulos)}
