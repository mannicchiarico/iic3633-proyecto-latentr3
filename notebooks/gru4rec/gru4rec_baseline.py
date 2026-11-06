"""Entrenamiento y evaluación del baseline GRU4Rec de LatentR3 (lo usa baseline_gru4rec.ipynb).

Está en un .py para poder correr la grilla en procesos paralelos (Jupyter no puede enviarles funciones
definidas en sus celdas). El protocolo común está en baseline_comun.py y las decisiones en README.md.
"""
import os
import functools
import numpy as np
import torch
import torch.nn.functional as F

from baseline_comun import (SEED, MAX_LEN, MAX_EPOCAS, PACIENCIA, device, fijar_semillas, verificar_sha256,
                            cargar_modulo, buscar_archivo, cargar, cargar_catalogo, hit, ndcg, dedup_titulos,
                            evaluar)

BATCH_SIZE = 64
# sha256 de gru4rec_pytorch.py en el commit fijado de hidasib/GRU4Rec_PyTorch_Official
COMMIT_OFICIAL = "d1fc31105577665d2f105de1c8d9a38d223a9a22"
SHA256_MODELO = "d5b52620f76f36b13e229a53d11176d822d472270f3183fda25a034051e61278"
GRU4RecModel = None  # se importa del código oficial en inicializar_proceso


def verificar_codigo(codigo):
    """Comprueba que gru4rec_pytorch.py sea el del commit revisado, antes de importarlo."""
    verificar_sha256(codigo, {"gru4rec_pytorch.py": SHA256_MODELO}, COMMIT_OFICIAL, "GRU4Rec_PyTorch_Official")


def inicializar_proceso(codigo):
    """Prepara un proceso: importa el modelo oficial y usa un hilo de CPU, para que 6 procesos no compitan."""
    global GRU4RecModel
    torch.set_num_threads(1)
    verificar_codigo(codigo)
    # Se carga solo gru4rec_pytorch.py, sin agregar la carpeta a sys.path: el repositorio trae módulos llamados
    # run y evaluation. Ese archivo importa solo math, numpy, pandas, torch, collections y time
    GRU4RecModel = cargar_modulo("gru4rec_oficial", os.path.join(codigo, "gru4rec_pytorch.py")).GRU4RecModel


# ---------- Datos ----------

def agrupar_por_largo(df):
    """{largo del historial: (posiciones de las filas, X (filas, largo), y (filas,))}"""
    largos = df["history_item_id"].map(len).to_numpy()
    grupos = {}
    for largo in np.unique(largos):
        pos = np.flatnonzero(largos == largo)
        X = torch.tensor(df["history_item_id"].iloc[pos].tolist(), device=device)
        y = torch.tensor(df["item_id"].to_numpy()[pos], device=device)
        grupos[int(largo)] = (pos, X, y)
    return grupos


def lotes(grupos, batch_size, rng):
    salida = []
    for _, X, y in grupos.values():
        perm = torch.from_numpy(rng.permutation(len(y))).to(device)
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            salida.append((X[idx], y[idx]))
    return [salida[i] for i in rng.permutation(len(salida))]


@functools.lru_cache(maxsize=None)
def cargar_categoria(base, categoria):
    """Datos de una categoría, cargados una sola vez por proceso: (dataframes, títulos, grupos por largo)."""
    datos = {split: cargar(base, split, categoria) for split in ["train", "valid", "test"]}
    titulos = cargar_catalogo(base, categoria)
    grupos = {split: agrupar_por_largo(df) for split, df in datos.items()}
    return datos, titulos, grupos


# ---------- Modelo ----------

def nuevo_modelo(n_items):
    modelo = GRU4RecModel(n_items, layers=[100], dropout_p_embed=0.0, dropout_p_hidden=0.0,
                          embedding=0, constrained_embedding=True).to(device)
    for modulo in modelo.modules():
        if isinstance(modulo, torch.nn.Embedding):
            modulo.sparse = False   # Adam no acepta gradientes dispersos
    return modelo


def codificar(modelo, X, training=False):
    H = [torch.zeros(X.shape[0], n, device=device) for n in modelo.layers]
    for t in range(X.shape[1]):
        E, _, _ = modelo.embed(X[:, t], H, None)
        h = modelo.hidden_step(E, H, training=training)
    return h


def codificar_todas(modelo, X, training=False):
    # Como codificar, pero devuelve el estado oculto de cada paso: (filas, largo, d)
    H = [torch.zeros(X.shape[0], n, device=device) for n in modelo.layers]
    estados = []
    for t in range(X.shape[1]):
        E, _, _ = modelo.embed(X[:, t], H, None)
        estados.append(modelo.hidden_step(E, H, training=training))
    return torch.stack(estados, 1)


def puntajes(modelo, h):
    return modelo.score_items(h, modelo.Wy.weight, modelo.By.weight)


def perdida_bce(modelo, h, y):
    # Paper, App. B: BCE del objetivo contra un negativo uniforme del catálogo
    negativos = torch.randint(modelo.n_items, y.shape, device=device)

    def logit(items):
        return (h * modelo.Wy(items)).sum(1) + modelo.By(items).squeeze(1)

    logits = torch.cat([logit(y), logit(negativos)])
    etiquetas = torch.cat([torch.ones(len(y), device=device), torch.zeros(len(y), device=device)])
    return F.binary_cross_entropy_with_logits(logits, etiquetas, reduction="sum") / len(y)


def perdida_ce(modelo, h, y):
    return F.cross_entropy(puntajes(modelo, h), y)


def perdida_bce_todas(modelo, X, y):
    # Cada posición predice el ítem siguiente (en la última, el objetivo de la fila) contra un negativo
    # uniforme. Sin padding: los lotes son de filas del mismo largo. Misma escala que perdida_bce
    H = codificar_todas(modelo, X, training=True)
    P = torch.cat([X[:, 1:], y[:, None]], 1)
    negativos = torch.randint(modelo.n_items, P.shape, device=device)

    def logit(items):
        return (H * modelo.Wy(items)).sum(2) + modelo.By(items).squeeze(2)

    logits = torch.cat([logit(P).flatten(), logit(negativos).flatten()])
    etiquetas = torch.cat([torch.ones(P.numel(), device=device), torch.zeros(P.numel(), device=device)])
    return F.binary_cross_entropy_with_logits(logits, etiquetas, reduction="sum") / P.numel()


def perdida_ce_todas(modelo, X, y):
    # CE sobre todo el catálogo en cada posición (el oficial usa softmax contra el lote y negativos
    # muestreados). Media sobre las posiciones
    H = codificar_todas(modelo, X, training=True)
    P = torch.cat([X[:, 1:], y[:, None]], 1)
    return F.cross_entropy(puntajes(modelo, H.reshape(-1, H.shape[2])), P.reshape(-1))


PERDIDAS = {"BCE (paper)": perdida_bce, "CE catálogo": perdida_ce,
            "BCE todas las posiciones": perdida_bce_todas, "CE todas las posiciones": perdida_ce_todas}
POR_POSICION = {perdida_bce_todas, perdida_ce_todas}   # reciben el historial X; las demás, el último estado oculto


@torch.no_grad()
def recomendar(modelo, grupos_df, n_filas, k=50):
    # Se piden 50 para que, al quitar títulos repetidos, queden al menos 10
    recs = [None] * n_filas
    for pos, X, _ in grupos_df.values():
        top = puntajes(modelo, codificar(modelo, X)).topk(k, dim=1).indices.cpu().numpy()
        for p, r in zip(pos, top):
            recs[p] = r.tolist()
    return recs


def entrenar(perdida, lr, wd, g_train, g_valid, df_valid, n_items):
    """Devuelve (modelo en su mejor época, mejor época, NDCG@10 en valid, épocas corridas)."""
    rng = np.random.default_rng(SEED)
    modelo = nuevo_modelo(n_items)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=wd)
    mejor, sin_mejora = {"ndcg": -1}, 0
    for epoca in range(1, MAX_EPOCAS + 1):
        for X, y in lotes(g_train, BATCH_SIZE, rng):
            opt.zero_grad()
            entrada = X if perdida in POR_POSICION else codificar(modelo, X, training=True)
            perdida(modelo, entrada, y).backward()
            opt.step()
        ndcg10 = evaluar(recomendar(modelo, g_valid, len(df_valid)), df_valid)["NDCG@10 id"]
        if ndcg10 > mejor["ndcg"]:
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            mejor, sin_mejora = {"ndcg": ndcg10, "epoca": epoca, "estado": estado}, 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(mejor["estado"])
    return modelo, mejor["epoca"], mejor["ndcg"], epoca


def correr_combinacion(base, categoria, variante, lr, wd):
    """Entrena una combinación y devuelve su resultado en valid y en test.

    Las métricas de test se calculan para todas las combinaciones porque cada una corre en otro
    proceso, pero la elección de la mejor configuración usa solo `NDCG@10 valid`.
    """
    datos, titulos, grupos = cargar_categoria(base, categoria)
    fijar_semillas()
    modelo, epoca, ndcg_valid, corridas = entrenar(PERDIDAS[variante], lr, wd, grupos["train"], grupos["valid"],
                                                   datos["valid"], len(titulos))
    recs = recomendar(modelo, grupos["test"], len(datos["test"]))
    return {"variante": variante, "categoria": categoria, "lr": lr, "weight_decay": wd, "época": epoca,
            "épocas corridas": corridas, "NDCG@10 valid": ndcg_valid, **evaluar(recs, datos["test"], titulos)}
