"""Entrenamiento y evaluación del baseline Caser de LatentR3 (lo usa baseline_caser.ipynb).

Usa el modelo `Caser` de la implementación en PyTorch del primer autor (graytowne/caser_pytorch,
commit c4e10e4), sin modificar su código. Para acelerar, las capas de convolución de cada modelo se
calculan con multiplicaciones de matrices equivalentes, y el puntaje de todo el catálogo se calcula con
una multiplicación sobre W2 y b2. El protocolo común (datos, semillas, métricas) está en
baseline_comun.py y las decisiones y resultados en README.md.
"""
import os
import sys
import time
import types
import functools
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

# MAX_LEN es L: el contexto máximo del paper (Apéndice A); el código original usa L=5 (train_caser.py L286)
from baseline_comun import (SEED, MAX_LEN, MAX_EPOCAS, PACIENCIA, device, fijar_semillas, verificar_sha256,
                            cargar_modulo, cargar, cargar_catalogo, evaluar)

# Valores por defecto de train_caser.py (L287, L291, L294, L301-306)
BATCH_SIZE = 512
D = 50                # dimensión latente
NV = 4                # filtros verticales
NH = 16               # filtros horizontales por cada altura 1..L
DROPOUT = 0.5
AC_CONV = "relu"
AC_FC = "relu"
T_REFERENCIA = 3      # objetivos por fila en la variante "BCE T=3"
NEGATIVOS_T3 = 3      # negativos por fila en la variante "BCE T=3"
# Convoluciones como multiplicación de matrices (ver conv_h_matmul y README.md, Adaptaciones y
# Problemas conocidos). False usa nn.Conv2d (MIOpen o cuDNN)
CONV_MATMUL = True
BLOQUE_EVAL = 1024    # filas por bloque al puntuar todo el catálogo en recomendar
# sha256 de los archivos que se cargan, en el commit c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1
COMMIT_OFICIAL = "c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1"
SHA256 = {"caser.py": "63dd99f91021d99093d33cc1c71e182acdca4c8d3c366e00369981a297a22d31",
          "utils.py": "8f63d96e9e3449a71543e382157c594104999df0ee36f484e2c4416ca2ab3ab1"}
Caser = None  # se carga del código original en inicializar_proceso


def verificar_codigo(codigo):
    """Comprueba que caser.py y utils.py sean los del commit revisado, antes de importarlos."""
    verificar_sha256(codigo, SHA256, COMMIT_OFICIAL, "graytowne/caser_pytorch")


def inicializar_proceso(codigo):
    """Prepara un proceso: importa el modelo original y usa un hilo de CPU, para que los procesos no compitan."""
    global Caser
    torch.set_num_threads(1)
    verificar_codigo(codigo)
    # caser.py hace `from utils import activation_getter` (L5). Se registra utils.py del mismo commit
    # como módulo "utils" solo mientras se ejecuta caser.py, sin agregar la carpeta a sys.path
    anterior = sys.modules.get("utils")
    sys.modules["utils"] = cargar_modulo("utils", os.path.join(codigo, "utils.py"))
    try:
        Caser = cargar_modulo("caser_oficial", os.path.join(codigo, "caser.py")).Caser
    finally:
        if anterior is None:
            del sys.modules["utils"]
        else:
            sys.modules["utils"] = anterior


# ---------- Datos ----------

def preparar(df, usuarios):
    """Tensores de un split.

    S: historial de cada fila con padding a la izquierda hasta MAX_LEN, como interactions.py (L217-219).
    Los ids van +1, porque el índice 0 es el padding (interactions.py L130-134).
    u: índice de usuario, (filas, 1) como en train_caser.py (L105). Los usuarios de train van de 1 a U;
    los que no tienen filas en train usan el índice 0, que queda fijo en cero.
    y: objetivo de cada fila, con su id original.
    Y3: el objetivo de la fila y los de las 2 filas siguientes del mismo usuario en el split, con los
    ids +1 y 0 donde no hay más filas. Solo lo usa la variante "BCE T=3", sobre train.
    """
    n = len(df)
    S = np.zeros((n, MAX_LEN), dtype=np.int64)
    for fila, historial in enumerate(df["history_item_id"]):
        S[fila, MAX_LEN - len(historial):] = np.asarray(historial) + 1
    u = df["user_id"].map(usuarios).fillna(0).to_numpy(dtype=np.int64).reshape(-1, 1)
    y = df["item_id"].to_numpy()
    # Las filas de un usuario están en el orden de su secuencia (process.py ordena por tiempo, de forma estable)
    Y3 = np.zeros((n, T_REFERENCIA), dtype=np.int64)
    for filas in df.groupby("user_id", sort=False).indices.values():
        for k in range(T_REFERENCIA):
            Y3[filas[:max(len(filas) - k, 0)], k] = y[filas[k:]] + 1
    return {nombre: torch.tensor(arreglo, device=device)
            for nombre, arreglo in [("S", S), ("u", u), ("y", y), ("Y3", Y3)]}


@functools.lru_cache(maxsize=None)
def cargar_categoria(base, categoria):
    """Datos de una categoría, cargados una sola vez por proceso: (dataframes, títulos, usuarios, tensores por split)."""
    datos = {split: cargar(base, split, categoria) for split in ["train", "valid", "test"]}
    titulos = cargar_catalogo(base, categoria)
    # Usuarios con filas en train → 1..U, en orden de aparición
    usuarios = {u: i + 1 for i, u in enumerate(datos["train"]["user_id"].unique())}
    preparados = {split: preparar(df, usuarios) for split, df in datos.items()}
    return datos, titulos, usuarios, preparados


def lotes(prep, batch_size, rng):
    # Una época: cada fila de train una vez, en orden aleatorio
    perm = torch.from_numpy(rng.permutation(len(prep["y"]))).to(device)
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        yield {nombre: tensor[idx] for nombre, tensor in prep.items()}


# ---------- Modelo ----------

def conv_h_matmul(self, x):
    """Lo mismo que nn.Conv2d(1, nh, (i, d)) sobre x (filas, 1, L, d), con multiplicaciones de matrices.

    Como el filtro cubre las d columnas, cada salida es el producto de una ventana de i filas
    consecutivas (i*d valores) con el filtro aplanado, más el sesgo. Devuelve (filas, nh, L-i+1, 1).
    """
    i = self.kernel_size[0]
    filas, _, largo, d = x.shape
    ventanas = x[:, 0].unfold(1, i, 1).transpose(2, 3).reshape(filas, largo - i + 1, i * d)
    salida = ventanas @ self.weight.view(self.out_channels, -1).T + self.bias
    return salida.permute(0, 2, 1).unsqueeze(3)


def conv_v_matmul(self, x):
    """Lo mismo que nn.Conv2d(1, nv, (L, 1)) sobre x (filas, 1, L, d): cada filtro es una combinación
    lineal de las L filas, columna por columna, más el sesgo. Devuelve (filas, nv, 1, d)."""
    salida = torch.einsum("bld,cl->bcd", x[:, 0], self.weight[:, 0, :, 0]) + self.bias[None, :, None]
    return salida.unsqueeze(2).contiguous()   # forward hace .view (caser.py L97)


def convoluciones_como_matmul(modelo):
    # Reemplaza solo el forward de las capas conv_v y conv_h de este modelo. Caser.forward, los parámetros
    # y el state_dict siguen siendo los del original
    modelo.conv_v.forward = types.MethodType(conv_v_matmul, modelo.conv_v)
    for conv in modelo.conv_h:
        conv.forward = types.MethodType(conv_h_matmul, conv)
    return modelo


def nuevo_modelo(n_items, n_usuarios, con_usuario=True):
    args = SimpleNamespace(L=MAX_LEN, d=D, nv=NV, nh=NH, drop=DROPOUT, ac_conv=AC_CONV, ac_fc=AC_FC)
    # El índice 0 es el padding entre los ítems y el usuario sin filas en train entre los usuarios.
    # La inicialización es la del constructor (caser.py L63-66)
    modelo = Caser(n_usuarios + 1, n_items + 1, args).to(device)
    with torch.no_grad():
        modelo.user_embeddings.weight[0] = 0
    if not con_usuario:
        # Variante "BCE sin usuario": todas las filas usan el índice 0, que no se entrena
        modelo.user_embeddings.weight.requires_grad_(False)
    if CONV_MATMUL:
        convoluciones_como_matmul(modelo)
    return modelo


def puntuar(modelo, S, u, items):
    # forward del original (caser.py L70-127) para items (filas, k), con los ids +1. Devuelve (filas, k).
    # view evita que el squeeze de L125 quite dimensiones cuando el lote tiene una sola fila
    return modelo(S, u, items).view(len(S), -1)


def representacion(modelo, S, u):
    """x = [z, embedding de usuario] (filas, 2d), el vector que forward multiplica por W2 (caser.py L114-115).

    Se toma del forward original con hooks en fc1 y en el embedding de usuario; forward se llama con un
    solo ítem por fila, que no se usa. Solo se repiten la activación y la concatenación de L114-115.
    """
    capturas = {}
    ganchos = [modelo.fc1.register_forward_hook(lambda m, i, o: capturas.__setitem__("fc1", o)),
               modelo.user_embeddings.register_forward_hook(lambda m, i, o: capturas.__setitem__("u", o))]
    try:
        modelo(S, u, torch.ones(len(S), 1, dtype=torch.long, device=S.device))
    finally:
        for gancho in ganchos:
            gancho.remove()
    return torch.cat([modelo.ac_fc(capturas["fc1"]), capturas["u"].squeeze(1)], 1)


def puntajes(modelo, S, u):
    # Puntaje de cada ítem del catálogo: (filas, n_items). Es b2 + W2·x de forward (L117-125) para todos
    # los ítems a la vez, sin copiar W2 por fila. Se salta el índice 0 (padding), así que la columna j
    # corresponde al id j
    return torch.addmm(modelo.b2.weight[1:, 0], representacion(modelo, S, u), modelo.W2.weight[1:].T)


def perdida_bce(modelo, lote):
    # Paper (Apéndice B): BCE del objetivo de la fila contra un negativo uniforme del catálogo.
    # Con T=1 y un negativo es la pérdida de train_caser.py (L154-170): media de los positivos más
    # media de los negativos
    y = lote["y"]
    negativos = torch.randint(modelo.item_embeddings.num_embeddings - 1, y.shape, device=device)
    logits = puntuar(modelo, lote["S"], lote["u"], torch.stack([y, negativos], 1) + 1)
    etiquetas = torch.zeros_like(logits)
    etiquetas[:, 0] = 1
    return F.binary_cross_entropy_with_logits(logits, etiquetas, reduction="sum") / len(y)


def perdida_ce(modelo, lote):
    # Variante de referencia: softmax sobre todo el catálogo, sin muestrear negativos. puntajes usa una
    # multiplicación con W2, así que no copia W2 por fila (con forward eran ~2,4 GB por lote en CDs)
    return F.cross_entropy(puntajes(modelo, lote["S"], lote["u"]), lote["y"])


def perdida_bce_t3(modelo, lote):
    # Variante de referencia: T=3 objetivos y 3 negativos por fila, como los valores por defecto del
    # original (train_caser.py L287, L294), con su pérdida (L166-170). Los objetivos que no existen
    # (0 en Y3) se excluyen. Con T=1 y un negativo se reduce a perdida_bce
    Y3 = lote["Y3"]
    negativos = torch.randint(1, modelo.item_embeddings.num_embeddings, (len(Y3), NEGATIVOS_T3), device=device)
    logits = puntuar(modelo, lote["S"], lote["u"], torch.cat([Y3, negativos], 1))
    positivos, negativos = logits[:, :T_REFERENCIA][Y3 != 0], logits[:, T_REFERENCIA:]
    return (F.binary_cross_entropy_with_logits(positivos, torch.ones_like(positivos))
            + F.binary_cross_entropy_with_logits(negativos, torch.zeros_like(negativos)))


# variante → (pérdida, usa el embedding de usuario)
VARIANTES = {"BCE (paper)": (perdida_bce, True),
             "BCE sin usuario": (perdida_bce, False),
             "CE catálogo": (perdida_ce, True),
             "BCE T=3": (perdida_bce_t3, True)}


def para_variante(prep, usuarios, variante):
    """(tensores por split, número de usuarios del modelo) de una variante.

    En "BCE sin usuario" todas las filas usan el índice 0, fijo en cero, y el modelo tiene un solo usuario.
    """
    if VARIANTES[variante][1]:
        return prep, len(usuarios)
    return {split: {**p, "u": torch.zeros_like(p["u"])} for split, p in prep.items()}, 0


@torch.no_grad()
def recomendar(modelo, prep, k=50, bloque=BLOQUE_EVAL):
    # Top-k ids por fila, en el orden de las filas. Se piden 50 para que, al quitar títulos repetidos,
    # queden al menos 10
    modelo.eval()
    recs = []
    for i in range(0, len(prep["y"]), bloque):
        h = puntajes(modelo, prep["S"][i:i + bloque], prep["u"][i:i + bloque])
        recs.extend(h.topk(k, dim=1).indices.cpu().tolist())
    return recs


def metricas_por_usuario(recs, df, titulos, usuarios):
    """HR@10 y NDCG@10 por título en las filas de usuarios con filas en train (conocidos) y sin ellas (nuevos)."""
    conocido = df["user_id"].isin(usuarios).to_numpy()
    salida = {}
    for grupo, mascara in [("conocidos", conocido), ("nuevos", ~conocido)]:
        r = evaluar([recs[i] for i in np.flatnonzero(mascara)], df[mascara], titulos)
        salida[f"HR@10 título {grupo}"] = r["HR@10 título"]
        salida[f"NDCG@10 título {grupo}"] = r["NDCG@10 título"]
    return salida


def entrenar(variante, lr, wd, p_train, p_valid, df_valid, n_items, n_usuarios, max_epocas=MAX_EPOCAS):
    """Devuelve (modelo en su mejor época, mejor época, NDCG@10 en valid, épocas corridas)."""
    perdida, con_usuario = VARIANTES[variante]
    rng = np.random.default_rng(SEED)
    modelo = nuevo_modelo(n_items, n_usuarios, con_usuario)
    # Adam con las betas por defecto y el L2 como weight_decay, igual que train_caser.py (L83-85)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=wd)
    mejor, sin_mejora = {"ndcg": -1}, 0
    for epoca in range(1, max_epocas + 1):
        modelo.train()   # recomendar deja el modelo en modo eval; el dropout solo va al entrenar
        for lote in lotes(p_train, BATCH_SIZE, rng):
            opt.zero_grad()
            perdida(modelo, lote).backward()
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
    datos, titulos, usuarios, prep = cargar_categoria(base, categoria)
    prep, n_usuarios = para_variante(prep, usuarios, variante)
    fijar_semillas()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    modelo, epoca, ndcg_valid, corridas = entrenar(variante, lr, wd, prep["train"], prep["valid"], datos["valid"],
                                                   len(titulos), n_usuarios, max_epocas)
    segundos = time.time() - inicio
    recs = recomendar(modelo, prep["test"])
    memoria = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    return {"variante": variante, "categoria": categoria, "lr": lr, "weight_decay": wd, "época": epoca,
            "épocas corridas": corridas, "segundos": segundos, "MiB de GPU": memoria, "NDCG@10 valid": ndcg_valid,
            **evaluar(recs, datos["test"], titulos),
            **metricas_por_usuario(recs, datos["test"], titulos, usuarios)}
