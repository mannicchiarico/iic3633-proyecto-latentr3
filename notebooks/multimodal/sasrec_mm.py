"""Baseline multimodal: SASRec con los embeddings SigLIP de los ítems (lo usa multimodal.ipynb).

Se usa el SASRec del port de pmixer (commit 9f6a02b) sin modificarlo y todo el protocolo de
../sasrec/sasrec_baseline.py (datos, pérdida, grilla, early stopping, métricas). Lo único que cambia es la tabla de
embeddings de ítems: `modelo.item_emb` se reemplaza por `EmbItemMM`, que suma al embedding de id una proyección
lineal (entrenable) del embedding SigLIP congelado del ítem. Como `EmbItemMM` se llama igual que nn.Embedding y
expone `weight` y `embedding_dim`, funcionan sin cambios log2feats y forward del port, y perdida_bce y puntajes del
baseline. Las decisiones están en README.md.
"""
import functools
import numpy as np
import torch

import sasrec_baseline as sb
from baseline_comun import evaluar

# Variantes: qué embedding SigLIP recibe cada ítem
VARIANTES = {
    "id + imagen": "imagen",                            # los ítems sin imagen usan un vector aprendido
    "id + imagen (título de respaldo)": "imagen_o_titulo",  # sin imagen: el embedding del título
    "id + título": "titulo",                            # control: separa el aporte de la imagen del de la proyección
}
VARIANTE_UNIMODAL = "solo id"   # el SASRec de siempre (sb.nuevo_modelo)
# Nombres de archivo (ASCII) de cada variante
NOMBRES = {"id + imagen": "id_imagen", "id + imagen (título de respaldo)": "id_imagen_respaldo_titulo",
           "id + título": "id_titulo", VARIANTE_UNIMODAL: "solo_id"}
UMBRAL_POPULARIDAD = 5          # ítems poco vistos: a lo más 5 interacciones en entrenamiento (decisión del proyecto)


class EmbItemMM(torch.nn.Module):
    """e(i) = id(i) + W·f(i) [+ v si el ítem no tiene embedding], con e(0) = 0 (padding)."""

    def __init__(self, emb_id, feats, falta=None):
        super().__init__()
        self.emb_id = emb_id                                   # el nn.Embedding del port, ya inicializado
        self.embedding_dim = emb_id.embedding_dim              # log2feats escala por su raíz
        self.register_buffer("feats", feats)                   # (n_items + 1, D), fila 0 en cero; congelado
        self.proy = torch.nn.Linear(feats.shape[1], self.embedding_dim, bias=False)
        torch.nn.init.xavier_normal_(self.proy.weight)         # misma inicialización que el resto del port
        if falta is not None:
            self.register_buffer("falta", falta.float())       # 1 si el ítem no tiene embedding
            self.vector_falta = torch.nn.Parameter(torch.zeros(self.embedding_dim))
        else:
            self.falta = None

    def forward(self, idx):
        idx = torch.as_tensor(idx, device=self.feats.device)
        e = self.emb_id(idx) + self.proy(self.feats[idx])
        if self.falta is not None:
            e = e + self.falta[idx].unsqueeze(-1) * self.vector_falta
        return e * (idx != 0).unsqueeze(-1)

    @property
    def weight(self):
        # Tabla completa (n_items + 1, d); la usa puntajes() del baseline
        return self.forward(torch.arange(self.feats.shape[0], device=self.feats.device))


def features(emb, tipo):
    """(feats con la fila 0 de padding, máscara de ítems sin embedding o None) para una variante."""
    imagen, tiene, titulo = emb["imagen"], emb["tiene_imagen"].astype(bool), emb["titulo"]
    if tipo == "imagen":
        f, falta = imagen, ~tiene
    elif tipo == "imagen_o_titulo":
        f, falta = np.where(tiene[:, None], imagen, titulo), None
    elif tipo == "titulo":
        f, falta = titulo, None
    else:
        raise ValueError(tipo)
    f = torch.from_numpy(np.vstack([np.zeros((1, f.shape[1]), dtype=np.float32), f]).astype(np.float32))
    if falta is not None:
        falta = torch.from_numpy(np.concatenate([[False], falta]))
    return f, falta


@functools.lru_cache(maxsize=None)
def _cargar_npz(ruta_npz):
    with np.load(ruta_npz) as datos:
        return {k: datos[k] for k in ("imagen", "tiene_imagen", "titulo")}


def nuevo_modelo_mm(n_items, ruta_npz, tipo):
    """SASRec del baseline (misma inicialización) con item_emb reemplazado por EmbItemMM."""
    modelo = sb.nuevo_modelo(n_items)
    f, falta = features(_cargar_npz(ruta_npz), tipo)
    assert f.shape[0] == n_items + 1, (f.shape, n_items)
    modelo.item_emb = EmbItemMM(modelo.item_emb, f, falta).to(sb.device)
    return modelo


def interacciones_train(df_train):
    """Interacciones de entrenamiento por ítem: usuarios distintos que lo tienen en algún historial u objetivo de
    train (definición del proyecto)."""
    por_usuario = {}
    for usuario, historial, objetivo in zip(df_train["user_id"], df_train["history_item_id"], df_train["item_id"]):
        por_usuario.setdefault(usuario, set()).update(historial)
        por_usuario[usuario].add(objetivo)
    conteo = {}
    for items in por_usuario.values():
        for item in items:
            conteo[item] = conteo.get(item, 0) + 1
    return conteo


def por_popularidad(recs, df_test, df_train, titulos, umbral=UMBRAL_POPULARIDAD):
    """Métricas de test separando las filas cuyo objetivo tiene ≤ umbral interacciones en train del resto."""
    conteo = interacciones_train(df_train)
    poco = np.array([conteo.get(i, 0) <= umbral for i in df_test["item_id"]])
    resultado = {}
    for nombre, mascara in [(f"objetivo con ≤{umbral} interacciones", poco), (f"objetivo con >{umbral}", ~poco)]:
        filas = np.flatnonzero(mascara)
        if len(filas):
            resultado[nombre] = {"filas": int(len(filas)),
                                 **evaluar([recs[i] for i in filas], df_test.iloc[filas], titulos)}
    return resultado


def correr(base, categoria, variante, lr, wd, ruta_npz=None, max_epocas=sb.MAX_EPOCAS, perdida="BCE (paper)"):
    """Una combinación de la grilla (se puede correr en otro proceso). Devuelve métricas totales y por popularidad."""
    crear = None if variante == VARIANTE_UNIMODAL else functools.partial(
        nuevo_modelo_mm, ruta_npz=ruta_npz, tipo=VARIANTES[variante])
    r = sb.correr_combinacion(base, categoria, perdida, lr, wd, max_epocas, crear_modelo=crear, con_recs=True)
    datos, titulos, _ = sb.cargar_categoria(base, categoria)
    recs = r.pop("recs")
    r.update(variante_mm=variante, perdida=perdida,
             popularidad=por_popularidad(recs, datos["test"], datos["train"], titulos))
    return r
