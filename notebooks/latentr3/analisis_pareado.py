"""Comparación pareada de las variantes de LatentR3 y de nuestro BIGRec/D3 en un dataset.

Lee los final_result.json versionados en resultados/ y, si existen los modelos en checkpoints/, mide cuánto
cambiaron los pesos entre el SFT y el RL. Escribe resultados/latentr3/<categoría>/analisis_pareado.json.
Solo CPU; corre en segundos (el cambio de pesos lee los dos .safetensors, ~3 GB cada uno).

Uso, desde la raíz del repo:  python notebooks/latentr3/analisis_pareado.py [Toys_and_Games]
"""
import glob
import json
import math
import os
import sys
from math import comb

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CATEGORIA = sys.argv[1] if len(sys.argv) > 1 else "Toys_and_Games"
RES = os.path.join(RAIZ, "resultados")
ARCHIVOS = {
    "BIGRec": os.path.join(RES, "bigrec", CATEGORIA, "predicciones", "bigrec", "final_result.json"),
    "D3": os.path.join(RES, "bigrec", CATEGORIA, "predicciones", "d3", "final_result.json"),
    "only SFT": os.path.join(RES, "latentr3", CATEGORIA, "sft", "predicciones", "test", "final_result.json"),
    "LatentR3 (RL)": os.path.join(RES, "latentr3", CATEGORIA, "rl_1e-5", "predicciones", "test", "final_result.json"),
    "D3 + LatentR3": os.path.join(RES, "latentr3", CATEGORIA, "rl_1e-5", "predicciones", "test_d3", "final_result.json"),
}
PARES = [("only SFT", "LatentR3 (RL)"), ("BIGRec", "only SFT"), ("BIGRec", "LatentR3 (RL)"),
         ("D3", "D3 + LatentR3"), ("LatentR3 (RL)", "D3 + LatentR3")]
N_BOOTSTRAP, SEMILLA = 2000, 0


def posicion(fila):
    """Posición del objetivo en el top-10 con la normalización de calc.py (L380, L387; última aparición)."""
    objetivo = fila["output"].strip(" \n\"")
    pos = [i for i, p in enumerate(fila["predict"]) if p.strip(" \n").strip("\"").strip(" ") == objetivo]
    return pos[-1] if pos else None


def aciertos(datos, k):
    pos = [posicion(f) for f in datos]
    hr = np.array([1.0 if p is not None and p < k else 0.0 for p in pos])
    ndcg = np.array([1 / math.log2(p + 2) if p is not None and p < k else 0.0 for p in pos])
    return hr, ndcg


def mcnemar_exacto(b01, b10):
    m = b01 + b10
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(b01, b10) + 1)) / 2 ** m)


def comparar(a, b, datos, rng):
    A, B = datos[a], datos[b]
    n = len(A)
    out = {"listas idénticas": sum(x["predict"] == y["predict"] for x, y in zip(A, B)),
           "mismo conjunto de 10": sum(set(x["predict"]) == set(y["predict"]) for x, y in zip(A, B)),
           "mismo top-1": sum(x["predict"][0] == y["predict"][0] for x, y in zip(A, B)), "filas": n}
    idx = rng.integers(0, n, (N_BOOTSTRAP, n))
    for k in (5, 10):
        ha, na = aciertos(A, k)
        hb, nb = aciertos(B, k)
        b01, b10 = int(((ha == 0) & (hb == 1)).sum()), int(((ha == 1) & (hb == 0)).sum())
        dif = nb - na
        bs = dif[idx].mean(1)
        out[f"@{k}"] = {f"HR {a}": ha.mean(), f"HR {b}": hb.mean(), "filas que gana": b01, "filas que pierde": b10,
                        "McNemar exacto p": mcnemar_exacto(b01, b10), "dif. NDCG": dif.mean(),
                        "dif. NDCG IC95 bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def cambio_pesos():
    """Norma relativa del cambio de cada tensor entre el modelo del SFT y el del RL (None si no están)."""
    base = os.path.join(RAIZ, "checkpoints", "latentr3", CATEGORIA)
    rutas = [glob.glob(os.path.join(base, d, "modelo", "*.safetensors")) for d in ("sft", "rl_1e-5")]
    if not all(len(r) == 1 for r in rutas):
        return None
    import torch
    from safetensors import safe_open
    cambios = {}
    with safe_open(rutas[0][0], "pt") as a, safe_open(rutas[1][0], "pt") as b:
        for clave in a.keys():
            x, y = a.get_tensor(clave), b.get_tensor(clave)
            if not torch.equal(x, y):
                x, y = x.float(), y.float()
                cambios[clave] = ((y - x).norm() / x.norm()).item()
        total = len(list(a.keys()))
    return {"tensores": total, "cambio relativo (norma) de los tensores que cambian": cambios}


def recompensa():
    e = json.load(open(os.path.join(RES, "latentr3", CATEGORIA, "rl_1e-5", "entrenamiento.json")))
    r = np.array(e["recompensa_por_paso"])
    gn = [x["grad_norm"] for x in e["log_history"] if "grad_norm" in x]
    tramos = range(0, len(r), 200)
    return {"pasos": len(r), "todas finitas": bool(np.isfinite(r).all()),
            "recompensa media por tramo de 200 pasos": [float(r[i:i + 200].mean()) for i in tramos],
            "grad_norm mediana por tramo de 200 pasos": [float(np.median(gn[i:i + 200])) for i in tramos]}


def main():
    datos = {k: json.load(open(v)) for k, v in ARCHIVOS.items() if os.path.isfile(v)}
    entradas = [f["input"] for f in next(iter(datos.values()))]
    for k, v in datos.items():
        assert [f["input"] for f in v] == entradas, f"{k}: filas en otro orden"
    rng = np.random.default_rng(SEMILLA)
    salida = {"categoria": CATEGORIA, "bootstrap": {"muestras": N_BOOTSTRAP, "semilla": SEMILLA},
              "pares": {f"{a} -> {b}": comparar(a, b, datos, rng) for a, b in PARES if a in datos and b in datos},
              "recompensa RL": recompensa(), "pesos SFT -> RL": cambio_pesos()}
    ruta = os.path.join(RES, "latentr3", CATEGORIA, "analisis_pareado.json")
    with open(ruta, "w") as f:
        json.dump(salida, f, indent=1, ensure_ascii=False)
    print(json.dumps(salida, indent=1, ensure_ascii=False))
    print("escrito en", ruta)


if __name__ == "__main__":
    main()
