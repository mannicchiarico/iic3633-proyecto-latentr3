"""Protocolo común de los baselines de LatentR3: constantes, carga de datos, métricas, semillas y carga
verificada del código oficial de cada modelo.

Lo importan los .py de GRU4Rec, SASRec, Caser, BIGRec, Base/COT y LatentR3, y el notebook de D3. Las decisiones comunes
(D1–D17) están en notebooks/README.md. `entorno()` y `verificar_gpu()` registran y comprueban la GPU; la segunda solo
actúa en Colab (colab/README.md).
"""
import os
import ast
import random
import shutil
import hashlib
import platform
import subprocess
import importlib.util
import importlib.metadata
import numpy as np
import pandas as pd
import torch

SEED = 42
MAX_LEN = 10        # contexto máximo del paper (Apéndice A)
MAX_EPOCAS = 200
PACIENCIA = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semillas():
    # Se llama antes de cada combinación, para que cada una parta del mismo estado aleatorio
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ---------- Código oficial ----------

def verificar_sha256(codigo, archivos, commit, nombre):
    """Comprueba que cada archivo de `archivos` ({ruta relativa a `codigo`: sha256}) sea el del commit
    revisado, antes de importarlo."""
    for archivo, esperado in archivos.items():
        ruta = os.path.join(codigo, archivo)
        with open(ruta, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if sha != esperado:
            raise ValueError(f"{ruta} no corresponde al commit {commit[:7]} de {nombre} (sha256 {sha}).")


def cargar_modulo(nombre, ruta):
    # Importa un archivo como módulo sin agregar su carpeta a sys.path
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# ---------- Datos ----------

def buscar_archivo(base, split, categoria):
    carpeta = os.path.join(base, split)
    archivos = [f for f in os.listdir(carpeta) if f.startswith(categoria)]
    if not archivos:
        raise FileNotFoundError(f"No hay un archivo de {categoria} en {os.path.abspath(carpeta)}")
    return os.path.join(carpeta, archivos[0])


def cargar(base, split, categoria):
    df = pd.read_csv(buscar_archivo(base, split, categoria))
    df["history_item_id"] = df["history_item_id"].apply(ast.literal_eval)
    assert df["history_item_id"].map(len).between(1, MAX_LEN).all()
    return df


def cargar_catalogo(base, categoria):
    # Títulos indexados por id, normalizados como en calc.py de los autores
    with open(buscar_archivo(base, "info", categoria), encoding="utf-8") as f:
        pares = [linea.rstrip("\n").rsplit("\t", 1) for linea in f if linea.strip()]
    titulos = {int(i): t.strip().strip('"').strip() for t, i in pares}
    # process.py asigna los ids 0..n-1, así que el id es directamente el índice del modelo
    assert sorted(titulos) == list(range(len(titulos)))
    return [titulos[i] for i in range(len(titulos))]


# ---------- Métricas ----------

def hit(recs, objetivo, k):
    return 1 if objetivo in list(recs[:k]) else 0


def ndcg(recs, objetivo, k):
    recs = list(recs[:k])
    if objetivo not in recs:
        return 0
    pos = recs.index(objetivo)
    return 1 / np.log2(pos + 2)


def dedup_titulos(recs, titulos):
    # Ids → títulos sin repetir, en orden: la lista que produciría el beam search de los LLM
    vistos = {}
    for i in recs:
        vistos.setdefault(titulos[i], None)
    return list(vistos)


def evaluar(recomendaciones, df, titulos=None, ks=(5, 10)):
    # Por id: acierto si el id coincide. Por título: acierto si el título coincide (calc.py)
    objetivos = df["item_id"].tolist()
    listas = {"id": ([r[:10] for r in recomendaciones], objetivos)}
    if titulos is not None:
        listas["título"] = ([dedup_titulos(r, titulos)[:10] for r in recomendaciones],
                            [titulos[o] for o in objetivos])
    resultados = {}
    for criterio, (recs, objs) in listas.items():
        for k in ks:
            resultados[f"HR@{k} {criterio}"] = np.mean([hit(r, o, k) for r, o in zip(recs, objs)])
            resultados[f"NDCG@{k} {criterio}"] = np.mean([ndcg(r, o, k) for r, o in zip(recs, objs)])
    if titulos is not None:
        # Un acierto por id siempre es un acierto por título, en la misma posición o antes
        for k in ks:
            assert resultados[f"HR@{k} título"] >= resultados[f"HR@{k} id"]
            assert resultados[f"NDCG@{k} título"] >= resultados[f"NDCG@{k} id"]
    return resultados


# ---------- Entorno y GPU (Colab) ----------

def en_colab():
    """True si el proceso corre en Google Colab."""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def _driver_nvidia():
    if shutil.which("nvidia-smi") is None:
        return None
    salida = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True)
    return salida.stdout.strip().splitlines()[0] if salida.returncode == 0 and salida.stdout.strip() else None


def bf16_nativo():
    """bf16 implementado en el hardware (no emulado). Falso en una T4 (capacidad 7.5)."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:   # torch sin el argumento: capacidad 8.0 o superior en CUDA
        return torch.version.hip is not None or torch.cuda.get_device_capability(0) >= (8, 0)


def entorno():
    """GPU, versiones y plataforma, para guardar junto a cada resultado."""
    info = {"plataforma": "colab" if en_colab() else "local", "sistema": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None)}
    for paquete in ("transformers", "trl", "accelerate"):
        try:
            info[paquete] = importlib.metadata.version(paquete)
        except importlib.metadata.PackageNotFoundError:
            info[paquete] = None
    if torch.cuda.is_available():
        propiedades = torch.cuda.get_device_properties(0)
        info.update(gpu=propiedades.name, memoria_total_gib=round(propiedades.total_memory / 2**30, 2),
                    capacidad=list(torch.cuda.get_device_capability(0)), bf16_nativo=bf16_nativo())
    else:
        info.update(gpu=None, memoria_total_gib=None, capacidad=None, bf16_nativo=False)
    info["driver"] = _driver_nvidia()
    return info


def precision_gpu():
    """Precisión de inferencia: bf16 si la GPU lo tiene en hardware (la de los autores); si no, fp16 (desviación)."""
    return torch.bfloat16 if bf16_nativo() else torch.float16


def verificar_gpu(etapa, modelo=None, memoria_min_gib=None, **config):
    """En Colab, se detiene con un mensaje claro si no hay GPU CUDA, si tiene menos memoria que `memoria_min_gib` o si
    algún parámetro de `modelo` no está en la GPU, e imprime la GPU y la configuración efectiva. Fuera de Colab no hace
    nada y devuelve None: en local se mantiene el comportamiento de siempre.
    """
    if not en_colab():
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(f"[{etapa}] No hay GPU. En Colab: Entorno de ejecución > Cambiar tipo de entorno de ejecución "
                           "> T4 GPU, y volver a correr la preparación. No se corre en CPU.")
    info = entorno()
    if memoria_min_gib is not None and info["memoria_total_gib"] < memoria_min_gib:
        raise RuntimeError(f"[{etapa}] La GPU {info['gpu']} tiene {info['memoria_total_gib']} GiB; hacen falta "
                           f"{memoria_min_gib} GiB.")
    if modelo is not None:
        fuera = [nombre for nombre, p in modelo.named_parameters() if p.device.type != "cuda"]
        if fuera:
            raise RuntimeError(f"[{etapa}] {len(fuera)} parámetros no están en la GPU (p. ej. {fuera[0]}).")
        print(f"[{etapa}] modelo en la GPU: OK")
        return info
    print(f"[{etapa}] GPU {info['gpu']} | {info['memoria_total_gib']} GiB | capacidad {info['capacidad']} | "
          f"bf16 nativo {info['bf16_nativo']} | driver {info['driver']} | torch {info['torch']} (CUDA {info['cuda']})")
    if config:
        print(f"[{etapa}] configuración: " + ", ".join(f"{k}={v}" for k, v in config.items()))
    return info
