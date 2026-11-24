"""Qué archivos de Colab pasan a resultados/ (lo usan exportar.py en Colab e importar_resultados.py en local).

Solo resultados pequeños y verificables: predicciones (final_result.json), registros de tiempo y GPU, cobertura de
imágenes y métricas. Nunca pesos, estados del entrenador, partes intermedias, imágenes ni embeddings.
"""
import re
import hashlib

# Carpeta de MyDrive/IIC3633 → carpeta de resultados/ (la misma estructura que checkpoints/ en local)
ORIGENES = {"checkpoints/bigrec": "resultados/bigrec",
            "checkpoints/multimodal": "resultados/multimodal",
            "colab": "resultados/colab"}

PATRONES = [re.compile(p) for p in [
    # Frente E: BIGRec y D3 de Games e Instruments, y la validación de la T4 con BIGRec Toys
    r"^resultados/bigrec/(Video_Games|Musical_Instruments)/predicciones/(bigrec|d3)/(registro|final_result)\.json$",
    r"^resultados/bigrec/Toys_and_Games/predicciones/bigrec_t4/(registro|final_result)\.json$",
    # Frente M: cobertura (4 datasets), manifiestos de imágenes y embeddings y resultados de SASRec + SigLIP (Toys)
    r"^resultados/multimodal/cobertura/(cobertura_(CDs_and_Vinyl|Musical_Instruments|Toys_and_Games|Video_Games)\.csv"
    r"|resumen\.json)$",
    r"^resultados/multimodal/Toys_and_Games/(manifiesto_imagenes\.csv|manifiesto_embeddings\.json|sasrec_mm\.json)$",
    r"^resultados/multimodal/Toys_and_Games/sasrec_mm/[A-Za-z0-9_.+\-]+\.json$",
    # Ambos: uso de la GPU, etapas y validación
    r"^resultados/colab/(evaluaciones|multimodal)/(gpu_log\.csv|etapas\.json|validacion_t4\.json)$",
]]
MANIFIESTO = "MANIFIESTO.sha256"


def permitido(ruta):
    """True si `ruta` (relativa a la raíz del repo, con '/') está en la lista blanca."""
    return ".." not in ruta.split("/") and not ruta.startswith("/") and any(p.match(ruta) for p in PATRONES)


def sha256_bytes(datos):
    return hashlib.sha256(datos).hexdigest()
