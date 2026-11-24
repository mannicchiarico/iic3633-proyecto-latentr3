"""Empaqueta en un zip los resultados de una persona para enviarlos (se corre en Colab, al final de cada sesión).

Recorre MyDrive/IIC3633 y guarda solo los archivos de la lista blanca (lista_blanca.py), con la ruta que tendrán en
resultados/ y un MANIFIESTO.sha256. Sin pesos, imágenes, embeddings ni partes intermedias. El zip queda en
MyDrive/IIC3633/export/ y se puede descargar desde el notebook.
"""
import os
import sys
import zipfile
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lista_blanca import ORIGENES, MANIFIESTO, permitido, sha256_bytes   # noqa: E402


def candidatos(iic3633):
    """[(ruta en disco, ruta en resultados/)] de todo lo que hay bajo los orígenes y pasa la lista blanca."""
    elegidos = []
    for origen, destino in ORIGENES.items():
        raiz = os.path.join(iic3633, origen)
        for carpeta, subcarpetas, archivos in os.walk(raiz):
            subcarpetas[:] = [s for s in subcarpetas if s not in ("modelo", "trainer")]   # no se recorren los pesos
            for archivo in archivos:
                ruta = os.path.join(carpeta, archivo)
                relativa = destino + "/" + os.path.relpath(ruta, raiz).replace(os.sep, "/")
                if permitido(relativa):
                    elegidos.append((ruta, relativa))
    return sorted(elegidos, key=lambda par: par[1])


def exportar(iic3633, persona, carpeta_salida=None):
    """Escribe el zip y devuelve su ruta. `persona`: "evaluaciones" o "multimodal" (solo nombra el archivo)."""
    elegidos = candidatos(iic3633)
    if not elegidos:
        raise FileNotFoundError(f"No hay resultados exportables bajo {iic3633}.")
    carpeta_salida = carpeta_salida or os.path.join(iic3633, "export")
    os.makedirs(carpeta_salida, exist_ok=True)
    fecha = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    ruta_zip = os.path.join(carpeta_salida, f"resultados_{persona}_{fecha}.zip")
    lineas = []
    with zipfile.ZipFile(ruta_zip + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for ruta, relativa in elegidos:
            with open(ruta, "rb") as f:
                datos = f.read()
            z.writestr(relativa, datos)
            lineas.append(f"{sha256_bytes(datos)}  {relativa}")
        z.writestr(MANIFIESTO, "\n".join(lineas) + "\n")
    os.replace(ruta_zip + ".tmp", ruta_zip)
    print(f"{ruta_zip}: {len(elegidos)} archivos")
    for linea in lineas:
        print("  ", linea.split("  ", 1)[1])
    return ruta_zip
