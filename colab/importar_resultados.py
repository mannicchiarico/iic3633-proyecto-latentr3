"""Copia a resultados/ el zip que exporta Colab, verificando cada archivo (se corre en local, desde la raíz del repo).

    python colab/importar_resultados.py resultados_evaluaciones_20261125-1830.zip [--seco] [--sobrescribir] [--sin-metricas]

1. Cada archivo del zip debe estar en la lista blanca (lista_blanca.py) y en el MANIFIESTO.sha256, con el mismo sha256.
2. No pisa un archivo de resultados/ que tenga otro contenido, salvo con --sobrescribir.
3. Después de escribir vuelve a calcular el sha256 de cada archivo.
4. Corre gao() de calc.py de los autores sobre cada final_result.json y comprueba que dé lo mismo que las métricas
   propias y 0 títulos fuera del catálogo (bigrec_baseline.verificar_con_calc).
"""
import os
import sys
import json
import zipfile
import argparse

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lista_blanca import MANIFIESTO, permitido, sha256_bytes   # noqa: E402


def leer_zip(ruta_zip):
    """{ruta: bytes} del zip, verificado contra la lista blanca y el manifiesto. Falla ante cualquier diferencia."""
    with zipfile.ZipFile(ruta_zip) as z:
        nombres = [n for n in z.namelist() if not n.endswith("/")]
        if MANIFIESTO not in nombres:
            raise ValueError(f"El zip no trae {MANIFIESTO}.")
        manifiesto = {}
        for linea in z.read(MANIFIESTO).decode("utf-8").splitlines():
            if linea.strip():
                sha, ruta = linea.split("  ", 1)
                manifiesto[ruta] = sha
        archivos = {n: z.read(n) for n in nombres if n != MANIFIESTO}
    errores = [f"fuera de la lista blanca: {n}" for n in archivos if not permitido(n)]
    errores += [f"no está en el manifiesto: {n}" for n in archivos if n not in manifiesto]
    errores += [f"en el manifiesto pero no en el zip: {n}" for n in manifiesto if n not in archivos]
    errores += [f"sha256 distinto del manifiesto: {n}" for n, datos in archivos.items()
                if n in manifiesto and sha256_bytes(datos) != manifiesto[n]]
    if errores:
        raise ValueError("Zip rechazado:\n  " + "\n  ".join(errores))
    return archivos


def importar(ruta_zip, raiz=RAIZ, seco=False, sobrescribir=False):
    archivos = leer_zip(ruta_zip)
    nuevos, iguales, conflictos = [], [], []
    for ruta, datos in archivos.items():
        destino = os.path.join(raiz, *ruta.split("/"))
        if os.path.isfile(destino):
            with open(destino, "rb") as f:
                (iguales if sha256_bytes(f.read()) == sha256_bytes(datos) else conflictos).append(ruta)
        else:
            nuevos.append(ruta)
    if conflictos and not sobrescribir:
        raise ValueError("Estos archivos ya existen con otro contenido (usar --sobrescribir si es a propósito):\n  "
                         + "\n  ".join(conflictos))
    escribir = nuevos + conflictos
    if not seco:
        for ruta in escribir:
            destino = os.path.join(raiz, *ruta.split("/"))
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with open(destino + ".tmp", "wb") as f:
                f.write(archivos[ruta])
            os.replace(destino + ".tmp", destino)
            with open(destino, "rb") as f:
                assert sha256_bytes(f.read()) == sha256_bytes(archivos[ruta]), destino
    return {"nuevos": nuevos, "sin cambios": iguales, "sobrescritos": conflictos if not seco else [],
            "escritos": [] if seco else escribir}


def verificar_metricas(rutas, raiz=RAIZ):
    """verificar_con_calc + metricas de cada final_result.json importado."""
    sys.path[:0] = [os.path.join(raiz, "notebooks"), os.path.join(raiz, "notebooks", "bigrec")]
    import bigrec_baseline as bb
    base = os.path.join(raiz, "repo_latentr3", "data")
    calc_py = os.path.join(raiz, "repo_latentr3", "src", "utils", "calc.py")
    resultados = {}
    for ruta in rutas:
        if not ruta.endswith("final_result.json"):
            continue
        categoria = ruta.split("/")[2]
        destino = os.path.join(raiz, *ruta.split("/"))
        info = bb.buscar_archivo(base, "info", categoria)
        bb.verificar_con_calc(destino, info, calc_py)
        m = bb.metricas(destino, info)
        assert m["fuera del catálogo"] == 0, (ruta, m["fuera del catálogo"])
        resultados[ruta] = {k: m[k] for k in ("HR@5", "HR@10", "NDCG@5", "NDCG@10", "filas", "fuera del catálogo")}
    return resultados


def main():
    parser = argparse.ArgumentParser(description="Importa a resultados/ un zip exportado desde Colab.")
    parser.add_argument("zip")
    parser.add_argument("--seco", action="store_true", help="solo verificar, sin escribir")
    parser.add_argument("--sobrescribir", action="store_true")
    parser.add_argument("--sin-metricas", action="store_true", help="no correr calc.py sobre los final_result.json")
    args = parser.parse_args()
    resumen = importar(args.zip, seco=args.seco, sobrescribir=args.sobrescribir)
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    if not args.seco and not args.sin_metricas:
        rutas = resumen["nuevos"] + resumen["sin cambios"] + resumen["sobrescritos"]
        print(json.dumps(verificar_metricas(rutas), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
