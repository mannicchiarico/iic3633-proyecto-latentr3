"""Tabla de ítems con sus URLs de imagen, desde la metadata de Amazon 2018 (paso 1 de la parte multimodal).

Se corre una vez en local, donde está raw/ (lo genera notebooks/datos.ipynb), y el resultado se versiona en
items/<categoría>.csv, para que Colab no tenga que descargar varios GB de metadata:

    python extraer_items.py --raw ../../raw --base ../../repo_latentr3/data --salida items

El cruce asin → item_id sale de los CSV de train, valid y test (columnas item_asin/item_id e
item_asins/history_item_id), y se comprueba que cubra todos los ids de info/. Si un asin aparece más de una vez en
la metadata, queda la última línea, como hace gao() de process.py con los títulos.
"""
import os
import ast
import csv
import gzip
import json
import hashlib
import argparse
import datetime
import pandas as pd

CATEGORIAS = ["CDs_and_Vinyl", "Musical_Instruments", "Toys_and_Games", "Video_Games"]
COLUMNAS = ["item_id", "asin", "title", "n_imageURLHighRes", "n_imageURL", "imageURLHighRes", "imageURL"]


def sha256_archivo(ruta, bloque=2**24):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while True:
            datos = f.read(bloque)
            if not datos:
                return h.hexdigest()
            h.update(datos)


def buscar(base, split, categoria):
    carpeta = os.path.join(base, split)
    archivos = [f for f in os.listdir(carpeta) if f.startswith(categoria)]
    assert len(archivos) == 1, (carpeta, categoria, archivos)
    return os.path.join(carpeta, archivos[0])


def titulos_info(base, categoria):
    with open(buscar(base, "info", categoria), encoding="utf-8") as f:
        pares = [linea.rstrip("\n").rsplit("\t", 1) for linea in f if linea.strip()]
    return {int(i): t for t, i in pares}


def asin_por_id(base, categoria):
    """{item_id: asin} desde los CSV; falla si un id tiene dos asin o un asin dos ids."""
    pares = set()
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(buscar(base, split, categoria))
        pares.update(zip(df["item_id"], df["item_asin"]))
        for ids, asins in zip(df["history_item_id"], df["item_asins"]):
            pares.update(zip(ast.literal_eval(ids), ast.literal_eval(asins)))
    por_id, por_asin = {}, {}
    for item_id, asin in pares:
        assert por_id.setdefault(int(item_id), asin) == asin, f"el id {item_id} tiene dos asin"
        assert por_asin.setdefault(asin, int(item_id)) == int(item_id), f"el asin {asin} tiene dos ids"
    return por_id


def normalizar_titulo(titulo):
    # Como gao() en process.py L55
    return titulo.replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")


def extraer(raw, base, categoria, salida):
    por_id = asin_por_id(base, categoria)
    info = titulos_info(base, categoria)
    faltan = sorted(set(info) - set(por_id))
    assert not faltan, f"{categoria}: {len(faltan)} ids de info/ no aparecen en los CSV (p. ej. {faltan[:5]})"
    buscados = {asin: item_id for item_id, asin in por_id.items()}
    meta, lineas, repetidos = {}, 0, 0
    ruta_meta = os.path.join(raw, f"meta_{categoria}.json.gz")
    with gzip.open(ruta_meta, "rt", encoding="utf-8") as f:
        for linea in f:
            lineas += 1
            m = json.loads(linea)
            if m.get("asin") in buscados:
                repetidos += m["asin"] in meta
                meta[m["asin"]] = {"title": m.get("title", ""), "imageURLHighRes": m.get("imageURLHighRes") or [],
                                   "imageURL": m.get("imageURL") or []}
    sin_meta = [i for i, a in por_id.items() if a not in meta]
    assert not sin_meta, f"{categoria}: {len(sin_meta)} ítems sin línea en la metadata"
    filas, titulo_distinto = [], 0
    for item_id in sorted(info):
        m = meta[por_id[item_id]]
        titulo_distinto += normalizar_titulo(m["title"]) != info[item_id]
        filas.append({"item_id": item_id, "asin": por_id[item_id], "title": info[item_id],
                      "n_imageURLHighRes": len(m["imageURLHighRes"]), "n_imageURL": len(m["imageURL"]),
                      "imageURLHighRes": json.dumps(m["imageURLHighRes"]), "imageURL": json.dumps(m["imageURL"])})
    os.makedirs(salida, exist_ok=True)
    ruta = os.path.join(salida, f"{categoria}.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    con_imagen = sum(1 for fila in filas if fila["n_imageURLHighRes"] or fila["n_imageURL"])
    return {"items": len(filas), "con_url_de_imagen": con_imagen, "sin_url_de_imagen": len(filas) - con_imagen,
            "lineas_meta": lineas, "asin_repetidos_en_meta": repetidos,
            "titulo_meta_distinto_de_info": titulo_distinto,
            "sha256_meta": sha256_archivo(ruta_meta), "sha256_csv": sha256_archivo(ruta),
            "sha256_info": sha256_archivo(buscar(base, "info", categoria))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default=os.path.join("..", "..", "raw"))
    parser.add_argument("--base", default=os.path.join("..", "..", "repo_latentr3", "data"))
    parser.add_argument("--salida", default="items")
    parser.add_argument("--categorias", nargs="+", default=CATEGORIAS)
    args = parser.parse_args()
    resumen = {"fecha": datetime.date.today().isoformat(), "categorias": {}}
    for categoria in args.categorias:
        resumen["categorias"][categoria] = extraer(args.raw, args.base, categoria, args.salida)
        print(categoria, json.dumps(resumen["categorias"][categoria], ensure_ascii=False))
    with open(os.path.join(args.salida, "fuente.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
