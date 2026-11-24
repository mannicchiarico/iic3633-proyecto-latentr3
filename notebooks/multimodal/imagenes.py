"""Imágenes de los ítems: cobertura, descarga y embeddings SigLIP (lo usa multimodal.ipynb).

- Cobertura: para cada ítem de items/<categoría>.csv, si tiene URL de imagen y si esa URL responde con una imagen.
- Descarga: baja la imagen de cada ítem, comprueba que PIL la abra y guarda un manifiesto con su sha256.
- Embeddings: SigLIP (imagen y título) en el mismo espacio, normalizados (norma L2 = 1).

Todo se puede retomar: cada paso se salta los ítems que ya están en su CSV de salida. Las decisiones están en README.md.
"""
import io
import os
import csv
import json
import time
import hashlib
import datetime
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

MODELO_SIGLIP = "google/siglip-so400m-patch14-384"   # decisión del usuario, 2026-11-24
LARGO_TEXTO = 64          # SigLIP se entrenó con textos rellenados a 64 tokens (padding="max_length")
HILOS = 16
TIMEOUT = 10              # segundos por intento
REINTENTOS = 2
# Cabecera genérica de navegador; no identifica al proyecto ni a las personas
CABECERAS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
COLUMNAS_COBERTURA = ["item_id", "asin", "campo", "url", "estado", "content_type", "bytes", "error", "fecha"]
COLUMNAS_DESCARGA = ["item_id", "asin", "campo", "url", "estado", "bytes", "sha256", "ancho", "alto", "error", "fecha"]


# ---------- Ítems y URLs ----------

def cargar_items(carpeta_items, categoria):
    df = pd.read_csv(os.path.join(carpeta_items, f"{categoria}.csv"))
    for campo in ("imageURLHighRes", "imageURL"):
        df[campo] = df[campo].apply(json.loads)
    assert df["item_id"].tolist() == list(range(len(df))), "items/<categoría>.csv debe tener los ids 0..n-1 en orden"
    return df


def elegir_url(fila):
    """La primera URL de alta resolución; si no hay, la primera de imageURL. Devuelve (campo, url) o (None, None)."""
    for campo in ("imageURLHighRes", "imageURL"):
        if fila[campo]:
            return campo, fila[campo][0]
    return None, None


def _pedir(url, leer):
    """GET con reintentos. Devuelve (estado, content_type, contenido o None, bytes, error)."""
    error = ""
    for intento in range(REINTENTOS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=CABECERAS), timeout=TIMEOUT) as r:
                contenido = r.read() if leer else None
                largo = len(contenido) if leer else int(r.headers.get("Content-Length") or 0)
                return r.status, r.headers.get("Content-Type", ""), contenido, largo, ""
        except urllib.error.HTTPError as e:
            if e.code < 500:        # 4xx: el enlace no existe; no tiene sentido reintentar
                return e.code, "", None, 0, f"HTTP {e.code}"
            error = f"HTTP {e.code}"
        except Exception as e:      # timeout, DNS, conexión
            error = f"{type(e).__name__}: {e}"[:200]
        time.sleep(1 + intento)
    return 0, "", None, 0, error


def _pendientes(df, ruta_csv):
    hechos = set(pd.read_csv(ruta_csv)["item_id"]) if os.path.isfile(ruta_csv) else set()
    return [fila for fila in df.to_dict("records") if fila["item_id"] not in hechos]


def _agregar(ruta_csv, columnas, filas):
    nuevo = not os.path.isfile(ruta_csv)
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        if nuevo:
            escritor.writeheader()
        escritor.writerows(filas)
        f.flush()


# ---------- Cobertura ----------

def revisar_urls(df, ruta_csv, hilos=HILOS, limite=None, cada=500):
    """Revisa si la URL elegida de cada ítem responde con una imagen (lee solo las cabeceras). Se puede retomar."""
    pendientes = _pendientes(df, ruta_csv)[:limite]

    def revisar(fila):
        campo, url = elegir_url(fila)
        hoy = datetime.date.today().isoformat()
        if url is None:
            return {"item_id": fila["item_id"], "asin": fila["asin"], "campo": "", "url": "", "estado": "",
                    "content_type": "", "bytes": 0, "error": "sin URL", "fecha": hoy}
        estado, tipo, _, largo, error = _pedir(url, leer=False)
        return {"item_id": fila["item_id"], "asin": fila["asin"], "campo": campo, "url": url, "estado": estado,
                "content_type": tipo, "bytes": largo, "error": error, "fecha": hoy}

    with ThreadPoolExecutor(hilos) as pool:
        for i in range(0, len(pendientes), cada):   # se escribe por bloques: una caída pierde a lo más un bloque
            _agregar(ruta_csv, COLUMNAS_COBERTURA, list(pool.map(revisar, pendientes[i:i + cada])))
            print(f"{os.path.basename(ruta_csv)}: {min(i + cada, len(pendientes))}/{len(pendientes)}", flush=True)
    return resumen_cobertura(ruta_csv, len(df))


def resumen_cobertura(ruta_csv, n_items):
    df = pd.read_csv(ruta_csv, keep_default_na=False)
    con_url = df["url"] != ""
    ok = con_url & (df["estado"].astype(str) == "200") & df["content_type"].str.startswith("image/")
    return {"items": n_items, "revisados": len(df), "con_url": int(con_url.sum()),
            "con_url_%": 100 * con_url.sum() / n_items, "url_responde_imagen": int(ok.sum()),
            "url_responde_imagen_%_de_items": 100 * ok.sum() / n_items,
            "url_caida_%_de_con_url": 100 * (con_url & ~ok).sum() / max(1, con_url.sum()),
            "campo_highres": int((df["campo"] == "imageURLHighRes").sum()),
            "campo_imageURL": int((df["campo"] == "imageURL").sum()),
            "fechas": sorted(set(df["fecha"]))}


# ---------- Descarga ----------

def descargar(df, carpeta, ruta_manifiesto, hilos=HILOS, limite=None, cada=500):
    """Descarga la imagen elegida de cada ítem a `carpeta`/<item_id>.img y comprueba que PIL la abra. Se puede retomar."""
    from PIL import Image
    os.makedirs(carpeta, exist_ok=True)
    pendientes = [f for f in _pendientes(df, ruta_manifiesto) if elegir_url(f)[1] is not None][:limite]

    def bajar(fila):
        campo, url = elegir_url(fila)
        registro = {"item_id": fila["item_id"], "asin": fila["asin"], "campo": campo, "url": url, "estado": "",
                    "bytes": 0, "sha256": "", "ancho": 0, "alto": 0, "error": "",
                    "fecha": datetime.date.today().isoformat()}
        estado, tipo, contenido, largo, error = _pedir(url, leer=True)
        registro.update(estado=estado, bytes=largo, error=error)
        if contenido is None:
            return registro
        try:
            with Image.open(io.BytesIO(contenido)) as imagen:
                imagen.load()
                registro.update(ancho=imagen.width, alto=imagen.height)
        except Exception as e:
            registro["error"] = f"PIL: {type(e).__name__}"
            return registro
        ruta = os.path.join(carpeta, f"{fila['item_id']}.img")
        with open(ruta + ".tmp", "wb") as f:
            f.write(contenido)
        os.replace(ruta + ".tmp", ruta)
        registro["sha256"] = hashlib.sha256(contenido).hexdigest()
        return registro

    with ThreadPoolExecutor(hilos) as pool:
        for i in range(0, len(pendientes), cada):
            _agregar(ruta_manifiesto, COLUMNAS_DESCARGA, list(pool.map(bajar, pendientes[i:i + cada])))
            print(f"descarga: {min(i + cada, len(pendientes))}/{len(pendientes)}", flush=True)
    manifiesto = pd.read_csv(ruta_manifiesto, keep_default_na=False)
    return {"con_url": int(sum(elegir_url(f)[1] is not None for f in df.to_dict("records"))),
            "descargadas": int((manifiesto["sha256"] != "").sum()), "fallidas": int((manifiesto["sha256"] == "").sum())}


def imagenes_validas(carpeta, ruta_manifiesto):
    """{item_id: ruta} de las imágenes descargadas cuyo sha256 coincide con el manifiesto."""
    manifiesto = pd.read_csv(ruta_manifiesto, keep_default_na=False)
    rutas = {}
    for fila in manifiesto[manifiesto["sha256"] != ""].to_dict("records"):
        ruta = os.path.join(carpeta, f"{fila['item_id']}.img")
        with open(ruta, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != fila["sha256"]:
                raise ValueError(f"{ruta} no coincide con el manifiesto")
        rutas[int(fila["item_id"])] = ruta
    return rutas


# ---------- Embeddings ----------

def cargar_siglip(modelo_id=MODELO_SIGLIP, revision=None, device="cuda"):
    """SigLIP en fp32 y su procesador (necesita sentencepiece para el tokenizer)."""
    import torch
    from transformers import AutoModel, AutoProcessor
    modelo = AutoModel.from_pretrained(modelo_id, revision=revision, torch_dtype=torch.float32).to(device).eval()
    procesador = AutoProcessor.from_pretrained(modelo_id, revision=revision)
    return modelo, procesador


def _normalizar(x):
    import torch
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()


def embeddings(df, rutas_imagenes, modelo, procesador, lote=32):
    """Embeddings de imagen (ítems con imagen válida; el resto queda en cero) y de título (todos), en el orden de los ids.

    Devuelve {"imagen": (n, D), "tiene_imagen": (n,), "titulo": (n, D)} en float32, normalizados.
    """
    import torch
    from PIL import Image
    device = next(modelo.parameters()).device
    n = len(df)
    titulos = df["title"].tolist()
    salida_t = []
    with torch.no_grad():
        for i in range(0, n, lote):
            entrada = procesador(text=titulos[i:i + lote], padding="max_length", max_length=LARGO_TEXTO,
                                 truncation=True, return_tensors="pt")
            salida_t.append(_normalizar(modelo.get_text_features(input_ids=entrada["input_ids"].to(device))))
        titulo = np.concatenate(salida_t)
        imagen = np.zeros_like(titulo)
        tiene = np.zeros(n, dtype=bool)
        ids = sorted(rutas_imagenes)
        for i in range(0, len(ids), lote):
            bloque = ids[i:i + lote]
            fotos = []
            for item_id in bloque:
                with Image.open(rutas_imagenes[item_id]) as foto:
                    fotos.append(foto.convert("RGB"))   # las PNG con transparencia o en paleta pasan a RGB
            entrada = procesador(images=fotos, return_tensors="pt")
            imagen[bloque] = _normalizar(modelo.get_image_features(pixel_values=entrada["pixel_values"].to(device)))
            tiene[bloque] = True
            if (i // lote) % 20 == 0:
                print(f"embeddings de imagen: {min(i + lote, len(ids))}/{len(ids)}", flush=True)
    return {"imagen": imagen.astype(np.float32), "tiene_imagen": tiene, "titulo": titulo.astype(np.float32)}


def guardar_embeddings(ruta_npz, emb, modelo, info_extra=None):
    """Guarda el .npz y un manifiesto JSON con su sha256, el modelo y su revisión (el commit resuelto de HF)."""
    np.savez(ruta_npz, **emb)
    with open(ruta_npz, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifiesto = {"archivo": os.path.basename(ruta_npz), "sha256": sha,
                  "modelo": getattr(modelo.config, "_name_or_path", None),
                  "revision": getattr(modelo.config, "_commit_hash", None), "dtype": "float32",
                  "dimension": int(emb["titulo"].shape[1]), "items": int(len(emb["titulo"])),
                  "items_con_imagen": int(emb["tiene_imagen"].sum()), "largo_texto": LARGO_TEXTO,
                  "normalizados": True, "fecha": datetime.date.today().isoformat(), **(info_extra or {})}
    ruta = os.path.join(os.path.dirname(ruta_npz), "manifiesto_embeddings.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=1, ensure_ascii=False)
    return manifiesto
