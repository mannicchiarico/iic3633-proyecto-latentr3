"""Preparación de una sesión de Colab (la llaman evaluar_bigrec_t4.ipynb y notebooks/multimodal/multimodal.ipynb).

Se puede correr en cada sesión nueva sin romper nada: instala solo lo que falta, clona solo lo que no está, rehace los
enlaces que no apuntan a donde deben y monta Drive si no está montado. El repo del proyecto ya lo clonó la primera
celda del notebook en /content/iic3633, en el commit fijado.

Frentes:
- "evaluaciones": evaluación de BIGRec y D3 en la GPU de Colab (necesita GPU).
- "multimodal": cobertura de imágenes (sin GPU), embeddings SigLIP y SASRec + SigLIP (con GPU).
"""
import os
import sys
import shutil
import subprocess
import importlib.metadata

REPO = "/content/iic3633"
CONTENIDO = "/content"
DRIVE = "/content/drive"
IIC3633 = os.path.join(DRIVE, "MyDrive", "IIC3633")

# Versiones de repo_latentr3/environment.yaml (transformers, accelerate) y de .venv-rocm (fire). torch no se toca
PAQUETES = {"evaluaciones": {"transformers": "4.51.3", "accelerate": "1.6.0", "fire": "0.7.1"},
            "multimodal": {"transformers": "4.51.3", "accelerate": "1.6.0", "sentencepiece": None}}
# Código externo, en el commit revisado (los .py verifican el sha256 de lo que cargan)
EXTERNOS = {"evaluaciones": [("DecodingMatters", "https://github.com/SAI990323/DecodingMatters",
                              "585c8427dd327a33155b5566f32bc22e21b5236c")],
            "multimodal": [("SASRec.pytorch", "https://github.com/pmixer/SASRec.pytorch",
                            "9f6a02b86a95e616fe815a0edc92dafa9911773d")]}
# Estructura que esperan los notebooks en Colab: todo suelto en /content
ENLACES = {"evaluaciones": {"baseline_comun.py": "notebooks/baseline_comun.py",
                            "bigrec_baseline.py": "notebooks/bigrec/bigrec_baseline.py",
                            "repo_latentr3": "repo_latentr3", "data_amazon_2018": "repo_latentr3/data"},
           "multimodal": {"baseline_comun.py": "notebooks/baseline_comun.py",
                          "sasrec_baseline.py": "notebooks/sasrec/sasrec_baseline.py",
                          "sasrec_mm.py": "notebooks/multimodal/sasrec_mm.py",
                          "imagenes.py": "notebooks/multimodal/imagenes.py",
                          "repo_latentr3": "repo_latentr3", "data_amazon_2018": "repo_latentr3/data"}}
ESPACIO_MIN_GB = {"evaluaciones": {"drive": 1, "contenido": 10}, "multimodal": {"drive": 3, "contenido": 10}}


def instalar(paquetes):
    """pip install de los paquetes que faltan o tienen otra versión; no reinstala torch."""
    pedir, importados = [], []
    for nombre, version in paquetes.items():
        try:
            actual = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual is None or (version is not None and actual != version):
            pedir.append(nombre if version is None else f"{nombre}=={version}")
            if nombre in sys.modules:
                importados.append(nombre)
    if pedir:
        print("Instalando:", " ".join(pedir))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pedir], check=True)
    if importados:
        raise RuntimeError(f"{importados} ya estaban importados con otra versión. Entorno de ejecución > Reiniciar "
                           "sesión, y volver a correr el notebook desde la primera celda.")
    for nombre, version in paquetes.items():
        actual = importlib.metadata.version(nombre)
        assert version is None or actual == version, (nombre, actual, version)
    return {nombre: importlib.metadata.version(nombre) for nombre in paquetes}


def clonar(destino, url, commit):
    if not os.path.isdir(os.path.join(destino, ".git")):
        subprocess.run(["git", "clone", "-q", url, destino], check=True)
    subprocess.run(["git", "-C", destino, "checkout", "-q", commit], check=True)
    cabeza = subprocess.run(["git", "-C", destino, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert cabeza == commit, (destino, cabeza, commit)


def enlazar(nombre, destino_en_repo):
    enlace, destino = os.path.join(CONTENIDO, nombre), os.path.join(REPO, destino_en_repo)
    assert os.path.exists(destino), destino
    if os.path.islink(enlace) and os.readlink(enlace) == destino:
        return
    if os.path.lexists(enlace):
        if not os.path.islink(enlace):
            raise RuntimeError(f"{enlace} existe y no es un enlace: moverlo o borrarlo a mano y volver a correr.")
        os.remove(enlace)
    os.symlink(destino, enlace)


def libre_gb(ruta):
    return shutil.disk_usage(ruta).free / 1e9


def preparar(frente):
    """Deja la sesión lista para un frente y devuelve las rutas que usan los notebooks."""
    assert frente in PAQUETES, frente
    assert os.path.isdir(os.path.join(REPO, ".git")), "Falta clonar el repo: correr la primera celda del notebook."
    versiones = instalar(PAQUETES[frente])
    for nombre, url, commit in EXTERNOS[frente]:
        clonar(os.path.join(CONTENIDO, nombre), url, commit)
    for nombre, destino in ENLACES[frente].items():
        enlazar(nombre, destino)
    os.chdir(CONTENIDO)
    for carpeta in (CONTENIDO, os.path.join(REPO, "colab"), os.path.join(REPO, "notebooks")):
        if carpeta not in sys.path:
            sys.path.insert(0, carpeta)

    from google.colab import drive
    if not os.path.isdir(os.path.join(DRIVE, "MyDrive")):
        drive.mount(DRIVE)
    os.makedirs(IIC3633, exist_ok=True)
    espacio = {"drive_libre_gb": round(libre_gb(os.path.join(DRIVE, "MyDrive")), 1),
               "contenido_libre_gb": round(libre_gb(CONTENIDO), 1)}
    minimo = ESPACIO_MIN_GB[frente]
    if espacio["contenido_libre_gb"] < minimo["contenido"]:
        raise RuntimeError(f"Quedan {espacio['contenido_libre_gb']} GB en /content; hacen falta {minimo['contenido']}.")
    if espacio["drive_libre_gb"] < minimo["drive"]:
        print(f"AVISO: quedan {espacio['drive_libre_gb']} GB en Drive (supuesto: la cifra refleja el cupo); hacen falta "
              f"~{minimo['drive']} GB. Vaciar la papelera de Drive o liberar espacio.")

    import torch
    from baseline_comun import verificar_gpu
    if frente == "evaluaciones":
        verificar_gpu("preparación", memoria_min_gib=14)   # BIGRec con lotes de 24 ocupa ~9,5 GB
    elif torch.cuda.is_available():
        verificar_gpu("preparación")
    else:
        print("Sin GPU: alcanza para la cobertura y la descarga; los embeddings y SASRec necesitan GPU (T4).")
    print("Versiones:", versiones, "| torch", torch.__version__, "|", espacio)
    return {"repo": REPO, "base": os.path.join(CONTENIDO, "data_amazon_2018"), "iic3633": IIC3633,
            "codigo_d3": os.path.join(CONTENIDO, "DecodingMatters"),
            "codigo_sasrec": os.path.join(CONTENIDO, "SASRec.pytorch"),
            "calc_py": os.path.join(REPO, "repo_latentr3", "src", "utils", "calc.py"), **espacio}
