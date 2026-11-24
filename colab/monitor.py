"""Registro del uso de la GPU durante una sesión de Colab.

Cada 60 s agrega a <carpeta>/gpu_log.csv una muestra de nvidia-smi (hora, GPU, utilización y memoria usada) con la
etapa en curso, y al terminar cada etapa agrega a <carpeta>/etapas.json su duración, su estado y el pico de
torch.cuda.max_memory_allocated(). La carpeta va en Drive para que el registro sobreviva a una desconexión.
"""
import os
import csv
import json
import time
import datetime
import threading
import subprocess
import contextlib

CONSULTA = ["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used", "--format=csv,noheader"]
ENCABEZADO = ["etapa", "timestamp", "name", "utilization.gpu [%]", "memory.used [MiB]"]


class Monitor:
    def __init__(self, carpeta, intervalo=60, comando=CONSULTA):
        os.makedirs(carpeta, exist_ok=True)
        self.ruta_log = os.path.join(carpeta, "gpu_log.csv")
        self.ruta_etapas = os.path.join(carpeta, "etapas.json")
        self.intervalo, self.comando = intervalo, comando
        self.etapa_actual = "sin etapa"
        self._parar = threading.Event()
        self._hilo = None

    def _muestra(self):
        try:
            salida = subprocess.run(self.comando, capture_output=True, text=True, timeout=20)
            lineas = [l for l in salida.stdout.strip().splitlines() if l.strip()] or [f"error: {salida.stderr.strip()}"]
        except Exception as e:   # sin nvidia-smi (p. ej. entorno sin GPU)
            lineas = [f"error: {type(e).__name__}"]
        nuevo = not os.path.isfile(self.ruta_log)
        with open(self.ruta_log, "a", newline="") as f:
            escritor = csv.writer(f)
            if nuevo:
                escritor.writerow(ENCABEZADO)
            for linea in lineas:
                escritor.writerow([self.etapa_actual] + [c.strip() for c in linea.split(",")])

    def _bucle(self):
        self._muestra()
        while not self._parar.wait(self.intervalo):
            self._muestra()

    def iniciar(self):
        if self._hilo is None or not self._hilo.is_alive():
            self._parar.clear()
            self._hilo = threading.Thread(target=self._bucle, daemon=True)
            self._hilo.start()
        return self

    def detener(self):
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=30)

    @contextlib.contextmanager
    def etapa(self, nombre):
        """with monitor.etapa("Video_Games bigrec"): ...  registra duración, estado y pico de memoria de la etapa."""
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        self.etapa_actual, inicio, estado = nombre, time.time(), "error"
        try:
            yield
            estado = "ok"
        finally:
            entrada = {"etapa": nombre, "inicio": datetime.datetime.fromtimestamp(inicio).isoformat(timespec="seconds"),
                       "segundos": round(time.time() - inicio, 1), "estado": estado,
                       "pico_memoria_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None}
            etapas = []
            if os.path.isfile(self.ruta_etapas):
                with open(self.ruta_etapas) as f:
                    etapas = json.load(f)
            etapas.append(entrada)
            with open(self.ruta_etapas + ".tmp", "w") as f:
                json.dump(etapas, f, indent=1, ensure_ascii=False)
            os.replace(self.ruta_etapas + ".tmp", self.ruta_etapas)
            self.etapa_actual = "sin etapa"
            print(f"[monitor] {nombre}: {estado}, {entrada['segundos'] / 60:.1f} min, pico {entrada['pico_memoria_gib']} GiB")
