# Resultados versionados

Copia exacta (mismo sha256) de los archivos de resultados que los notebooks escriben en `checkpoints/`, que no se versiona porque también contiene los pesos de los modelos (~3 GB). La estructura de carpetas es la misma que en `checkpoints/`. Los archivos se copiaron el 2026-11-06; si se vuelve a correr un experimento, hay que copiarlos de nuevo.

| Ruta | Contenido | Lo genera |
|---|---|---|
| `bigrec/Toys_and_Games/entrenamiento.json` | Log del SFT de BIGRec: pérdida por época, tiempos, memoria y configuración | `notebooks/bigrec/baseline_bigrec.ipynb` |
| `bigrec/Toys_and_Games/predicciones/{bigrec,d3}/registro.json` | Tiempo, filas y memoria de la evaluación | `notebooks/bigrec/baseline_bigrec.ipynb`, `notebooks/d3/baseline_d3.ipynb` |
| `bigrec/Toys_and_Games/predicciones/{bigrec,d3}/final_result.json` | Top-10 por fila de test (entrada de `repo_latentr3/src/utils/calc.py`) | Los mismos |
| `base_cot/<categoría>/predicciones/{base,cot_256}/registro.json` y `final_result.json` | Lo mismo para Base (4 categorías) y COT (Toys) | `notebooks/base_cot/base_cot.ipynb` |
| `latentr3/_tiempo/*.json`, `latentr3/_tiempo_rl/tiempos.json` | Prueba de tiempo y memoria de LatentR3 en Toys (SFT, generación y RL) | Funciones de medición de una versión anterior de `notebooks/latentr3/latentr3.py` (p. ej. `_tiempo_sft`, citada en el traceback de `_tiempo/tiempos.json`), que ya no están en el archivo |

No se incluyen los pesos (`modelo/`), los estados del entrenador (`trainer*/`) ni las partes intermedias (`parte_*.json`), que se pueden volver a generar.
