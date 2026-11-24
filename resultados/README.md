# Resultados versionados

Copia exacta (mismo sha256) de los archivos de resultados que los notebooks escriben en `checkpoints/`, que no se versiona porque también contiene los pesos de los modelos (~3 GB). La estructura de carpetas es la misma que en `checkpoints/`. Los archivos se copiaron el 2026-11-06; si se vuelve a correr un experimento, hay que copiarlos de nuevo.

| Ruta | Contenido | Lo genera |
|---|---|---|
| `bigrec/Toys_and_Games/entrenamiento.json` | Log del SFT de BIGRec: pérdida por época, tiempos, memoria y configuración | `notebooks/bigrec/baseline_bigrec.ipynb` |
| `bigrec/Toys_and_Games/predicciones/{bigrec,d3}/registro.json` | Tiempo, filas y memoria de la evaluación | `notebooks/bigrec/baseline_bigrec.ipynb`, `notebooks/d3/baseline_d3.ipynb` |
| `bigrec/Toys_and_Games/predicciones/{bigrec,d3}/final_result.json` | Top-10 por fila de test (entrada de `repo_latentr3/src/utils/calc.py`) | Los mismos |
| `base_cot/<categoría>/predicciones/{base,cot_256}/registro.json` y `final_result.json` | Lo mismo para Base (4 categorías) y COT (Toys) | `notebooks/base_cot/base_cot.ipynb` |
| `bigrec/{Video_Games,Musical_Instruments}/predicciones/{bigrec,d3}/registro.json` y `final_result.json` | Evaluación de BIGRec y D3 en Colab (T4, fp16; el `registro.json` trae `dtype` y la GPU de cada sesión) | `colab/evaluar_bigrec_t4.ipynb` |
| `bigrec/Toys_and_Games/predicciones/bigrec_t4/` y `colab/evaluaciones/validacion_t4.json` | Validación de la GPU de Colab: BIGRec Toys evaluado en la T4 frente a la corrida local | `colab/evaluar_bigrec_t4.ipynb` |
| `multimodal/cobertura/`, `multimodal/Toys_and_Games/` | Cobertura de imágenes (4 datasets), manifiestos de imágenes y embeddings, y resultados de SASRec + SigLIP | `notebooks/multimodal/multimodal.ipynb` |
| `colab/{evaluaciones,multimodal}/gpu_log.csv`, `etapas.json` | Uso de la GPU cada 60 s y duración y pico de memoria de cada etapa en Colab | `colab/monitor.py` |
| `latentr3/_tiempo/*.json`, `latentr3/_tiempo_rl/tiempos.json` | Prueba de tiempo y memoria de LatentR3 en Toys (SFT, generación y RL) | Funciones de medición de una versión anterior de `notebooks/latentr3/latentr3.py` (p. ej. `_tiempo_sft`, citada en el traceback de `_tiempo/tiempos.json`), que ya no están en el archivo |

No se incluyen los pesos (`modelo/`), los estados del entrenador (`trainer*/`) ni las partes intermedias (`parte_*.json`), que se pueden volver a generar. Tampoco las imágenes ni los embeddings (`siglip.npz`); sus manifiestos traen el sha256.

Lo que viene de Colab se copia con `python colab/importar_resultados.py <zip>`, que solo acepta los archivos de `colab/lista_blanca.py` y verifica el sha256 de cada uno contra el manifiesto del zip.
