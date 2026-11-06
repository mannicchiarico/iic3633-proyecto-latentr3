# BIGRec (`baseline_bigrec.ipynb`)

Baseline BIGRec de LatentR3: Qwen2.5-1.5B con ajuste completo, entrenado para escribir el título del siguiente ítem y evaluado con decodificación restringida al catálogo. Corresponde a la fila BIGRec de la Tabla 1 del paper, que según la Tabla 2 ("w/o Reasoning") da los mismos números que LatentR3 sin el token latente. Mide HR@5, HR@10, NDCG@5 y NDCG@10 por título.

**Estado:** corrido solo en Toys_and_Games (SFT y evaluación; celdas 15, 19 y 21). CDs_and_Vinyl, Musical_Instruments y Video_Games no están corridos.

`bigrec_baseline.py` también implementa la variante D3 (`LENGTH_PENALTY["d3"] = 0.0`), que usa el notebook de [`../d3/README.md`](../d3/README.md) sobre el mismo modelo entrenado aquí.

## Fuente y versión

- Repo de origen: [SAI990323/DecodingMatters](https://github.com/SAI990323/DecodingMatters) (D3, la reimplementación con Qwen de BIGRec hecha por sus autores), commit `585c8427dd327a33155b5566f32bc22e21b5236c`.
- Se cargan sin modificar dos archivos, con estos sha256 (el notebook los verifica antes de importar, celda 4):

  | Archivo | sha256 |
  |---|---|
  | `code/dataset.py` (`D3Dataset`: prompt y etiquetas) | `2b0ea46ef8cab6f9f97516e050c219a009405a4698c1e5da0ffef8aaf9f8b706` |
  | `code/LogitProcesser.py` (`CFEnhancedLogitsProcessor`: decodificación restringida) | `f0b52d7accfca944641274bd67ad4e3b52ce4cd36224659cd2a94ed5ad73bae6` |

- Clonado en la raíz del repo (carpeta ignorada por git; el repo de origen no tiene licencia, así que el código se usa sin modificarlo ni redistribuirlo):

  ```bash
  git clone https://github.com/SAI990323/DecodingMatters DecodingMatters
  git -C DecodingMatters checkout 585c8427dd327a33155b5566f32bc22e21b5236c
  sha256sum DecodingMatters/code/dataset.py DecodingMatters/code/LogitProcesser.py
  ```

- Por qué D3 y no LatentR3: el código de LatentR3 no permite quitar el token latente sin modificarlo (su dataset siempre agrega `<|Thought|>`, el modelo hace dos pases y el trie inserta el token). Su código deriva de D3. `D3Dataset` es igual a `LatentRDataset` salvo la línea que agrega `<|Thought|>`, y `CFEnhancedLogitsProcessor` es igual a la de LatentR3 salvo que lee los últimos 4 tokens del prompt en vez de 5. Decisión del proyecto.
- Los argumentos de entrenamiento son los de `latent_attention_train.py` de LatentR3, que coinciden con los de `code/train.py` de D3. El BIGRec original ([SAI990323/BIGRec](https://github.com/SAI990323/BIGRec): LLaMA-7B, LoRA en 8 bits, otro prompt, grounding por embeddings) no es lo que mide la Tabla 1 y solo sirve de referencia.
- Código propio: `bigrec_baseline.py` (sha256 `e9fc8dc8a366930c3780b3cf25087f8dccffc2c25ac7a08a99d638bd585eafb3`), que importa de [`../baseline_comun.py`](../baseline_comun.py) la semilla, la búsqueda de archivos y la carga verificada del código de D3.

## Configuración

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo | `Qwen/Qwen2.5-1.5B` (base), ajuste completo, bf16, sin tokens nuevos | paper, App. B |
| Prompt | El de LatentR3 sin `<|Thought|>`; igual en tokens, etiquetas y máscara en 200 filas de cada split | medido, celda 8 |
| Que la Tabla 1 use ese prompt | El paper no publica el prompt de BIGRec | deducción del proyecto (base: el código de LatentR3 deriva del de D3, los papers comparten autores y "w/o Reasoning" = BIGRec) |
| Loss | Entropía cruzada solo sobre el título y EOS; el prompt lleva -100 | paper, Eq. 4; `D3Dataset` |
| lr | 3e-4, sin grilla (la grilla del paper es {3e-3, 3e-4, 3e-5}) | decisión del proyecto; lo confirma el autor en el issue #2 del repo de LatentR3 |
| Batch | 128 efectivo | issue #2 y código de los autores; el paper no lo especifica |
| Micro-batch usado en Toys | 2, acumulación 64, gradient checkpointing, `adamw_torch_fused` (`CONFIG_MEMORIA`, celda 3) | medido, `resultados/bigrec/Toys_and_Games/entrenamiento.json` |
| Micro-batch base del `.py` | 4 con `adamw_torch`; la prueba rápida lo elegía en CDs, pero en Toys no cupo | `bigrec_baseline.py` (`MICRO_BATCH`); ver Problemas conocidos |
| Épocas y parada | Hasta 3 épocas; loss de `valid` por época; paciencia 1; se queda la época de menor loss | paper, App. B (paciencia 1); código de los autores |
| Warmup y scheduler | 20 pasos y bajada lineal (default de HF) | código de los autores |
| Semilla | 42 (default del Trainer); una corrida | repo de los autores; paper, App. E (corrida determinista) |
| Datos | CSV de `repo_latentr3/data/`, una muestra por fila. Toys: 53898 train, 6737 valid, 6738 test, 6299 ítems | medido, celda 6 (coincide con la Tabla 3 del paper) |
| Candidatos | Trie con todos los títulos de `info/` (6271 títulos únicos, 6271 caminos); no se excluye el historial | medido, celda 9; `latent_attention_eval.py` |
| Decodificación | 10 beams, 10 secuencias, `length_penalty` 1.0, `max_new_tokens` 128, lotes de 24 | repo de los autores, `latent_attention_eval.py` y `latent_train.sh` |
| Criterio de acierto | Por título, normalizado igual que `repo_latentr3/src/utils/calc.py`; se verifica contra `calc.py` sobre la misma salida | decisión del proyecto |
| Hardware y entorno | AMD Radeon RX 9070 XT (16 GB, solo la GPU dedicada visible), torch 2.14.1+rocm7.14, transformers 4.51.3, bf16 | medido, celda 4 y `entrenamiento.json` |

Los autores usaron 2 A100 con torch 2.6 y CUDA (paper, App. B).

## Adaptaciones respecto de la fuente

| Cambio | Motivo | Efecto esperado en los resultados |
|---|---|---|
| Modelo Qwen2.5-1.5B con ajuste completo en vez de LLaMA-7B con LoRA | Es lo que mide la Tabla 1 | Ninguno respecto de LatentR3 |
| Prompt `D3Dataset` en vez del Alpaca de BIGRec | Es lo que queda de LatentR3 al quitar el token latente | Ninguno respecto de LatentR3 sin token |
| Paciencia 1 (BIGRec original: 5) | Paper y código de LatentR3 | Puede parar en la época 2 |
| `entrenar` en `bigrec_baseline.py` con los `TrainingArguments` de LatentR3 | `code/train.py` de D3 no corre tal cual: pasa `candidate_file`, que `D3Dataset` no acepta, e importa `code.dataset`, que choca con el módulo `code` de Python | Ninguno; los argumentos son los mismos |
| 1 GPU de 16 GB con gradient checkpointing, AdamW fused y micro-batch 2 (autores: ~25 GB por GPU, 2 GPUs, issue #2) | No cabe de otro modo | Mismo batch efectivo y mismo algoritmo; cambia el orden de los datos y el redondeo |
| Generación como LatentR3 (10 beams, lotes de 24 con `attention_mask`, `max_new_tokens` 128); D3 usa 64 y lote de 1 | Es la evaluación de la Tabla 1 | Los lotes con padding pueden cambiar el top-10 por redondeo en bf16 (ver Limitaciones) |
| Trie con raíz de 4 tokens (`### Response: \n`); `construir_trie` copia la lógica de D3, que está dentro de su `main` | No hay token latente | Ninguno |
| No se piden `output_scores` | Solo ocupan memoria (unos 146 MB por paso con lotes de 24) | Ninguno; el orden de los beams es el mismo |
| Se decodifican solo los tokens generados | No depender de que un título contenga "Response:" | Mismo título tras la normalización de `calc.py` |
| Salida en partes de 256 filas y `final_result.json` | Permite retomar; el formato es el que lee `calc.py` | Ninguno |

## Limitaciones y desviaciones respecto del paper

- Entorno distinto: torch 2.14 con ROCm en vez de 2.6 con CUDA. El autor advierte que el resultado cambia con el entorno (issue #2).
- 1 GPU de 16 GB en vez de 2 A100: el SFT solo cambia en la acumulación, pero cambian el orden de los datos y el redondeo.
- Una semilla (42) y kernels de GPU no deterministas.
- lr por dataset no especificado en el paper: se usa 3e-4 en todos, sin grilla.
- Que la Tabla 1 use el prompt de D3 es una deducción del proyecto.
- bf16 obligatorio: el notebook se detiene si la GPU no lo soporta (no sirve una T4).
- Retomar un entrenamiento interrumpido pierde hasta una época, porque los checkpoints se guardan por época.
- Solo hay 10 predicciones por fila, así que HR@20 y NDCG@20 son iguales a HR@10 y NDCG@10.
- El criterio es solo por título: un modelo generativo no distingue ítems con el mismo título, así que no hay métricas por id.
- Con lotes de 24 y de 1 el top-10 de una misma fila puede diferir: en la prueba rápida en CDs, 82 de 100 filas (medido fuera del notebook, sin output guardado; ver Resultados).
- Faltan CDs, Instruments y Games: el baseline no está completo.

## Cómo correrlo

1. Entorno: `.venv-rocm` o `.venv-cuda` (ver el README de la raíz), con `transformers==4.51.3`, `accelerate==1.6.0` y `fire`. Directorio de trabajo: `notebooks/bigrec/`, que es el default de Jupyter y VS Code.
2. Clonar `DecodingMatters/` en la raíz (ver Fuente y versión).
3. Tener `Qwen/Qwen2.5-1.5B` en la caché de Hugging Face (ver el README de la raíz). En local el notebook trabaja sin conexión (`HF_HUB_OFFLINE=1`).
4. Editar `DATASETS` en la celda 3. Está en `['toys_data']`; los nombres válidos son `cd_data`, `music_data`, `toys_data` y `games_data`. Cada dataset entrena su propio modelo desde el modelo base.
5. Si el equipo tiene GPU integrada, la celda 2 deja visible solo la GPU 0 en local, antes de importar `torch`. Si el kernel ya importó `torch`, hay que reiniciarlo. Sin esto el Trainer reparte el modelo con `DataParallel` y se agota la memoria.
6. Ejecutar en orden. Las celdas 10–13 (prueba rápida) son opcionales: `CONFIG_MEMORIA` ya está fijada en la celda 3 (micro-batch 2, checkpointing, `adamw_torch_fused`), y la prueba rápida, si se corre, la sobrescribe. Luego: 15 (entrenamiento), 17 (evaluación), 19 (métricas) y 21 (comparación con el paper).
7. Retomar: volver a correr la celda. El entrenamiento continúa desde el último checkpoint y se salta si `entrenamiento.json` dice que terminó; la evaluación se salta las partes ya escritas.
8. Colab: subir `bigrec_baseline.py`, `baseline_comun.py` y `repo_latentr3/src/`, clonar `DecodingMatters/` en el directorio de trabajo y usar una GPU con bf16 (A100 o L4). Las salidas van a `MyDrive/IIC3633/checkpoints/bigrec/`.
9. Prueba rápida (celdas 10–13): entrena 8 pasos en CDs con 1024 filas, elige la configuración de memoria, evalúa 100 filas de test con lotes de 24 y de 1, verifica contra `calc.py` y estima la duración por dataset. Escribe en `checkpoints/bigrec/_prueba/` y lo borra al final.

Archivos que escribe (excepción a la convención de que los notebooks no escriben archivos), en `checkpoints/bigrec/<categoría>/`, ignorado por git:
- `trainer/`: checkpoints por época de ~9 GB, como máximo 2; se borran al terminar.
- `modelo/`: la mejor época, ~3,1 GB; se conserva porque la evaluación y D3 lo usan.
- `entrenamiento.json`: loss de `valid` por época, tiempos, memoria y versiones.
- `predicciones/bigrec/`: partes, `registro.json` y `final_result.json`.

## Costos medidos

| Etapa | GPU | Memoria GPU | RAM | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|---|---|
| SFT Toys (1263 pasos, 53898 filas) | RX 9070 XT | 12,996098 GB asignados (máx.), 15,296875 GB reservados (máx.) | no medido | 10166,66 s (2,824073 h); mediana de 7,86 s por paso de 128 filas | medido | `resultados/bigrec/Toys_and_Games/entrenamiento.json`; celda 15 |
| Evaluación de `valid` en el SFT | RX 9070 XT | incluida en el SFT | no medido | ~73,5 s por evaluación (6737 filas, 0,0109 s por fila) | medido | `entrenamiento.json` |
| Evaluación Toys (BIGRec, 6738 filas, lotes de 24) | RX 9070 XT | 9,54 GB (máx.) | no medido | 1724,18 s (~0,256 s por fila) | medido | `resultados/bigrec/Toys_and_Games/predicciones/bigrec/registro.json`; celda 17 |
| Evaluación Toys (D3, mismo modelo, `length_penalty` 0.0) | RX 9070 XT | 9,47 GB (máx.) | no medido | 1756,83 s | medido | `resultados/bigrec/Toys_and_Games/predicciones/d3/registro.json` |
| SFT y evaluación de CDs, Instruments y Games | | | | no medido | no corrido | |

Prueba rápida en CDs (config. micro-batch 4, `adamw_torch`; medido fuera del notebook, sin output guardado): 6,85 s por paso, 14,54 GB asignados y 15,17 GB reservados, 0,0110 s por fila de `valid`, 0,175 s por fila de test con lotes de 24 (7,58 GB) y 0,356 s con lotes de 1 (3,26 GB). Las horas por dataset que estimaba la prueba tampoco tienen output guardado; el SFT real de Toys (2,82 h con micro-batch 2) es la única medición completa.

## Resultados

Toys_and_Games, una corrida, semilla 42 (medido, celdas 19 y 21).

| Fuente | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| Nuestro BIGRec | 0.0770 | 0.1060 | 0.0563 | 0.0657 |
| Paper, Tabla 1 (BIGRec = Tabla 2 "w/o Reasoning") | 0.0701 | 0.0931 | 0.0508 | 0.0582 |

Nuestro resultado queda por encima del paper en las cuatro métricas.

Otras métricas de la celda 19: HR@1 0.0352, HR@3 0.0591, NDCG@1 0.0352, NDCG@3 0.049; HR@20 0.106 y NDCG@20 0.0657 (iguales a las @10, solo hay 10 predicciones).

Verificaciones de Toys (celda 19 y `entrenamiento.json`):
- 0 títulos fuera del catálogo, 0 repetidos y 6738 filas.
- `eval_loss` de `valid` por época: 0,731 (época 1), 0,5748 (época 2), 0,6258 (época 3, 2,995). La mejor es la época 2 (`checkpoint-844`, 0,574830174446106). Hubo 1263 pasos en total.
- Verificaciones sin GPU (celda 9): 6299 ítems, 6271 títulos únicos y 6271 caminos del trie; el título más largo ocupa 69 tokens; 1 título supera los 64 tokens del corte de D3 y ninguno los 128 usados; 0 títulos con "Response:". Largo de las filas (tokens): train media 154,9, p99 280, máximo 365; test media 159,2, p99 268,6, máximo 339.

Las verificaciones equivalentes de CDs no tienen output en este notebook, pero sus cifras aparecen en otros: 5841 ítems (`../caser/baseline_caser.ipynb` y `../gru4rec/baseline_gru4rec.ipynb`, celda 7), 5377 títulos únicos y 95 tokens en el título más largo (`../base_cot/base_cot.ipynb`, celda 9). Sin output guardado: la prueba rápida (ver Costos medidos; incluye 82 de 100 filas con top-10 distinto entre lotes de 24 y de 1) salieron de una corrida fuera del notebook.

Pendiente: CDs, Instruments y Games.

## Dónde quedan los resultados

Las celdas se numeran desde 0 (posición en el JSON del notebook).

- Local, no versionado: `checkpoints/bigrec/<categoría>/` (`entrenamiento.json`, `modelo/`, `predicciones/bigrec/`).
- Versionado: `resultados/bigrec/Toys_and_Games/` con `entrenamiento.json` y `predicciones/bigrec/` y `predicciones/d3/` (cada una con `registro.json` y `final_result.json`).
- Celdas: 6 (tamaños de datos), 8 (prompt), 9 (trie y largos), 15 (SFT), 17 (evaluación), 19 (métricas de Toys), 21 (comparación con el paper).

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| Sin memoria con micro-batch 4 y `adamw_torch` en Toys | Los prompts de Toys llegan a 365 tokens (en CDs, 296); la prueba rápida eligió esa configuración en CDs | Se usa micro-batch 2 con `adamw_torch_fused` y gradient checkpointing (`CONFIG_MEMORIA`). El mensaje exacto del fallo (tamaño y memoria asignada) no tiene output guardado | Ninguno: mismo batch efectivo y algoritmo |
| Avisos de OOM del asignador de HIP durante el SFT y la evaluación | La GPU opera cerca del límite de 16 GB (reservada 15,30 GB) | El asignador reintenta y el proceso termina (avisos de 662700032 y 778043392 bytes en la celda 15; otros en la celda 17) | Ninguno observado |
| Si el equipo tiene GPU integrada, el Trainer usa `DataParallel` y agota la memoria | PyTorch ve dos GPUs | La celda 2 deja visible solo la GPU 0 | Ninguno |
| Aviso "missing keys: `lm_head.weight`" al cargar la mejor época | Qwen2.5-1.5B tiene `tie_word_embeddings: true` | Sin acción | Ninguno |
| Aviso "Sliding Window Attention is enabled but not implemented for `sdpa`" | El modelo tiene `use_sliding_window: false` | Sin acción | Ninguno |
| `code/train.py` de D3 no corre tal cual | Pasa `candidate_file` a `D3Dataset` e importa `code.dataset`, que choca con el módulo `code` de Python | Se reimplementa `entrenar` con los mismos argumentos | Ninguno |
