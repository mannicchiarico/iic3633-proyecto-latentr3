# Base y COT (`base_cot.ipynb`)

Baselines **Base** y **COT** de la Tabla 1 de LatentR3: Qwen2.5-1.5B sin ajustar, que escribe el título del siguiente ítem con decodificación restringida al catálogo. Base usa el prompt directo; COT genera primero un razonamiento zero-shot al estilo Rec-SAVER y después el título. Corresponden a las columnas Base y COT de la Tabla 1 del paper (y a la Tabla 6 en CDs, que agrega COT-RL).

**Estado:** Base corrido en las 4 categorías, con el test completo. COT con tope de 256 tokens corrido solo en Toys (6738 filas, `DATASETS_COT = ['toys_data']`). Pendientes: COT en CDs, Instruments y Games, y la comparación con la Tabla 6 del paper (COT-RL).

## Fuente y versión

- **Paper:** Sección 4.1 ("Base (direct LLM recommendation), COT (Tsai et al., 2024)"; "for explicit reasoning, we only include the direct CoT baseline"), Apéndice B, Sección 4.2, Apéndice D (Tabla 6) y Apéndice G.
  - Base: "directly prompts the backbone LLM to perform the recommendation task" (App. B).
  - COT: "the zero-shot version of Rec-SAVER, which uses CoT reasoning to prompt the LLM for recommendation, i.e., instructing the LLM to produce a reasoning process before generating the final results" (App. B). Backbone: Qwen2.5-1.5B.
  - El paper no publica los prompts ni el tope del razonamiento, no dice cómo se obtiene el top-10 de COT ni si usa decodificación restringida. El repo de los autores no tiene código para Base ni COT.
- **Rec-SAVER:** Tsai et al., *Leveraging LLM reasoning enhances personalized recommender systems*, Findings of ACL 2024 (https://aclanthology.org/2024.findings-acl.780/). Predice el rating (1–5) de un ítem dado con PaLM 2-M zero-shot sobre Amazon Beauty y Movies/TV, no el siguiente ítem. Su prompt (Tabla 12) lleva por ítem del historial título, brand, categories, description, price, rating y reseña, más la metadata del ítem nuevo, y pide `### Reason ###` y después `### Rating ###`. No especifica la decodificación del zero-shot.
- **Código de D3** (`D3Dataset` y `CFEnhancedLogitsProcessor`), reutilizado sin modificar mediante `../bigrec/bigrec_baseline.py` (ver [`../bigrec/README.md`](../bigrec/README.md#cómo-correrlo)). Repo https://github.com/SAI990323/DecodingMatters, commit `585c8427dd327a33155b5566f32bc22e21b5236c`:

  ```bash
  git clone https://github.com/SAI990323/DecodingMatters DecodingMatters
  git -C DecodingMatters checkout 585c8427dd327a33155b5566f32bc22e21b5236c
  sha256sum DecodingMatters/code/dataset.py DecodingMatters/code/LogitProcesser.py
  # 2b0ea46ef8cab6f9f97516e050c219a009405a4698c1e5da0ffef8aaf9f8b706  DecodingMatters/code/dataset.py
  # f0b52d7accfca944641274bd67ad4e3b52ce4cd36224659cd2a94ed5ad73bae6  DecodingMatters/code/LogitProcesser.py
  ```

  El notebook verifica los sha256 antes de importar (`bb.cargar_d3`, celda 4).
- **Métricas:** `repo_latentr3/src/utils/calc.py` (repo de los autores). Cada resultado se verifica corriendo `gao` de `calc.py` sobre el mismo `final_result.json`.
- **Modelo:** `Qwen/Qwen2.5-1.5B` (base, no Instruct), en bf16 y sin entrenamiento: el mismo modelo sirve para los 4 datasets.

## Configuración

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo | `Qwen/Qwen2.5-1.5B`, bf16, sin ajuste | paper, App. B |
| Datos | Mismos CSV `test` y catálogo `info/` de `repo_latentr3/data/` que los demás baselines; no se usa `train` ni `valid` | decisión del proyecto |
| Filas de test | CDs 6157, Instruments 8313, Toys 6738, Games 9397 | medido, celda 9 |
| Prompt de Base | `D3Dataset` sin modificar, con sus erratas ("recetenly", "palyed") | decisión del proyecto |
| Prompt de COT | Prompt de Base más el rating de cada ítem, las frases de razonamiento de la Tabla 12 de Rec-SAVER (adaptadas) y la sección `### Reason: ` (ver "Prompts") | decisión del proyecto |
| Historial | Títulos (Base y COT) y ratings (COT); sin metadata ni reseñas | decisión del proyecto |
| Etapa 1 de COT (razonamiento) | Greedy (beam 1), hasta `###`, EOS o 256 tokens (`MAX_TOKENS_RAZON`), lotes fijos de 64 | beam 1: paper, App. G; tope y lote: decisión del proyecto |
| Top-10 (Base y etapa 2 de COT) | 10 beams restringidos al catálogo con el trie, `length_penalty` 1.0, `max_new_tokens` 128 | igual que BIGRec; repo de los autores, `latent_attention_eval.py:32` (`length_penalty`) |
| Lote del top-10 | 24 en Base, 12 en COT | Base: igual que BIGRec; COT: decisión del proyecto (ver "Adaptaciones") |
| Criterio de acierto | Por título, normalizado como `calc.py` | paper (criterio de la Tabla 1); verificado contra `calc.py` |
| Corridas | Una por configuración, sin muestreo | paper, App. E (determinista) |
| `SAMPLE_TEST` | -1 (test completo) | celda 3 |
| Hardware | AMD Radeon RX 9070 XT (16 GB), ROCm | medido, celda 4 |
| Entorno | `torch 2.14.1+rocm7.14`, `transformers 4.51.3` (la versión de los autores) | medido, celda 4 |

### Prompts

Base es el prompt de D3 sin modificar. En COT, todo lo que no es el rating, la segunda línea de la instrucción ni la sección `### Reason: ` es idéntico al de Base:

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Given a list of {categoría} the user recetenly enjoy, please write a new {categoría} that the user may bought
User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. What information can you infer about the user's preferences? Your reasoning explanation should be based on any commonalities in the user history items and inferred user tastes or preferences. After your reasoning, write the title of the new item.
### User Input: 
The user has palyed the following {categoría}s before: "A" (rating: 4), "B" (rating: 5)

### Reason: 
```

- **Etapa 1:** el modelo continúa desde `### Reason: \n` con greedy.
- **Etapa 2:** prompt, más el razonamiento recortado antes del primer `###`, más `\n\n### Response: \n`, y desde ahí el título restringido al catálogo. La sección de respuesta se agrega como tokens aparte, así que los últimos 4 tokens son siempre la raíz del trie de BIGRec y se reutilizan sin cambios el trie y el procesador de logits de D3.

## Adaptaciones respecto de la fuente

Respecto de Rec-SAVER (Tabla 12):

| Cambio | Motivo | Efecto esperado en los resultados |
|---|---|---|
| Se quita la metadata del ítem nuevo | En recomendación del siguiente ítem sería el objetivo: fuga | Evita una fuga que inflaría COT |
| La pregunta pasa de "how they will rate the new product?" a "write the title of the new item" | Otra tarea | Razonamiento orientado al título, no al rating |
| Salida `### Reason: ` y `### Response: ` con el título entre comillas, en vez de `### Reason ###` y `### Rating ###` | Estilo de encabezados de D3; `### Response: \n` es la raíz del trie | Permite reutilizar el trie de BIGRec |
| Historial con títulos entre comillas y `(rating: N)`, en vez de un bloque con título, brand, categories, description, price, rating y reseña por ítem | Alineado con D3 y con lo que ven los demás LLM del proyecto | Menos información que Rec-SAVER; su propia ablación indica que sin reseñas el razonamiento casi no ayuda |
| Preámbulo de D3 en vez del de Rec-SAVER | Alineado con el repo de los autores | Sin efecto medido |
| El prompt termina en `### Reason: \n` | Un modelo base completa texto; no sigue instrucciones de formato | Sin efecto medido |
| Lote del top-10 de COT de 12, no de 24 | Los prompts de la etapa 2 miden unos 410 tokens con el razonamiento (celda 12); en la prueba rápida la celda 11 registra avisos de OOM del asignador con lotes de 24. La configuración probada con lote 24 no está guardada en el output | El top-10 depende del tamaño del lote por el padding en bf16 (ver "Limitaciones"); el lote queda fijo y se anota en `registro.json` |
| Se descarta el catálogo de ratings de Rec-SAVER | Es otra tarea y otro modelo | Ninguno sobre la tarea de este proyecto |
| Tope del razonamiento de 256 tokens | Decisión del proyecto, revisada con la prueba rápida (celda 12): con 512 se cortaba el 18 % y con 256 el 23 % | Un razonamiento cortado puede cambiar el título; ver V6 |

## Limitaciones y desviaciones respecto del paper

- **Prompts propios:** los de Base y COT son del proyecto, no los de los autores. La comparación con la Tabla 1 mide qué tan cerca queda el resultado, no si es el mismo método.
- **Rec-SAVER es otra tarea:** predice ratings con PaLM 2-M y reseñas; aquí se usa un modelo de 1,5B sin reseñas ni metadata.
- **Modelo base sin instrucciones:** puede no seguir el formato, repetirse o copiar títulos del historial en el razonamiento. Se mide cómo terminan los razonamientos (ver "Resultados"). No se usa penalización por repetición, que el paper no especifica.
- **Tope de 256 tokens:** en Toys, el 24,2 % de los razonamientos de la corrida completa termina en el tope (celda 19).
- **El top-10 depende del tamaño del lote**, por el padding en bf16. En BIGRec cambiaron 82 de 100 filas entre lotes de 24 y de 1 (ver [`../bigrec/README.md`](../bigrec/README.md)). Por eso los lotes quedan fijos.
- **Recompras:** el objetivo entre comillas aparece en el prompt en el 3,1 % (CDs), 0,9 % (Instruments), 1,5 % (Toys) y 2,1 % (Games) de las filas de test (celda 9). Es parte de los datos y también afecta a los demás baselines.
- **Medición de tiempo distinta a la del paper:** el paper mide con 100 muestras, 1 A100 y lotes de 4 (App. G); aquí se usan lotes de 24 (Base), 64 (razonamiento) y 12 (título) en una RX 9070 XT.
- **COT incompleto:** solo Toys tiene COT. Los números de COT en las otras categorías no existen todavía.
- **Entorno distinto:** ROCm y torch 2.14 en vez de A100 y torch 2.6.

## Cómo correrlo

- **Entorno:** `.venv-rocm` o `.venv-cuda` (ver el README de la raíz). Requiere GPU con bf16.
- **Directorio de trabajo:** `notebooks/base_cot/`. En local el notebook trabaja sin conexión (`HF_HUB_OFFLINE=1`) y deja visible solo la GPU 0.
- **Fuentes externas:** `DecodingMatters/` en el commit `585c842` en la raíz del repositorio (ver "Fuente y versión"), `Qwen/Qwen2.5-1.5B` en la caché de Hugging Face y `repo_latentr3/src/utils/calc.py`. `base_cot_baseline.py` agrega a `sys.path` `notebooks/` (por `baseline_comun.py`) y `notebooks/bigrec/` (por `bigrec_baseline.py`).
- **Variables que se editan (celda 3):**
  - `DATASETS_BASE` (por defecto los 4) y `DATASETS_COT` (por defecto `['toys_data']`) son independientes. Agregar un dataset después no repite los ya evaluados.
  - `MAX_TOKENS_RAZON` (256) y `SAMPLE_TEST` (-1 = test completo). El nombre de la carpeta de salida lleva ambos (`base`, `cot_256`, `cot_256_n1000`...), así que cambiarlos no mezcla corridas.
- **Orden:** ejecutar las celdas en orden. Las celdas 6 a 9 (conteos, ejemplos de prompts y verificaciones V1 a V4) no usan GPU. Las celdas 10 a 15 son la prueba rápida (100 filas de Toys; escribe en `checkpoints/base_cot/_prueba/` y lo borra al final). La celda 17 es la evaluación completa y la 19 calcula las métricas.
- **Retomar:** volver a correr la celda de evaluación, que se salta las partes ya escritas. Si una carpeta tiene partes de otra configuración, la evaluación falla en vez de mezclarlas.
- **Archivos que escribe** (excepción a la convención de que los notebooks no escriben archivos). En `checkpoints/base_cot/<categoría>/predicciones/<carpeta>/`, ignorado por git: partes de 256 filas, `registro.json` y `final_result.json` con el formato de `calc.py`. En COT, cada fila guarda además `reasoning`, `tokens_generados`, `fin` (`marca`, `eos` o `tope`) y `tokens_etapa2`.

### Verificaciones

| Código | Qué comprueba | Resultado |
|---|---|---|
| V1 | Trie de Base: todos los prompts terminan en la raíz y los títulos generables son exactamente los del catálogo | Correcta en los 4 datasets (celda 9) |
| V2 | Alineación: sin ratings ni razonamiento, COT da los mismos tokens que `D3Dataset` en 200 filas de cada split | Correcta en los 4 datasets (celda 9) |
| V3 | Etapa 2 termina en la raíz del trie con razonamientos de distintas formas | Correcta en los 4 datasets (celda 9) |
| V4 | Fuga: el objetivo entre comillas no aparece en ningún prompt salvo las recompras | Correcta; recompras 1,5 % (Toys), 3,1 % (CDs), 2,1 % (Games), 0,9 % (Instruments) (celda 9) |
| V5 | Las métricas coinciden con `calc.py` y no hay títulos fuera del catálogo | Correcta: 0 títulos fuera del catálogo en la prueba rápida (celda 11) y en la corrida completa (celda 19) |
| V6 | El razonamiento con tope 256 es el prefijo del de 512 (solo en la prueba rápida) | 77 idénticos y 23 cortados en 256 que son prefijo del de 512 (celda 12) |

## Costos medidos

GPU: AMD Radeon RX 9070 XT (16 GB). RAM: no medida. Los tiempos por fila de cada corrida completa salen de `registro.json` (celda 19).

| Etapa | Memoria GPU | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|
| Prueba rápida, Base (100 filas de Toys) | 8,63 GB | 0,309 s por fila | medido | celda 11 |
| Prueba rápida, COT-256 | 10,02 GB | 0,750 s por fila (0,2045 razonamiento + 0,545 título) | medido | celda 11 |
| Prueba rápida, COT-512 | 12,8 GB | 1,617 s por fila (0,440 razonamiento + 1,177 título) | medido | celda 11 |
| Base, CDs (6157 filas) | 8,54 GB | 0,306 h (0,179 s por fila) | medido | celda 19, `resultados/base_cot/CDs_and_Vinyl/predicciones/base/registro.json` |
| Base, Instruments (8313 filas) | 11,22 GB | 0,790 h (0,342 s por fila) | medido | celda 19, `resultados/base_cot/Musical_Instruments/predicciones/base/registro.json` |
| Base, Toys (6738 filas) | 9,47 GB | 0,486 h (0,260 s por fila) | medido | celda 19, `resultados/base_cot/Toys_and_Games/predicciones/base/registro.json` |
| Base, Games (9397 filas) | 9,32 GB | 0,579 h (0,222 s por fila) | medido | celda 19, `resultados/base_cot/Video_Games/predicciones/base/registro.json` |
| COT-256, Toys (6738 filas) | 10,44 GB | 1,322 h = 4757 s (1292 de razonamiento y 3465 de título; 0,706 s por fila) | medido | celdas 17 y 19, `resultados/base_cot/Toys_and_Games/predicciones/cot_256/registro.json` |
| COT-256, CDs / Instruments / Games | no medida | 1,28 / 1,73 / 1,96 h | estimado (cálculo: s por fila de la prueba rápida por filas de test) | celda 14 |

- **COT frente a Base:** 2,4× en la prueba rápida (celda 14); el paper reporta ~25× en Toys y ~30× en Games, con lotes de 4 (Sec. 4.2 y App. G). En la corrida completa de Toys, 0,706 s frente a 0,260 s por fila.
- **Referencia:** BIGRec en Toys tardó 0,256 s por fila (1724 s para 6738 filas, lotes de 24) (celda 14).
- **Estimación previa a la corrida (celda 14):** Base 0,53 / 0,71 / 0,58 / 0,81 h (CDs / Instruments / Toys / Games), 2,63 h en total; COT-256 1,28 / 1,73 / 1,40 / 1,96 h, 6,37 h en total. La corrida completa de Toys midió 1,322 h frente a 1,40 h estimadas.

## Resultados

### Resultados propios (test completo, medido, celda 19)

HR@K y NDCG@K por título.

| Categoría | Método | Filas | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| CDs | Base | 6157 | 0.0102 | 0.0185 | 0.0258 | 0.0372 | 0.0102 | 0.0148 | 0.0178 | 0.0216 |
| Instruments | Base | 8313 | 0.0023 | 0.0207 | 0.0396 | 0.0635 | 0.0023 | 0.0124 | 0.0200 | 0.0279 |
| Toys | Base | 6738 | 0.0080 | 0.0217 | 0.0380 | 0.0650 | 0.0080 | 0.0155 | 0.0223 | 0.0310 |
| Games | Base | 9397 | 0.0161 | 0.0220 | 0.0278 | 0.0399 | 0.0161 | 0.0195 | 0.0218 | 0.0257 |
| Toys | COT (tope 256) | 6738 | 0.0085 | 0.0203 | 0.0373 | 0.0644 | 0.0085 | 0.0151 | 0.0221 | 0.0308 |

HR@20 y NDCG@20 son iguales a HR@10 y NDCG@10 en todas las filas, porque el top-10 solo genera 10 títulos. Títulos fuera del catálogo y repetidos: 0 en todas las filas. No hay COT medido en CDs, Instruments ni Games.

### Estadísticas del razonamiento de COT

| Corrida | Termina en `###` | Termina en EOS | Termina en el tope | Vacíos | Tokens generados (media / mediana / p90) | Fuente |
|---|---|---|---|---|---|---|
| Prueba rápida, tope 512 (100 filas) | 70 % | 12 % | 18 % | 0 | 203,46 / 128,5 / 512 | medido, celda 12 |
| Prueba rápida, tope 256 (100 filas) | 67 % | 10 % | 23 % | 0 | 150,19 / 128,5 / 256 | medido, celda 12 |
| Corrida completa, Toys, tope 256 (6738 filas) | 67,3 % | 8,5 % | 24,2 % | 0 | 145,986 / 124 / 256 | medido, celda 19 |

Tokens de la etapa 2 (media): 410,58 con tope 256 y 463,77 con tope 512 en la prueba rápida (celda 12); 407,877 en la corrida completa de Toys (celda 19).

### Tabla 1 del paper y comparación (celda 21)

Valores del paper copiados de la Tabla 1; la fila de Tabla 6 es COT-RL en CDs.

| Categoría | Fuente | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| CDs | paper, Tabla 1: Base | 0.0195 | 0.0252 | 0.0148 | 0.0167 |
| CDs | nuestro Base | 0.0258 | 0.0372 | 0.0178 | 0.0216 |
| CDs | paper, Tabla 1: COT | 0.0302 | 0.0406 | 0.0213 | 0.0246 |
| CDs | paper, Tabla 6: COT-RL | 0.0297 | 0.0390 | 0.0202 | 0.0232 |
| Instruments | paper, Tabla 1: Base | 0.0154 | 0.0192 | 0.0296 | 0.0411 |
| Instruments | nuestro Base | 0.0396 | 0.0635 | 0.0200 | 0.0279 |
| Instruments | paper, Tabla 1: COT | 0.0135 | 0.0199 | 0.0261 | 0.0452 |
| Toys | paper, Tabla 1: Base | 0.0203 | 0.0359 | 0.0128 | 0.0178 |
| Toys | nuestro Base | 0.0380 | 0.0650 | 0.0223 | 0.0310 |
| Toys | paper, Tabla 1: COT | 0.0261 | 0.0496 | 0.0153 | 0.0229 |
| Toys | nuestro COT | 0.0373 | 0.0644 | 0.0221 | 0.0308 |
| Toys | nuestro BIGRec (referencia) | 0.0770 | 0.1060 | 0.0563 | 0.0657 |
| Games | paper, Tabla 1: Base | 0.0236 | 0.0311 | 0.0190 | 0.0214 |
| Games | nuestro Base | 0.0278 | 0.0399 | 0.0218 | 0.0257 |
| Games | paper, Tabla 1: COT | 0.0120 | 0.0194 | 0.0082 | 0.0105 |

- **Anomalía de Instruments en el paper:** las filas Base y COT de Instruments tienen NDCG > HR (NDCG@10 de 0.0411 y 0.0452 frente a HR@10 de 0.0192 y 0.0199), imposible con un solo ítem relevante. No sirven como referencia.
- **Nuestro Base supera al del paper** en las 3 categorías con referencia válida (CDs, Toys, Games). El paper reporta Base muy por debajo de los modelos tradicionales; nuestro Base queda en HR@10 de 0.0372 a 0.0650.
- **COT en Toys:** nuestro COT (HR@10 0.0644) queda prácticamente igual que nuestro Base (0.0650), mientras que el COT del paper (0.0496) mejora sobre su Base (0.0359). Nuestro COT supera al COT del paper en Toys.
- **Pendiente:** COT en CDs, Instruments y Games, y la comparación con la Tabla 6 (COT-RL) en CDs, que requiere el COT de CDs.

### Cifras sin output guardado

- "~35 % de ítems ya vistos en el top-10 (34 % en Base)" de la prueba rápida: medido fuera del notebook, sin output guardado.
- El ejemplo de lote de 24 que no cupo en 16 GB en la etapa 2 de COT: medido fuera del notebook, sin output guardado (la celda 11 solo muestra los avisos de OOM del asignador, sin lotes reducidos).

## Dónde quedan los resultados

Las celdas se numeran desde 0 (posición en el JSON del notebook).

- **Local, no versionado:** `checkpoints/base_cot/<categoría>/predicciones/<base|cot_256>/` con las partes, `registro.json` y `final_result.json`.
- **Copias versionadas:** `resultados/base_cot/<categoría>/predicciones/<base|cot_256>/` (`registro.json` y `final_result.json`; los `final_result.json` incluyen en COT el razonamiento por fila). Existen `base` en las 4 categorías y `cot_256` solo en `Toys_and_Games`.
- **Celdas que respaldan cada cifra:**

| Celdas | Respaldan |
|---|---|
| 9 | Conteos de filas y verificaciones V1 a V4 (incluye recompras y tokens por prompt) |
| 11 | Prueba rápida: tiempos por fila, memoria y V5 |
| 12 | Estadísticas del razonamiento y V6 |
| 14 | Horas estimadas y referencia de BIGRec en Toys |
| 17 | Evaluación completa (log por parte) |
| 19 | Métricas, horas, memoria y estadísticas del razonamiento de la corrida completa |
| 21 | Comparación con la Tabla 1 y la Tabla 6 del paper y con BIGRec |

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| Avisos `HIPCachingAllocator ... memory allocation failed with OOM` en las celdas 11 y 17 | El pico de memoria se acerca a los 16 GB (total del dispositivo 17095983104 bytes) | No hubo lotes reducidos (`lotes reducidos` = 0 en las celdas 11 y 19) | Ninguno medido |
| Top-10 sensible al tamaño del lote | Padding en bf16 | Lotes fijos (24, 64 y 12) anotados en `registro.json` | Cambiar el lote puede cambiar el top-10 |
| NDCG > HR en el paper (Instruments, Base y COT) | Imposible con un solo ítem relevante; error del paper | No se usa como referencia | La comparación en Instruments no es válida |
| COT sin correr en CDs, Instruments y Games | Costo estimado de 1,28 / 1,73 / 1,96 h (celda 14) | Pendiente: agregar los datasets a `DATASETS_COT` | No hay comparación de COT con la Tabla 1 en esas categorías |
| 24,2 % de los razonamientos de Toys cortados en el tope de 256 tokens | Modelo base sin instrucciones que no cierra el razonamiento | Decisión del proyecto de dejar el tope en 256 | Posible subestimación de COT respecto de un tope mayor |
