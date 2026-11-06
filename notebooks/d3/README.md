# D3 (`baseline_d3.ipynb`)

D3 (Bao et al., [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.589/)) es el modelo de BIGRec con otra decodificación: se quita la normalización por largo en el beam search (los "ghost tokens") y, en su versión completa, se agrega un *text-free assistant* (TFA) que mezcla logits de un modelo tradicional. Aquí se evalúa solo la primera pieza, sin reentrenar. Corresponde a la fila "D3" de la Tabla 1 del paper (HR@K y NDCG@K con K = 5 y 10).

**Estado:** corrido en `Toys_and_Games` (celdas 2 a 15). `CDs_and_Vinyl`, `Musical_Instruments` y `Video_Games` no están corridos (`DATASETS = ['toys_data']`).

## Fuente y versión

D3 no tiene código propio en este repo: usa `../bigrec/bigrec_baseline.py` (variante `"d3"`, que solo cambia `length_penalty`) y el modelo entrenado de BIGRec (ver `../bigrec/README.md`).

| Elemento | Valor |
|---|---|
| Repo de origen | [SAI990323/DecodingMatters](https://github.com/SAI990323/DecodingMatters) |
| Commit | `585c8427dd327a33155b5566f32bc22e21b5236c` |
| Archivos que se importan | `code/dataset.py` (`D3Dataset`, prompt y etiquetas), `code/LogitProcesser.py` (`CFEnhancedLogitsProcessor`) |
| sha256 `code/dataset.py` | `2b0ea46ef8cab6f9f97516e050c219a009405a4698c1e5da0ffef8aaf9f8b706` |
| sha256 `code/LogitProcesser.py` | `f0b52d7accfca944641274bd67ad4e3b52ce4cd36224659cd2a94ed5ad73bae6` |
| Verificación | `bb.cargar_d3` compara los sha256 antes de importar y falla si no coinciden (celda 4) |

Clonado, desde la raíz del repo:

```bash
git clone https://github.com/SAI990323/DecodingMatters DecodingMatters
git -C DecodingMatters checkout 585c8427dd327a33155b5566f32bc22e21b5236c
```

## Configuración

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo | el `modelo/` de BIGRec de la categoría; D3 no se reentrena | decisión del proyecto (el paper no especifica si D3 usa un modelo aparte) |
| Checkpoint usado (Toys) | `checkpoint-844`, la época con menor `eval_loss` (0.57483; por época 0.731, 0.5748, 0.6258) | medido, celda 6 |
| TFA (ensemble con modelo tradicional) | no se usa | decisión del proyecto, coherente con el paper, App. B (LatentR3 omite el ensemble con modelos tradicionales) |
| `guidance_scale` | no se pasa, default 1.0: `CFEnhancedLogitsProcessor` no mezcla logits | deducción del proyecto (base: `evaluate2.sh:8` de DecodingMatters@585c842 pasa `--length_penalty 0.0` y no `--guidance_scale`; el default está en `evaluate.py:38` y el retorno sin mezcla en `code/LogitProcesser.py:122-124`) |
| `length_penalty` | 0.0 (BIGRec usa 1.0) | repo de los autores, `evaluate2.sh:8`; que los autores de LatentR3 usaran exactamente 0.0 no está especificado |
| `num_beams` / secuencias devueltas | 10 / 10, decodificación restringida por trie a títulos del catálogo | `bigrec_baseline.py` (`NUM_BEAMS`), igual a BIGRec |
| `max_new_tokens` | 128 | el de BIGRec (`MAX_NEW_TOKENS`); el código de D3 usa 64 (`evaluate.py:115`) y el paper no lo especifica para esta corrida |
| Lote de generación | 24 | medido, `registro.json` (`batch_size`) |
| Inferencia | determinista (sin muestreo) | `bigrec_baseline.py` |
| Semilla | 42, la del SFT de BIGRec; la inferencia no la usa | `baseline_comun.py` (`SEED`) |
| Datos | `repo_latentr3/data`, split `test` de Toys: 6738 filas | medido, celda 8 y `registro.json` |
| Hardware y entorno | AMD Radeon RX 9070 XT (16 GB), ROCm; `torch 2.14.1+rocm7.14`, `transformers 4.51.3`, bf16 | medido, celda 4 |

`transformers 4.51.3` es la versión de los autores. Los autores advierten que otra versión cambia los resultados.

## Adaptaciones respecto de la fuente

| Cambio | Motivo | Efecto esperado en los resultados |
|---|---|---|
| Se evalúa sobre el modelo de BIGRec en vez de uno entrenado para D3 | Mismo procedimiento de entrenamiento; la diferencia BIGRec vs D3 queda como efecto puro de la decodificación | Si los autores reentrenaron D3, sus valores pueden diferir por ruido de entrenamiento |
| Sin TFA | LatentR3 omite el ensemble (paper, App. B) y el `evaluate2.sh` publicado ya corre sin mezclar logits | Ninguno respecto de ese script |
| `max_new_tokens` 128 en vez de 64 | Se usa el de BIGRec para que ambas variantes comparen igual | Solo afecta títulos de más de 64 tokens; no medido para D3 |
| Generación por lotes de 24 con `attention_mask`; código de generación de `bigrec_baseline.py` | Cabe en 16 GB; el código de D3 genera con lote 1 | Posibles diferencias numéricas menores por el padding; no medidas |
| Evaluación por partes de 256 filas, retomable | Corridas de unos 29 min | Ninguno |

## Limitaciones y desviaciones respecto del paper

- Solo `Toys_and_Games`: no hay evidencia propia para las otras tres categorías.
- Una sola corrida determinista, sin intervalos ni varias semillas (el paper reporta una corrida, App. E).
- Entorno distinto del de los autores (ROCm 7.14 en una GPU de 16 GB, con lotes de 24): puede haber diferencias numéricas.
- D3 sin TFA y sobre el modelo de BIGRec: no es D3 completo.
- Nuestros BIGRec y D3 en Toys superan al paper (HR@5 0.0770 y 0.0873 frente a 0.0701 y 0.0830). La causa no está identificada, así que no se debe leer como mejora del método.
- La mejora relativa de D3 sobre BIGRec es menor que la del paper (tabla de Resultados).

## Cómo correrlo

1. Entorno: `.venv-rocm` o `.venv-cuda` (ver README de la raíz), con GPU y bf16.
2. Directorio de trabajo: `notebooks/d3/` (los datos se leen desde `../../repo_latentr3/data`).
3. Clonar `DecodingMatters/` en la raíz del repo, en el commit indicado arriba.
4. Tener el modelo de BIGRec entrenado: `checkpoints/bigrec/<categoría>/modelo/` y `entrenamiento.json` con `terminado: true` (se entrena con `../bigrec/baseline_bigrec.ipynb`).
5. Editar `DATASETS` (celda 3) para elegir qué categorías evaluar; hoy `['toys_data']`.
6. Ejecutar las celdas en orden. La celda 8 evalúa y escribe en `checkpoints/bigrec/<categoría>/predicciones/d3/`; si se corta, se retoma volviendo a correrla. Las celdas 10, 12, 14 y 15 usan todas las categorías que tengan predicciones de BIGRec y D3.
7. En Colab: montar Drive, subir `bigrec_baseline.py`, `baseline_comun.py` y `repo_latentr3/src/` al directorio de trabajo, y clonar `DecodingMatters/`. Los modelos se leen de `MyDrive/IIC3633/checkpoints/bigrec/`.

No toca el modelo ni las predicciones de BIGRec.

## Costos medidos

| Etapa | GPU | Memoria GPU | RAM | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|---|---|
| Entrenamiento | no aplica (se usa el modelo de BIGRec) | no aplica | no aplica | no aplica | no aplica | ver `../bigrec/README.md` |
| Evaluación D3, Toys (6738 filas, lotes de 24) | RX 9070 XT, 16 GB | 9.47 GB (máximo; `memoria_max_gb` 9.466790199279785) | no medida | 1756.8 s (`segundos` 1756.827294588089), unos 0.26 s por fila | medido | `resultados/bigrec/Toys_and_Games/predicciones/d3/registro.json`, celda 8 |
| Evaluación BIGRec, Toys (referencia) | igual | 9.54 GB (9.541008949279785) | no medida | 1724.18 s (1724.1821973323822) | medido | `resultados/bigrec/Toys_and_Games/predicciones/bigrec/registro.json` |
| Evaluación D3 en CDs, Instruments, Games | | | | | no medido | no corridas |

## Resultados

Toys, test completo (6738 filas, 0 títulos fuera del catálogo, 0 repetidos). Medido, celda 10 (verificado contra `calc.py`).

| Modelo | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| BIGRec (nuestro) | 0.0770 | 0.1060 | 0.0563 | 0.0657 |
| D3 (nuestro) | 0.0873 | 0.1153 | 0.0649 | 0.0740 |

La celda 10 también imprime HR@1/3/20 y NDCG@1/3/20; ver el output. HR@20 es igual a HR@10 porque se generan solo 10 títulos.

Mejora relativa de D3 sobre BIGRec en %, celda 15:

| Fuente | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| paper, Tabla 1 | 18.4 | 10.2 | 20.1 | 15.8 |
| nuestra | 13.3 | 8.8 | 15.2 | 12.7 |

Efecto de la decodificación en Toys (celda 12): el modelo y los beams son los mismos, solo cambia el orden final.

| Medida | Valor |
|---|---|
| Top-1 distinto entre BIGRec y D3 | 58.44 % de las filas |
| Top-10 distinto como lista | 99.96 % |
| Top-10 distinto como conjunto | 84.86 % |
| Largo medio del top-1 (caracteres) | 74.18 (BIGRec) y 52.71 (D3) |

D3 elige títulos más cortos, como predice su argumento sobre el sesgo hacia títulos largos.

Paper, Tabla 1, fila D3 (copiada en el dict `PAPER` del código de la celda 14; su output solo muestra Toys), y fila BIGRec como referencia:

| Modelo | Dataset | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| D3 | Toys_and_Games | 0.0830 | 0.1026 | 0.0610 | 0.0674 |
| D3 | CDs_and_Vinyl | 0.1122 | 0.1272 | 0.0906 | 0.0955 |
| D3 | Video_Games | 0.0608 | 0.0860 | 0.0423 | 0.0505 |
| D3 | Musical_Instruments | 0.0984 | 0.1167 | 0.0848 | 0.0907 |
| BIGRec | Toys_and_Games | 0.0701 | 0.0931 | 0.0508 | 0.0582 |

## Dónde quedan los resultados

Las celdas se numeran desde 0.

- Local, no versionado: `checkpoints/bigrec/Toys_and_Games/predicciones/d3/` (partes, `registro.json`, `final_result.json`) y `.../predicciones/bigrec/`.
- Copias versionadas: `../../resultados/bigrec/Toys_and_Games/predicciones/d3/` y `.../predicciones/bigrec/` (`final_result.json` y `registro.json`); modelo y entrenamiento en `../../resultados/bigrec/Toys_and_Games/entrenamiento.json`.
- Celda 6: checkpoint elegido. Celda 8: evaluación (avisos de memoria, tiempo por fila). Celda 10: métricas. Celda 12: efecto de la decodificación. Celda 14: tabla contra el paper. Celda 15: mejora relativa.

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| La celda 8 imprime 3 avisos `HIPCachingAllocator ... memory allocation failed with OOM` (impresos entre las líneas de las partes 4 y 5, 10 y 11, y 25 y 26 de 27) | Memoria libre insuficiente en la GPU de 16 GB en esos lotes (por ejemplo, 490733568 B libres al pedir 1247805440 B) | El allocator se recupera y la evaluación completó las 27 partes | Sin efecto visible: 6738 filas evaluadas y 0 títulos fuera del catálogo; no se comparó contra una corrida sin avisos |
| Si se cambia `transformers` de 4.51.3, la celda 4 avisa | Los autores advierten que otra versión cambia los resultados | Usar 4.51.3 | Resultados distintos |
| Faltan DecodingMatters/ o `repo_latentr3/src/` | Dependencias externas no versionadas | La celda 4 lanza error con la instrucción de clonado | No aplica |
