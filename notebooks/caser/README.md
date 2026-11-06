# Caser (`baseline_caser.ipynb`)

Baseline Caser (Tang y Wang, 2018) del paper de LatentR3. Mide HR y NDCG @5 y @10 con todo el catálogo como candidatos y se compara con la fila Caser de la Tabla 1 del paper. Además de la variante del paper (BCE (paper)), el notebook corre tres variantes de referencia.

**Estado:** corrido completo (4 variantes × 4 datasets × 9 combinaciones de lr y weight decay, 144 corridas, `SEED = 42`, RX 9070 XT, 6 procesos, `CONV_MATMUL = True`). Los outputs están guardados en el notebook. El protocolo común (datos, métricas, grilla, early stopping) y las decisiones comunes están en el [README de `notebooks/`](../README.md#decisiones-comunes-d1d17); aquí van solo las partes propias de Caser.

## Fuente y versión

- Repo de origen: [graytowne/caser_pytorch](https://github.com/graytowne/caser_pytorch), implementación en PyTorch del primer autor del paper de Caser. Commit `c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1` (el último del repositorio). Licencia LGPL-3.0: el código se usa sin modificarlo ni redistribuirlo.
- Del repo solo se usa la clase `Caser` (`caser.py`) con su `forward`, y `activation_getter` de `utils.py`, que `caser.py` importa. El resto está en `caser_baseline.py`.
- El [código original en MATLAB](https://github.com/graytowne/caser) sirve solo de referencia. Su README recomienda la versión en PyTorch, y sus valores por defecto coinciden con los de PyTorch.
- sha256 de los archivos usados (el notebook los verifica antes de importar el modelo):

```bash
git clone https://github.com/graytowne/caser_pytorch caser_pytorch   # en la raíz del repositorio
git -C caser_pytorch checkout c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1
sha256sum caser_pytorch/caser.py caser_pytorch/utils.py
# 63dd99f91021d99093d33cc1c71e182acdca4c8d3c366e00369981a297a22d31  caser_pytorch/caser.py
# 8f63d96e9e3449a71543e382157c594104999df0ee36f484e2c4416ca2ab3ab1  caser_pytorch/utils.py
```

La carpeta `caser_pytorch/` está en `.gitignore`. `utils.py` se registra como módulo `utils` solo mientras se ejecuta `caser.py`, sin tocar `sys.path`; ambos archivos importan solo `torch`, `numpy` y `random`.

## Configuración

| Parámetro | Valor | Fuente |
|---|---|---|
| Optimizador | Adam, betas por defecto, L2 como `weight_decay` | paper, App. B; repo de los autores, `train_caser.py:83-85` |
| Grilla | lr {1e-2, 1e-3, 1e-4} × weight decay {1e-4, 1e-5, 1e-6} = 9 combinaciones por variante y dataset | paper, App. B |
| Elección de la configuración | NDCG@10 por id en `valid` | decisión del proyecto (ver README común) |
| Épocas | hasta 200, early stopping con paciencia 10 sobre NDCG@10 por id en `valid`; se restaura la mejor época | decisión del proyecto (el paper no lo especifica) |
| Semilla | 42, una sola corrida | paper, App. E (una corrida) |
| Largo de secuencia L | 10, padding a la izquierda | paper, App. A |
| Objetivos por fila T | 1 (3 en la variante BCE T=3) | decisión del proyecto |
| Dimensión, filtros, dropout, batch | d = 50, 4 filtros verticales, 16 horizontales por altura, ReLU, dropout 0.5, batch 512 | repo de los autores, `train_caser.py:291, 301-306` (el paper no los especifica) |
| Datos | una muestra por fila de `train/`: historial (≤10 ítems) → `item_id`; mismas filas que usan los LLM | decisión del proyecto |
| Negativos (BCE) | 1 por fila, uniforme sobre el catálogo (puede coincidir con el positivo) | decisión del proyecto (convención común con GRU4Rec y SASRec) |
| Usuarios | embedding para los usuarios con filas en `train`; índice 0 reservado, en cero, para los demás | decisión del proyecto |
| Candidatos | todo el catálogo, sin el padding; los ítems del historial no se excluyen | decisión del proyecto (ver README común) |
| Procesos | 6 (`N_PROCESOS = N_PROCESOS_CE = 6`), 1 hilo de CPU por proceso, `spawn` | decisión del proyecto |
| Hardware | AMD Ryzen 5 7600X y AMD Radeon RX 9070 XT (16 GB) | equipo del proyecto; el notebook imprime la GPU (celda 3) |
| Entorno | `.venv-rocm` (ver README raíz) | README raíz |

Variantes (nombres iguales a los del notebook y de las tablas):

- **BCE (paper):** la variante del paper. BCE del objetivo de la fila contra un negativo uniforme, con embedding de usuario. Es la única comparable con la Tabla 1.
- **BCE sin usuario:** la misma BCE con el vector de usuario fijo en cero en todas las filas, así que el puntaje depende solo de los ≤10 ítems del historial (la misma información que GRU4Rec, SASRec y los LLM). Mide cuánto aporta el embedding de usuario y, junto con las métricas por usuarios conocidos y nuevos, si el vector cero perjudica a los usuarios nuevos.
- **CE catálogo:** entropía cruzada (softmax) sobre todo el catálogo, sin negativos muestreados, con la misma grilla. Mide cuánto le cuesta a Caser la BCE con un solo negativo.
- **BCE T=3:** mismas filas de `train`, pero cada fila supervisa su objetivo y los 2 ítems siguientes del mismo usuario que estén en `train`, contra 3 negativos uniformes. Replica la supervisión por defecto del original (T=3 y 3 negativos, con su pérdida) sobre las filas del proyecto; no es el original completo (no usa sus ventanas ni excluye los ítems del usuario de los negativos). Los objetivos que faltan (las últimas 1 o 2 filas de cada usuario) se excluyen de la pérdida. No hay fuga de datos: los objetivos extra son de filas posteriores del mismo usuario que también están en `train`.

## Adaptaciones respecto de la fuente

Las dos últimas filas son cambios de cálculo, no de modelo.

| Cambio | Motivo | Efecto esperado en los resultados |
|---|---|---|
| Una muestra por fila de `train`, en vez de ventanas deslizantes de L+T ítems por usuario (`interactions.py:148, 209-236`) | Mismas muestras que los LLM. Con L=10 y T=1 las ventanas del original son un subconjunto de las filas: usarían solo el 33,6 % (CDs), 27,1 % (Instruments), 23,9 % (Toys) y 28,5 % (Games) de ellas (sin output guardado) | Más muestras de entrenamiento y más historiales con padding que en el original |
| L = 10 y T = 1, en vez de L = 5 y T = 3 (`train_caser.py:286-287`; `main_caser.m:7-8`) | L = 10 lo fija el paper (App. A); T = 1 sigue de usar una fila por muestra. La variante BCE T=3 conserva T = 3 | Menos señal por muestra que el original |
| Un negativo uniforme del catálogo por fila, en vez de 3 por secuencia sin los ítems del usuario (`train_caser.py:130, 203-234`) | Convención común de los baselines; BCE T=3 usa 3 | Pérdida con menos negativos |
| Pérdida con `binary_cross_entropy_with_logits`, en vez de la media de `log(sigmoid)` positivos y negativos (`train_caser.py:166-170`) | Es la misma función, numéricamente estable | Ninguno |
| Grilla de lr y weight decay y early stopping en `valid`, en vez de lr 1e-3, l2 1e-6, 50 épocas fijas que evalúan en test cada 10 y no eligen época (`train_caser.py:83-85, 119-201, 292-293`) | Lo pide el paper (grilla) y el protocolo común (early stopping) | Distinto al original |
| Evaluación por fila sobre todo el catálogo, HR y NDCG @5 y @10 por id y por título, en vez de precisión, recall y MAP por usuario sin los ítems ya vistos (`evaluation.py:67-89`) | Métricas y candidatos del paper | Las métricas del original no se pueden comparar |
| Se puntúa el catálogo sin el índice 0 (padding); el original también puntúa el 0 (`train_caser.py:263`) | El padding no es un ítem; así la columna j es el id j | Ninguno |
| Ids de `process.py` (0..n-1) más 1 dentro del modelo; el 0 es el padding con embedding entrenable, como en el original (`interactions.py:130-134`; `caser.py:40`) | Mantener los ids del catálogo y el padding del original | Ninguno |
| Usuarios sin filas en `train`: índice 0 puesto en cero al crear el modelo. El original no admite usuarios nuevos | El 43,1–51,5 % de las filas de test son de usuarios sin filas en `train` (celda 7). El índice 0 nunca aparece en `train`, así que su gradiente es cero y sigue en cero | Los usuarios nuevos se puntúan sin información de usuario |
| Puntaje para unos pocos ítems con `.view`, en vez del `squeeze` de `caser.py:125` | El `squeeze` borra una dimensión cuando el lote tiene una sola fila | Ninguno |
| Puntaje de todo el catálogo (evaluación y CE catálogo) como `b2 + x·W2ᵀ` en una multiplicación, con x = [z, usuario] tomado de `forward` con hooks en `fc1` y en el embedding de usuario, en bloques de 1024 filas, en vez de `forward` por usuario con `for_pred=True` (`caser.py:117-125`; `train_caser.py:236-278`) | `forward` copia W2 por fila (128 × n_ítems × 100 por bloque, ~2,4 GB por lote en CE catálogo) | Mismos puntajes (Δ = 0 con el mismo modelo, sin output guardado) |
| Convoluciones como multiplicación de matrices: se reemplaza el `forward` de `conv_h` (ventanas de i filas con `unfold` por el filtro aplanado) y de `conv_v` (`einsum` sobre las L filas) en cada modelo creado, en vez de `nn.Conv2d` (`caser.py:43-47`). `Caser.forward`, los parámetros y el `state_dict` son los del original | El backward de las convoluciones de MIOpen ocupaba el 94 % del tiempo de GPU (profiler, sin output guardado) | Mismo resultado salvo redondeo. La prueba rápida (celda 11) verifica en cada variante puntajes con `atol` 1e-5 y gradientes con error relativo ≤ 1e-4 frente a `nn.Conv2d`; el Δ máximo de 6e-6 en puntajes y 3,3e-6 relativo en gradientes no tiene output guardado. `CONV_MATMUL = False` vuelve a `nn.Conv2d` |

## Limitaciones y desviaciones respecto del paper

- **Protocolo de los autores desconocido:** el paper no dice cómo entrenaron Caser ni si usaron el embedding de usuario. Una diferencia con la Tabla 1 puede deberse al protocolo y no a la implementación.
- **Padding en la mayoría de las filas:** con L = 10, entre el 78 % (CDs) y el 93,5 % (Toys) de las filas de `train` tiene menos de 10 ítems de historial (sin output guardado). Los filtros ven sobre todo posiciones de padding; en el original el padding solo aparece en la ventana única de los usuarios cortos.
- **Usuarios nuevos con vector cero:** en la variante principal, el 43,1–51,5 % de las filas de test usa un vector de usuario que no aparece en el entrenamiento (celda 7). Son el 25–31 % de los usuarios de test (sin output guardado). La variante BCE sin usuario y las métricas por conocidos y nuevos miden su efecto.
- **Menos señal que el Caser original:** cada fila supervisa un solo objetivo. BCE T=3 mide ese efecto.
- **Una sola semilla:** con diferencias chicas entre configuraciones, la elección de la grilla puede cambiar con otra semilla. 10 de las 16 configuraciones ganadoras usan lr 1e-2 (celda 23), y con ese lr dos corridas pueden dar pesos distintos (ver Problemas conocidos).
- **Cálculo distinto del original:** convoluciones y puntajes con multiplicaciones de matrices. Frente a `nn.Conv2d`, en combinaciones cortas de CDs las métricas difirieron hasta 0,010 con lr 1e-2, 0,004 con lr 1e-3 y 0,0002 con lr 1e-4 y en CE (medido fuera del notebook, sin output guardado). Es el mismo orden que la variación entre corridas con lr 1e-2.
- **Ítems ausentes de `train`:** ver las limitaciones comunes en el [README de `notebooks/`](../README.md).

## Cómo correrlo

1. Activar el entorno (`.venv-rocm` o `.venv-cuda`, ver README raíz) y abrir el notebook con `notebooks/caser/` como directorio de trabajo. Los datos se leen de `../../repo_latentr3/data`.
2. Clonar `caser_pytorch/` en la raíz del repositorio con los comandos de "Fuente y versión". En Colab, la carpeta va en `caser_pytorch/` dentro del directorio de trabajo, y hay que subir `caser_baseline.py` y `../baseline_comun.py` a ese directorio.
3. `caser_baseline.py` va junto al notebook y `baseline_comun.py` en `notebooks/`; el notebook agrega esa carpeta a `sys.path`.
4. Ejecutar las celdas en orden: importaciones y datos (celdas 2 a 7), prueba rápida (celda 11), y una celda por variante (15, 17, 19 y 21, 36 combinaciones cada una). Las tablas usan solo las variantes que se hayan corrido; para comparar con la Tabla 1 basta con BCE (paper). Después van las celdas 23 (mejor configuración), 25 (comparación con el paper), 27 (conocidos y nuevos) y 29 (pisos MostPop).
5. `N_PROCESOS` y `N_PROCESOS_CE` (celda 3) fijan los procesos de la grilla; `CONV_MATMUL` en `caser_baseline.py` elige el cálculo de las convoluciones.

La prueba rápida (celda 11) entrena 3 épocas en CDs con cada variante, después de una época de calentamiento que no se mide. Verifica que `puntajes` dé lo mismo que `forward` para todo el catálogo y que las convoluciones como multiplicación de matrices den los mismos puntajes y gradientes que `nn.Conv2d`. Imprime el tiempo por época, la memoria de GPU y el peor caso de la grilla con un proceso.

La grilla imprime una línea por combinación. `corridas` guarda un DataFrame por variante. El notebook no escribe archivos.

## Costos medidos

Equipo: Ryzen 5 7600X y RX 9070 XT. RAM: no medida.

| Etapa | GPU | Memoria GPU | RAM | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|---|---|
| Prueba rápida, BCE (paper), 1 proceso | RX 9070 XT | 171,55 MiB | no medida | 0,60 s por época | medido | celda 11 |
| Prueba rápida, BCE sin usuario | RX 9070 XT | 185,29 MiB | no medida | 0,49 s por época | medido | celda 11 |
| Prueba rápida, CE catálogo | RX 9070 XT | 203,56 MiB | no medida | 0,52 s por época | medido | celda 11 |
| Prueba rápida, BCE T=3 | RX 9070 XT | 206,17 MiB | no medida | 0,53 s por época | medido | celda 11 |
| Peor caso de la grilla con 1 proceso (todas las combinaciones a 200 épocas), BCE (paper) / sin usuario / CE catálogo / T=3 | RX 9070 XT | — | no medida | 1,49 / 1,21 / 1,28 / 1,33 h | estimado (cálculo: s por época de CDs × 200 épocas × 9 combinaciones × 4,97, que es la razón entre las filas de `train` de los 4 datasets y las de CDs: 244.824 frente a 49.251, celda 7) | celda 11 |
| Grilla BCE (paper), 36 combinaciones, 6 procesos | RX 9070 XT | no medida en total | no medida | 13,1 min | medido | celda 15 |
| Grilla BCE sin usuario, 6 procesos | RX 9070 XT | no medida en total | no medida | 5,2 min | medido | celda 17 |
| Grilla CE catálogo, 6 procesos | RX 9070 XT | no medida en total | no medida | 7,3 min | medido | celda 19 |
| Grilla BCE T=3, 6 procesos | RX 9070 XT | no medida en total | no medida | 5,9 min | medido | celda 21 |
| Total de las 4 grillas | RX 9070 XT | no medida | no medida | 31,5 min | medido (suma de las celdas 15 a 21) | celdas 15, 17, 19 y 21 |
| Antes de las adaptaciones de cálculo (`nn.Conv2d` y `forward` para todo el catálogo), BCE, CDs, 1 proceso | RX 9070 XT | 423–2436 MiB (las 4 variantes) | no medida | 2,36 s por época | medido fuera del notebook, sin output guardado | — |
| Paralelismo, BCE en CDs: 1 / 3 / 6 procesos | RX 9070 XT | 1,9 / 3,1 / 4,7 GB de GPU total del equipo | no medida | 1,64 / 3,67 / 4,44 épocas por segundo en total (aceleración 2,24× con 3 y 2,71× con 6) | medido fuera del notebook, sin output guardado | — |
| Paralelismo, CE catálogo en Toys (catálogo más grande): 1 / 4 procesos | RX 9070 XT | 2,0 / 3,7 GB de GPU total del equipo | no medida | 1,54 / 3,79 épocas por segundo en total (2,47×) | medido fuera del notebook, sin output guardado | — |

Lo que cambió: con las convoluciones y los puntajes como multiplicación de matrices, una época de CDs pasó de 2,36 s a 0,60 s (BCE), y 6 procesos dan una aceleración de 2,71× sobre 1. Por eso la grilla completa tomó 31,5 minutos. Ninguna combinación llegó a 200 épocas (celdas 15 a 21).

## Resultados

Las cifras salen de los outputs del notebook, que los redondea a 4 decimales. Mejor configuración por variante y dataset, elegida solo con el NDCG@10 por id en `valid` (celda 23; el weight decay sale de las líneas que imprimen las celdas 15, 17, 19 y 21, buscando la combinación con el mismo lr, época y NDCG@10 en `valid`, porque la tabla de la celda 23 lo redondea a `0.0000` cuando es 1e-5 o 1e-6). Las 8 métricas son de `test`. `—` en NDCG@5 id: la tabla de la celda 23 tiene 21 columnas y pandas ocultó esa; para BCE (paper) el valor sale de la celda 25 y para las demás no está en el output.

| Variante | Dataset | lr | wd | Época | Épocas corridas | NDCG@10 valid | HR@5 id | NDCG@5 id | HR@10 id | NDCG@10 id | HR@5 título | NDCG@5 título | HR@10 título | NDCG@10 título |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BCE (paper) | cd_data | 1e-3 | 1e-5 | 163 | 173 | 0.0260 | 0.0297 | 0.0230 | 0.0414 | 0.0268 | 0.0328 | 0.0250 | 0.0465 | 0.0294 |
| BCE (paper) | music_data | 1e-2 | 1e-5 | 32 | 42 | 0.0795 | 0.0754 | 0.0610 | 0.0986 | 0.0685 | 0.0754 | 0.0610 | 0.0988 | 0.0685 |
| BCE (paper) | toys_data | 1e-3 | 1e-5 | 151 | 161 | 0.0350 | 0.0324 | 0.0228 | 0.0435 | 0.0264 | 0.0324 | 0.0228 | 0.0435 | 0.0264 |
| BCE (paper) | games_data | 1e-2 | 1e-5 | 10 | 20 | 0.0324 | 0.0276 | 0.0177 | 0.0441 | 0.0230 | 0.0276 | 0.0177 | 0.0441 | 0.0230 |
| BCE sin usuario | cd_data | 1e-2 | 1e-6 | 89 | 99 | 0.0314 | 0.0387 | — | 0.0516 | 0.0324 | 0.0406 | 0.0295 | 0.0567 | 0.0346 |
| BCE sin usuario | music_data | 1e-3 | 1e-4 | 63 | 73 | 0.0840 | 0.0796 | — | 0.0971 | 0.0745 | 0.0796 | 0.0689 | 0.0971 | 0.0745 |
| BCE sin usuario | toys_data | 1e-2 | 1e-5 | 38 | 48 | 0.0274 | 0.0191 | — | 0.0285 | 0.0158 | 0.0191 | 0.0128 | 0.0285 | 0.0158 |
| BCE sin usuario | games_data | 1e-2 | 1e-5 | 8 | 18 | 0.0335 | 0.0235 | — | 0.0399 | 0.0210 | 0.0237 | 0.0158 | 0.0403 | 0.0212 |
| CE catálogo | cd_data | 1e-2 | 1e-4 | 21 | 31 | 0.0652 | 0.0773 | — | 0.0926 | 0.0680 | 0.0823 | 0.0661 | 0.1013 | 0.0721 |
| CE catálogo | music_data | 1e-2 | 1e-4 | 6 | 16 | 0.1056 | 0.0956 | — | 0.1175 | 0.0896 | 0.0956 | 0.0826 | 0.1175 | 0.0896 |
| CE catálogo | toys_data | 1e-2 | 1e-4 | 26 | 36 | 0.0669 | 0.0603 | — | 0.0761 | 0.0517 | 0.0603 | 0.0466 | 0.0761 | 0.0517 |
| CE catálogo | games_data | 1e-2 | 1e-4 | 13 | 23 | 0.0685 | 0.0566 | — | 0.0827 | 0.0477 | 0.0566 | 0.0393 | 0.0827 | 0.0477 |
| BCE T=3 | cd_data | 1e-3 | 1e-4 | 53 | 63 | 0.0204 | 0.0270 | — | 0.0422 | 0.0252 | 0.0307 | 0.0228 | 0.0486 | 0.0285 |
| BCE T=3 | music_data | 1e-2 | 1e-4 | 11 | 21 | 0.0665 | 0.0677 | — | 0.0847 | 0.0579 | 0.0677 | 0.0524 | 0.0848 | 0.0579 |
| BCE T=3 | toys_data | 1e-3 | 1e-4 | 59 | 69 | 0.0272 | 0.0236 | — | 0.0350 | 0.0190 | 0.0236 | 0.0153 | 0.0350 | 0.0190 |
| BCE T=3 | games_data | 1e-3 | 1e-4 | 45 | 55 | 0.0371 | 0.0330 | — | 0.0543 | 0.0283 | 0.0331 | 0.0216 | 0.0544 | 0.0284 |

**Comparación con la Tabla 1 del paper** (fila Caser del paper; filas propias de la celda 25). La fila comparable es "BCE (paper), título"; las otras variantes son solo referencia.

| Dataset | Fuente | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| cd_data | Paper, Tabla 1 | 0.0469 | 0.0689 | 0.0312 | 0.0382 |
| | BCE (paper), id | 0.0297 | 0.0414 | 0.0230 | 0.0268 |
| | BCE (paper), título | 0.0328 | 0.0465 | 0.0250 | 0.0294 |
| | BCE sin usuario, título | 0.0406 | 0.0567 | 0.0295 | 0.0346 |
| | CE catálogo, título | 0.0823 | 0.1013 | 0.0661 | 0.0721 |
| | BCE T=3, título | 0.0307 | 0.0486 | 0.0228 | 0.0285 |
| music_data | Paper, Tabla 1 | 0.0781 | 0.0977 | 0.0564 | 0.0627 |
| | BCE (paper), id | 0.0754 | 0.0986 | 0.0610 | 0.0685 |
| | BCE (paper), título | 0.0754 | 0.0988 | 0.0610 | 0.0685 |
| | BCE sin usuario, título | 0.0796 | 0.0971 | 0.0689 | 0.0745 |
| | CE catálogo, título | 0.0956 | 0.1175 | 0.0826 | 0.0896 |
| | BCE T=3, título | 0.0677 | 0.0848 | 0.0524 | 0.0579 |
| toys_data | Paper, Tabla 1 | 0.0251 | 0.0384 | 0.0170 | 0.0214 |
| | BCE (paper), id | 0.0324 | 0.0435 | 0.0228 | 0.0264 |
| | BCE (paper), título | 0.0324 | 0.0435 | 0.0228 | 0.0264 |
| | BCE sin usuario, título | 0.0191 | 0.0285 | 0.0128 | 0.0158 |
| | CE catálogo, título | 0.0603 | 0.0761 | 0.0466 | 0.0517 |
| | BCE T=3, título | 0.0236 | 0.0350 | 0.0153 | 0.0190 |
| games_data | Paper, Tabla 1 | 0.0324 | 0.0538 | 0.0211 | 0.0280 |
| | BCE (paper), id | 0.0276 | 0.0441 | 0.0177 | 0.0230 |
| | BCE (paper), título | 0.0276 | 0.0441 | 0.0177 | 0.0230 |
| | BCE sin usuario, título | 0.0237 | 0.0403 | 0.0158 | 0.0212 |
| | CE catálogo, título | 0.0566 | 0.0827 | 0.0393 | 0.0477 |
| | BCE T=3, título | 0.0331 | 0.0544 | 0.0216 | 0.0284 |

- Con BCE (paper) por título, el NDCG@10 de Caser queda por debajo de la Tabla 1 en CDs (0.0294 frente a 0.0382) y Games (0.0230 frente a 0.0280), y por encima en Instruments (0.0685 frente a 0.0627) y Toys (0.0264 frente a 0.0214).
- CE catálogo supera a la Tabla 1 y a las otras variantes en los 4 datasets. NDCG@10 por título: 0.0721, 0.0896, 0.0517 y 0.0477.
- Quitar el embedding de usuario mejora la BCE en CDs e Instruments (NDCG@10 por título 0.0346 y 0.0745, frente a 0.0294 y 0.0685) y la empeora en Toys y Games (0.0158 y 0.0212, frente a 0.0264 y 0.0230).
- BCE T=3 no mejora a la BCE con T=1, salvo en Games (0.0284 frente a 0.0230).
- Ninguna configuración, ganadora o no, llegó al tope de 200 épocas. La que más corrió fue BCE (paper) en CDs con lr 1e-3 y wd 1e-5 (173 épocas).

**Usuarios conocidos y nuevos** (celda 27). HR@10 y NDCG@10 por título en `test`, separando las filas de usuarios con filas en `train` (conocidos) de las de usuarios sin ellas (nuevos):

| Dataset | Variante | HR@10 conocidos | NDCG@10 conocidos | HR@10 nuevos | NDCG@10 nuevos |
|---|---|---|---|---|---|
| cd_data | BCE (paper) | 0.0451 | 0.0338 | 0.0478 | 0.0252 |
| | BCE sin usuario | 0.0447 | 0.0310 | 0.0683 | 0.0382 |
| | CE catálogo | 0.0816 | 0.0598 | 0.1206 | 0.0842 |
| | BCE T=3 | 0.0493 | 0.0326 | 0.0478 | 0.0244 |
| music_data | BCE (paper) | 0.0626 | 0.0402 | 0.1464 | 0.1058 |
| | BCE sin usuario | 0.0618 | 0.0444 | 0.1436 | 0.1141 |
| | CE catálogo | 0.0762 | 0.0541 | 0.1721 | 0.1364 |
| | BCE T=3 | 0.0535 | 0.0334 | 0.1260 | 0.0902 |
| toys_data | BCE (paper) | 0.0202 | 0.0113 | 0.0736 | 0.0460 |
| | BCE sin usuario | 0.0155 | 0.0077 | 0.0453 | 0.0263 |
| | CE catálogo | 0.0379 | 0.0233 | 0.1257 | 0.0886 |
| | BCE T=3 | 0.0221 | 0.0108 | 0.0518 | 0.0297 |
| games_data | BCE (paper) | 0.0404 | 0.0212 | 0.0475 | 0.0248 |
| | BCE sin usuario | 0.0351 | 0.0191 | 0.0452 | 0.0231 |
| | CE catálogo | 0.0564 | 0.0318 | 0.1074 | 0.0625 |
| | BCE T=3 | 0.0419 | 0.0217 | 0.0661 | 0.0347 |

- En la mayoría de los casos, las filas de usuarios nuevos tienen mejores métricas que las de conocidos, también sin el embedding de usuario. Por ejemplo, BCE (paper) en Instruments: NDCG@10 de 0.1058 en nuevos frente a 0.0402 en conocidos. La excepción es CDs con BCE (paper) y con BCE T=3, donde los conocidos tienen mejor NDCG@10. La causa no se analizó; una hipótesis sin verificar es que las secuencias de los usuarios nuevos caen enteras en el período de `test`, cerca de los ítems más recientes.
- El vector cero de los usuarios nuevos no los perjudica de forma uniforme. Sin el embedding de usuario, el NDCG@10 de los nuevos sube en CDs (0.0382 frente a 0.0252) e Instruments (0.1141 frente a 0.1058), y baja en Toys (0.0263 frente a 0.0460) y Games (0.0231 frente a 0.0248).

**Pisos MostPop** (celda 29; MostPop por frecuencia en `train`, definido en el README común): BCE (paper) los supera en los 4 datasets. HR@10 / NDCG@10 por id: CDs 0.0414 / 0.0268, Instruments 0.0986 / 0.0685, Toys 0.0435 / 0.0264 y Games 0.0441 / 0.0230.

**Tabla del paper (Tabla 1, fila Caser):** es la que aparece como "Paper, Tabla 1" en la comparación de arriba.

**Cifras sin output guardado** (medidas fuera del notebook): el 62,9 % (Toys) a 72,5 % (CDs) de las filas de `train` tiene los 3 objetivos de BCE T=3; el padding del 78–93,5 %; el 33,6 / 27,1 / 23,9 / 28,5 % de ventanas del original; el 25–31 % de usuarios nuevos; y los costos marcados así en la tabla de costos.

## Dónde quedan los resultados

Las celdas se numeran desde 0 (posición en el JSON del notebook).

- Locales: ninguno. El notebook no escribe archivos; los resultados quedan en el output de las celdas y en los DataFrames `corridas` (un DataFrame por variante, con las 9 métricas de test, `NDCG@10 valid`, época, épocas corridas, segundos y MiB de GPU por combinación).
- Versionados en `resultados/`: no hay copia de Caser (solo `base_cot`, `bigrec` y `latentr3`).
- Celdas que respaldan cada tabla: datos de test y usuarios nuevos, celda 7; prueba rápida y tiempo por época, celda 11; tiempos de la grilla, celdas 15, 17, 19 y 21; mejor configuración, celda 23; comparación con la Tabla 1, celda 25; conocidos y nuevos, celda 27; pisos MostPop, celda 29.

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| El backward de las convoluciones de MIOpen domina el tiempo de GPU (94 % según el profiler) | Kernels Winograd de MIOpen de ~0,65 ms; `cudnn.benchmark = True` no cambió nada y `cudnn.enabled = False` fue 21 veces más lento (todo sin output guardado) | Convoluciones y puntajes como multiplicación de matrices (`CONV_MATMUL = True`); la época de CDs pasó de 2,36 a 0,60 s | Mismo resultado salvo redondeo; ver Limitaciones para el Δ de métricas en combinaciones cortas |
| No es exactamente reproducible con lr 1e-2 | Dos corridas de la misma combinación pueden dar pesos distintos, también en secuencia. La causa no se identificó; una hipótesis sin verificar es una operación de GPU cuyo orden de suma depende de la carga | Se corre en 6 procesos igual, porque correr en secuencia no garantiza la igualdad. Con lr 1e-3 las repeticiones coincidieron. En 12 repeticiones de lr 1e-2, wd 1e-4, 5 épocas, en 6 procesos a la vez, la versión actual dio los mismos pesos 10 veces y la anterior 12 resultados distintos (sin output guardado) | Con lr 1e-2 el NDCG@10 por título en test llegó a variar entre 0,0059 y 0,0159 en 5 épocas (sin output guardado). La configuración ganadora puede cambiar con otra corrida cuando hay casi un empate en `valid` |
| Warning de MIOpen al arrancar: `ParseAndLoadDb ... gfx1201_32.HIP.fdb.txt is unreadable` y `rocSHMEM Could not open libnuma` (celdas 11 y 15) | Base de datos de MIOpen no disponible para gfx1201 y `libnuma1` no instalado | Sin acción; `sudo apt install libnuma1` quita el segundo aviso (README raíz) | Ninguno observado |
| La tabla de la celda 23 oculta `NDCG@5 id` y redondea `weight_decay` a `0.0000` | pandas muestra 21 columnas en pantalla y el output redondea a 4 decimales | `NDCG@5 id` de BCE (paper) sale de la celda 25; el weight decay se recupera de las líneas de las celdas 15 a 21 | Ninguno; falta `NDCG@5 id` de las otras 3 variantes |
