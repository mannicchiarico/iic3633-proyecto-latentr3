# SASRec (`baseline_sasrec.ipynb`)

Baseline SASRec de LatentR3: usa la clase `SASRec` del port en PyTorch de pmixer, sin modificarla, entrenada con la grilla del paper y evaluada en los cuatro datasets. Corresponde a la fila SASRec de la Tabla 1 del paper (HR@5, HR@10, NDCG@5 y NDCG@10).

**Estado:** corrido en las cuatro variantes (BCE (paper), CE catálogo, BCE todas las posiciones y CE todas las posiciones), con `SEED = 42`. El output guardado del notebook conserva completo solo "CE todas las posiciones" en la tabla final; ver "Dónde quedan los resultados". Pendiente: la medición de paralelismo ([D17](../README.md#decisiones-comunes-d1d17)).

Las decisiones comunes de los baselines tradicionales (códigos D1 a D17), el protocolo, el criterio "por id" y "por título" y las limitaciones comunes están en el [README de `notebooks/`](../README.md#decisiones-comunes-d1d17). Aquí va solo lo propio de SASRec.

## Fuente y versión

| Dato | Valor |
|---|---|
| Repo | https://github.com/pmixer/SASRec.pytorch (port en PyTorch de [kang205/SASRec](https://github.com/kang205/SASRec), escrito para TensorFlow 1.12 y Python 2, que no corre en este entorno) |
| Commit | `9f6a02b86a95e616fe815a0edc92dafa9911773d` (2025-04-15), el último antes del cambio de posición de las LayerNorm (`fcf5588`, 2025-05-21). Hasta ahí cada bloque es igual al del original: LayerNorm solo sobre la consulta, residual desde la consulta y LayerNorm antes de la capa feed-forward |
| Archivo usado | `python/model.py`, sha256 `eefdc8211841d6fbb7b8e16ec68e76af87e148156daaefb45ad0cd3c94d1e8e9` (verificado con `sasrec_baseline.py`, constante `SHA256_MODELO`) |
| Licencia | Apache-2.0 |

```bash
# desde la raíz del repositorio (la carpeta está en .gitignore)
git clone https://github.com/pmixer/SASRec.pytorch SASRec.pytorch
git -C SASRec.pytorch checkout 9f6a02b86a95e616fe815a0edc92dafa9911773d
sha256sum SASRec.pytorch/python/model.py
```

Del port se usa solo la clase `SASRec`: `log2feats` (embeddings, posiciones y bloques de atención con máscara causal), `forward` (solo en "BCE todas las posiciones") e `item_emb`. El notebook verifica el sha256 y carga solo `python/model.py`, que importa únicamente `numpy` y `torch`, sin agregar la carpeta a `sys.path`. El resto (datos, pérdidas, early stopping, evaluación) está en `sasrec_baseline.py`; los datos, las semillas y las métricas vienen de `../baseline_comun.py`.

## Configuración

| Parámetro | Valor | Fuente |
|---|---|---|
| Optimizador | `torch.optim.Adam`, betas (0.9, 0.98), con `weight_decay` | paper, App. B (Adam y weight decay); betas del original (`model.py:103` del port) |
| Grilla | lr {1e-2, 1e-3, 1e-4} × weight decay {1e-4, 1e-5, 1e-6}: 9 combinaciones por dataset y variante | paper, App. B |
| Elección de la mejor | NDCG@10 por id en `valid` (el primero de la grilla si hay empate) | [decisiones comunes](../README.md#decisiones-comunes-d1d17) |
| Largo de secuencia | 10, con padding a la izquierda | paper, App. A |
| Dimensión, bloques, cabezas | 50, 2, 1 | valores por defecto del original (`main.py:20-27`) |
| Dropout | 0.5 (el port usa 0.2) | original (`main.py:20-27`); su README lo baja a 0.2 solo para ml-1m |
| Batch | 128 | valor por defecto del original |
| Épocas y parada | hasta 200 épocas, early stopping con paciencia 10 sobre NDCG@10 por id en `valid`; se conserva la mejor época | decisión del proyecto (`baseline_comun.py`: `MAX_EPOCAS`, `PACIENCIA`) |
| Semilla | 42 (una corrida por combinación) | decisión del proyecto |
| Datos | una muestra por fila de `train/`: historial (hasta 10 ítems) a `item_id` | decisión del proyecto, mismas muestras que los LLM |
| Candidatos | todo el catálogo; HR y NDCG @5 y @10, por id y por título | paper, App. B y Sec. 4.1; criterio en el README de `notebooks/` |
| Inicialización | Xavier normal y padding en cero | `main.py:58-65` del port, replicada en `nuevo_modelo` |
| Hardware | AMD Radeon RX 9070 XT (16 GB), ROCm, `N_PROCESOS = 6` procesos que comparten la GPU, 1 hilo de CPU por proceso | medido, celda 3 y celdas 15 a 21 |

Variantes (36 combinaciones cada una, en celdas separadas):

| Variante | Pérdida | Qué mide |
|---|---|---|
| BCE (paper) | BCE en la última posición: el objetivo de la fila contra un negativo uniforme del catálogo | La que pide el paper; es la comparable con la Tabla 1 |
| CE catálogo | Softmax sobre todo el catálogo en la última posición | Cuánto le cuesta a SASRec la BCE con un negativo |
| BCE todas las posiciones | BCE con un negativo uniforme por posición; cada posición del historial predice el ítem siguiente con su contexto causal, como en el SASRec original; media sobre posiciones que no son padding | Cuánto le cuesta entrenar solo en la última posición |
| CE todas las posiciones | Softmax sobre todo el catálogo en cada posición, media sobre posiciones que no son padding | Si la supervisión de todas las posiciones suma algo cuando la pérdida ya es CE |

Solo "BCE (paper)" se compara con la Tabla 1; las otras tres son referencia. "BCE todas las posiciones" recibe más de una predicción supervisada por fila, y como las filas de un usuario se solapan, cada transición se cuenta hasta 10 veces. No hay fuga de datos: los ítems del historial de una fila de train son objetivos de filas anteriores del mismo usuario, que también están en train. Verificación sin entrenar de "CE todas las posiciones" (CDs, un lote): da lo mismo que calcular la CE posición por posición, y con historiales de largo 1 da lo mismo que "CE catálogo" (medido fuera del notebook, sin output guardado).

## Adaptaciones respecto de la fuente

| Cambio | Motivo | Efecto esperado en los resultados |
|---|---|---|
| Una muestra por fila de `train/` en vez de una ventana por usuario con sus últimas `maxlen` transiciones (`sampler.py:12-33`; `utils.py:33-66` del port) | Mismas muestras que usan los LLM. La ventana por usuario dejaría fuera parte de las transiciones y daría el mismo peso a todos los usuarios (con `maxlen = 10`, usaría el 80,4 % de las transiciones en CDs y el 94,9 % en Toys; medido fuera del notebook, sin output guardado) | Más muestras por usuario activo que el original |
| BCE solo en la última posición (el original usa todas, `model.py:83-91`; `main.py:108-110`) | La BCE del paper se calcula sobre el objetivo; igual que GRU4Rec y el SFT de los LLM | Menos señal por fila; la variante "BCE todas las posiciones" mide el efecto |
| Un negativo uniforme del catálogo por fila, que puede coincidir con el positivo (el original excluye los ítems del usuario, `sampler.py:5-9, 28`) | Convención común de los baselines | Ruido menor en la pérdida |
| Adam con betas (0.9, 0.98) y grilla de lr y weight decay (el original usa lr 1e-3 sin regularización, `l2_emb` 0) | Lo pide el paper | Ninguno respecto del paper |
| `maxlen` 10 (original 50, port 200) | Paper, App. A | Historiales más cortos que los del original |
| Dropout 0.5 (el port usa 0.2) | Valor por defecto del original | Más regularización que el port |
| Hasta 200 épocas con early stopping en `valid` (original: 201 épocas, evalúa cada 20 y no elige época, `main.py:59-79`; port: 1000 épocas) | Protocolo común | Mejor época elegida por `valid` |
| Evaluación sobre todo el catálogo, por fila (el original usa 100 negativos muestreados por usuario, con un máximo de 10 000 usuarios, `util.py:47-69`) | Métricas y candidatos del paper | Métricas más bajas que con candidatos muestreados |
| Ids del catálogo (0 a n-1) se suman 1 dentro del modelo; los puntajes se calculan sin la fila de padding, así que la columna j es el id j | El port reserva el 0 para el padding | Ninguno |
| Sin máscara de padding (el original enmascara el padding como clave y consulta y lo vuelve a cero después de cada bloque, `modules.py:185-208`; `model.py:46, 66`) | No se modifica el código del port | Ver Limitaciones |

## Limitaciones y desviaciones respecto del paper

- **El paper no describe el entrenamiento de SASRec** más allá de la grilla y la BCE con negativos aleatorios (App. B): épocas, parada, número de negativos, arquitectura y datos de entrenamiento son decisión del proyecto. Una diferencia con la Tabla 1 puede deberse al protocolo y no a la implementación.
- **Padding sin máscara en el port.** Las posiciones de padding entran como vectores cero, pero el port no las excluye de la atención ni las vuelve a cero entre bloques, como sí hace el original. Desde el segundo bloque pueden aportar valores distintos de cero a la última posición, sobre todo en historiales cortos. No se corrige para no modificar el código.
- **Dropout 0.5 frente a 0.2 del port**: sigue al SASRec original; no se exploró otro valor.
- **Una sola semilla.** Con diferencias chicas entre configuraciones, la mejor de la grilla puede cambiar con otra semilla. En CE catálogo de Instruments hay empate en `valid` (0.1092) entre wd 1e-6 (época 83) y 1e-5 (época 84); se eligió wd 1e-5 por el orden de la grilla.
- **No es exactamente reproducible en GPU** (RX 9070 XT, CDs, lr 1e-3, wd 1e-5, 2 épocas, dos corridas seguidas en un mismo proceso; medido fuera del notebook, sin output guardado): en CE catálogo las métricas de test difirieron hasta 1,14e-3 y los pesos hasta 0,114; en CE todas las posiciones, hasta 4,70e-4 y 0,069; en BCE y BCE todas las posiciones las métricas coincidieron y los pesos difirieron hasta 1,5e-8 y 6,3e-6. En CPU, dos corridas dieron resultados idénticos en las cuatro variantes (también sin output guardado). La causa no se identificó (hipótesis sin verificar: alguna operación no determinista de la GPU en la atención).
- **Paralelismo sin medir (D17):** la grilla corrió con `N_PROCESOS = 6` sin la medición previa de una época con 1, 3 y 6 procesos, ni la comprobación de que el paralelo da lo mismo que el secuencial en 2 o 3 combinaciones.
- **Ítems ausentes de train:** ver las limitaciones comunes en el [README de `notebooks/`](../README.md).
- **Tope de 200 épocas:** ninguna combinación llegó. La que más corrió fue BCE (paper) en Toys con lr 1e-4 y wd 1e-4: 154 épocas (celda 15).

## Cómo correrlo

1. Entorno: `.venv-rocm` o `.venv-cuda` (ver el README de la raíz).
2. Clonar el port en la raíz del repositorio y fijar el commit (ver "Fuente y versión"). En Colab, la carpeta va en `SASRec.pytorch/` dentro del directorio de trabajo, y `sasrec_baseline.py` y `baseline_comun.py` se suben al mismo directorio.
3. Directorio de trabajo: `notebooks/sasrec/` (el notebook agrega `notebooks/` a `sys.path` para importar `baseline_comun.py`). Los datos se leen de `../../repo_latentr3/data`.
4. Variables a revisar: `N_PROCESOS` (celda 3, 6 en los resultados de abajo).
5. Orden: importaciones y datos (celdas 2 a 7), prueba rápida (celda 11, opcional), definición de `correr_grilla` (celda 13), una celda de grilla por variante (15, 17, 19 y 21), tabla final (23), comparación con el paper (25) y pisos MostPop (27).
6. Para correr solo una variante sin perder los outputs guardados de las otras: correr las celdas de importación, datos, prueba rápida (opcional) y `correr_grilla`, y después solo la celda de esa variante. Las celdas 23 y 25 mostrarían solo las variantes de esa sesión, y la 27 falla si "BCE (paper)" no está en `corridas`.

El notebook imprime una línea por combinación, la mejor configuración por variante y dataset (elegida solo con `valid`), la comparación con el paper y el chequeo de los pisos MostPop. `corridas` guarda un DataFrame por variante. No escribe archivos.

## Costos medidos

| Etapa | GPU | Memoria GPU | RAM | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|---|---|
| Prueba rápida (3 épocas, CDs, BCE (paper)) | RX 9070 XT | 98 MiB | no medido | 2,8 s por época (incluye la evaluación en `valid`) | medido | celda 11 |
| Peor caso de la grilla con 1 proceso | RX 9070 XT | no medido | no medido | 20,6 h | estimado por el notebook a partir de la prueba rápida | celda 11 |
| Grilla BCE (paper), 36 combinaciones, 6 procesos | RX 9070 XT | no medido | no medido | 20,7 min | medido | celda 15 (`[36/36]`) |
| Grilla CE catálogo | RX 9070 XT | no medido | no medido | 25,7 min | medido | celda 17 |
| Grilla BCE todas las posiciones | RX 9070 XT | no medido | no medido | 31,8 min | medido | celda 19 |
| Grilla CE todas las posiciones | RX 9070 XT | no medido | no medido | 27,6 min | medido | celda 21 |
| Paralelismo con 1, 3 y 6 procesos (D17) | RX 9070 XT | no medido | no medido | no medido | pendiente | |

## Resultados

### Resultados propios

Todas las cifras son de `test`, redondeadas a 4 decimales por el notebook (`SEED = 42`, RX 9070 XT, 6 procesos). La mejor configuración se eligió solo con NDCG@10 por id en `valid`. lr, wd, época, épocas corridas y NDCG@10 de `valid` salen de las líneas por combinación de las celdas 15, 17, 19 y 21; la tabla de la celda 23 muestra `0.0000` como wd cuando es 1e-5 o 1e-6, por el redondeo, así que el wd se tomó de esas líneas.

Para BCE (paper), CE catálogo y BCE todas las posiciones, las métricas de `test` salen de la tabla de la celda 25, que no muestra la configuración ganadora; la tabla que la mostraba (celda 23) se volvió a ejecutar solo con CE todas las posiciones. La configuración de esas tres variantes se reconstruyó de las líneas por combinación de las celdas 15, 17 y 19 con la misma regla del código (`idxmax` de NDCG@10 en `valid` por variante y dataset; ante empate gana la primera combinación en el orden de la grilla, lr {1e-2, 1e-3, 1e-4} × wd {1e-4, 1e-5, 1e-6}). La regla da una sola configuración por variante y dataset en los 12 casos, salvo el empate de la tabla de abajo.

Mejor configuración por variante y dataset:

| Variante | Dataset | lr | wd | Época | Épocas corridas | NDCG@10 valid | Fuente |
|---|---|---|---|---|---|---|---|
| BCE (paper) | cd_data | 1e-3 | 1e-6 | 30 | 40 | 0.0471 | celda 15 |
| BCE (paper) | music_data | 1e-3 | 1e-6 | 19 | 29 | 0.0649 | celda 15 |
| BCE (paper) | toys_data | 1e-4 | 1e-6 | 138 | 148 | 0.0595 | celda 15 |
| BCE (paper) | games_data | 1e-3 | 1e-5 | 23 | 33 | 0.0515 | celda 15 |
| CE catálogo | cd_data | 1e-4 | 1e-4 | 120 | 130 | 0.0838 | celda 17 |
| CE catálogo | music_data | 1e-4 | 1e-5 | 84 | 94 | 0.1092 | celda 17. Empate a 4 decimales con wd 1e-6 (época 83 de 93, 0.1092); el log redondea, así que no se puede confirmar cuál eligió `idxmax`: se asume wd 1e-5 por el orden de la grilla, y las métricas de la celda 25 corresponden a la que haya elegido |
| CE catálogo | toys_data | 1e-4 | 1e-5 | 95 | 105 | 0.0794 | celda 17 |
| CE catálogo | games_data | 1e-4 | 1e-4 | 122 | 132 | 0.0754 | celda 17 |
| BCE todas las posiciones | cd_data | 1e-3 | 1e-5 | 37 | 47 | 0.0548 | celda 19 |
| BCE todas las posiciones | music_data | 1e-3 | 1e-5 | 23 | 33 | 0.0665 | celda 19 |
| BCE todas las posiciones | toys_data | 1e-4 | 1e-4 | 99 | 109 | 0.0608 | celda 19 |
| BCE todas las posiciones | games_data | 1e-3 | 1e-6 | 12 | 22 | 0.0503 | celda 19 |
| CE todas las posiciones | cd_data | 1e-2 | 1e-6 | 21 | 31 | 0.0812 | celda 21 |
| CE todas las posiciones | music_data | 1e-3 | 1e-5 | 4 | 14 | 0.1037 | celda 21 |
| CE todas las posiciones | toys_data | 1e-3 | 1e-4 | 24 | 34 | 0.0744 | celda 21 |
| CE todas las posiciones | games_data | 1e-3 | 1e-6 | 33 | 43 | 0.0704 | celda 21 |

Métricas de `test` de la mejor configuración. "sin output guardado" significa que la cifra no está en ningún output actual del notebook, porque la celda 23 se volvió a ejecutar y su output contiene solo "CE todas las posiciones"; no se reconstruyó.

| Variante | Dataset | HR@5 id | NDCG@5 id | HR@10 id | NDCG@10 id | HR@5 título | NDCG@5 título | HR@10 título | NDCG@10 título | Fuente |
|---|---|---|---|---|---|---|---|---|---|---|
| BCE (paper) | cd_data | 0.0640 | 0.0409 | 0.0903 | 0.0493 | 0.0697 | 0.0466 | 0.0986 | 0.0559 | celda 25 (HR@5, HR@10, NDCG@5, NDCG@10) |
| BCE (paper) | music_data | 0.0764 | 0.0423 | 0.1030 | 0.0509 | 0.0765 | 0.0425 | 0.1033 | 0.0512 | celda 25 |
| BCE (paper) | toys_data | 0.0561 | 0.0364 | 0.0772 | 0.0432 | 0.0564 | 0.0368 | 0.0773 | 0.0435 | celda 25 |
| BCE (paper) | games_data | 0.0412 | 0.0262 | 0.0687 | 0.0350 | 0.0414 | 0.0262 | 0.0690 | 0.0351 | celda 25 |
| CE catálogo | cd_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.1145 | 0.0812 | 0.1493 | 0.0925 | celda 25 |
| CE catálogo | music_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0994 | 0.0840 | 0.1209 | 0.0909 | celda 25 |
| CE catálogo | toys_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0821 | 0.0560 | 0.1125 | 0.0658 | celda 25 |
| CE catálogo | games_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0636 | 0.0411 | 0.0985 | 0.0523 | celda 25 |
| BCE todas las posiciones | cd_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0872 | 0.0551 | 0.1192 | 0.0653 | celda 25 |
| BCE todas las posiciones | music_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0826 | 0.0465 | 0.1022 | 0.0527 | celda 25 |
| BCE todas las posiciones | toys_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0600 | 0.0367 | 0.0868 | 0.0454 | celda 25 |
| BCE todas las posiciones | games_data | sin output guardado | sin output guardado | sin output guardado | sin output guardado | 0.0444 | 0.0274 | 0.0749 | 0.0372 | celda 25 |
| CE todas las posiciones | cd_data | 0.0997 | 0.0757 | 0.1257 | 0.0841 | 0.1041 | 0.0786 | 0.1338 | 0.0882 | celda 21 |
| CE todas las posiciones | music_data | 0.0923 | 0.0784 | 0.1150 | 0.0858 | 0.0924 | 0.0786 | 0.1150 | 0.0859 | celda 21 |
| CE todas las posiciones | toys_data | 0.0793 | 0.0523 | 0.1064 | 0.0611 | 0.0795 | 0.0525 | 0.1064 | 0.0612 | celda 21 |
| CE todas las posiciones | games_data | 0.0590 | 0.0376 | 0.0916 | 0.0481 | 0.0590 | 0.0376 | 0.0916 | 0.0481 | celda 21 |

Observaciones (NDCG@10 por título):

- BCE (paper) frente a la Tabla 1: por debajo en CDs (0.0559 frente a 0.0691), Instruments (0.0512 frente a 0.0758) y Toys (0.0435 frente a 0.0510), y prácticamente igual en Games (0.0351 frente a 0.0350). HR@10 por título supera a la Tabla 1 en Instruments, Toys y Games, así que la diferencia está sobre todo en la posición del acierto dentro del top-10.
- CE catálogo supera a la Tabla 1 en los 4 datasets: 0.0925, 0.0909, 0.0658 y 0.0523.
- BCE todas las posiciones supera a BCE (paper) en los 4 datasets: 0.0653 frente a 0.0559 (CDs), 0.0527 frente a 0.0512 (Instruments), 0.0454 frente a 0.0435 (Toys) y 0.0372 frente a 0.0351 (Games).
- CE todas las posiciones queda por debajo de CE catálogo en los 4 datasets (0.0882 frente a 0.0925, 0.0859 frente a 0.0909, 0.0612 frente a 0.0658, 0.0481 frente a 0.0523) y por encima de BCE todas las posiciones (0.0882, 0.0859, 0.0612 y 0.0481 frente a 0.0653, 0.0527, 0.0454 y 0.0372). Con CE, supervisar todas las posiciones no ayuda a SASRec. Frente a la Tabla 1 también supera en los 4 datasets (0.0691, 0.0758, 0.0510 y 0.0350).
- Pisos MostPop (celda 27, solo sobre BCE (paper) por id): se superan en los 4 datasets. HR@10 / NDCG@10: CDs 0.0903 / 0.0493 (piso 0.0197 / 0.0091), Instruments 0.1030 / 0.0509 (0.0659 / 0.0330), Toys 0.0772 / 0.0432 (0.0138 / 0.0082) y Games 0.0687 / 0.0350 (0.0208 / 0.0111). La celda 27 lleva el aviso "Kernel crashed" de VS Code, pero su texto está completo.

### Paper (Tabla 1, fila SASRec)

| Dataset | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| cd_data | 0.0841 | 0.1054 | 0.0622 | 0.0691 |
| music_data | 0.0793 | 0.0950 | 0.0708 | 0.0758 |
| toys_data | 0.0601 | 0.0760 | 0.0458 | 0.0510 |
| games_data | 0.0416 | 0.0633 | 0.0280 | 0.0350 |

(Valores copiados del diccionario `PAPER` de la celda 25.)

## Dónde quedan los resultados

Las celdas se numeran desde 0 (posición en el JSON del notebook). No se escriben archivos: los resultados están solo en los outputs de `baseline_sasrec.ipynb` y en el DataFrame `corridas` de la sesión; no hay copia en `resultados/`.

| Contenido | Celda |
|---|---|
| Prueba rápida (tiempo, memoria de GPU, peor caso) | 11 |
| Líneas por combinación y tiempo de la grilla de BCE (paper) | 15 |
| Ídem CE catálogo | 17 |
| Ídem BCE todas las posiciones | 19 |
| Ídem CE todas las posiciones y su tabla con las 8 métricas | 21 |
| Tabla final por variante (el output guardado contiene solo CE todas las posiciones) | 23 |
| Comparación con el paper (HR@5, HR@10, NDCG@5, NDCG@10 por título, y por id solo para BCE (paper)); sin CE todas las posiciones | 25 |
| Pisos MostPop | 27 |

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| El output de la celda 23 contiene solo CE todas las posiciones | Se volvió a ejecutar la celda con `corridas` de una sola variante | Las cifras de las otras variantes se conservan desde las celdas 15, 17, 19 y 25; las métricas por id de CE catálogo y BCE todas las posiciones no tienen output | No hay forma de verificar esas 16 cifras sin volver a correr las grillas |
| Celda 25 sin CE todas las posiciones | Su output es anterior a esa variante | Las cifras de esa variante están en la celda 21 | Ninguno |
| Celda 27 con "Kernel crashed" | Aviso de VS Code al guardar | El texto del output está completo | Ninguno |
| Tabla con `0.0000` en wd | `round(4)` sobre 1e-5 y 1e-6 | El wd se toma de las líneas por combinación | Ninguno |
| No determinismo en GPU | Causa no identificada | Sin resolver | La mejor configuración puede cambiar si hay casi un empate en `valid` |
| Medición D17 pendiente | No se hizo antes de fijar `N_PROCESOS = 6` | Pendiente | Sin efecto esperado en las métricas; el tiempo de la grilla depende de este valor |
