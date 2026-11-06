# GRU4Rec (`baseline_gru4rec.ipynb`)

Baseline GRU4Rec de LatentR3: entrena el modelo de la implementación oficial en PyTorch con el protocolo del paper y las mismas muestras que usan los métodos con LLM, y mide HR@5, HR@10, NDCG@5 y NDCG@10 por id y por título. Se compara con la fila GRU4Rec de la Tabla 1 del paper.

**Estado:** corrido el 2026-11-05 (grilla de BCE (paper) y CE catálogo, celdas 11 y 13) y el 2026-11-06 (BCE todas las posiciones, celda 15). CE todas las posiciones (celda 17) está pendiente: terminó en `KeyboardInterrupt` y no tiene resultados.

El entrenamiento está en `gru4rec_baseline.py`, junto al notebook, y el protocolo común (datos, métricas, semillas) en `../baseline_comun.py`. Está en un `.py` porque la grilla corre en 6 procesos en paralelo y Jupyter no puede enviar a otros procesos las funciones definidas en sus celdas. Los pisos MostPop, el criterio por id y por título, los candidatos, la búsqueda y las limitaciones comunes están en el [README de `notebooks/`](../README.md) (sección de pisos y protocolo común). Los códigos D1–D7 que aparecen abajo son las [decisiones comunes D1–D17](../README.md#decisiones-comunes-d1d17).

## Fuente y versión

| Elemento | Valor |
|---|---|
| Repo de origen | https://github.com/hidasib/GRU4Rec_PyTorch_Official |
| Commit | `d1fc31105577665d2f105de1c8d9a38d223a9a22` |
| Archivo usado | `gru4rec_pytorch.py`, solo la clase `GRU4RecModel` (métodos `embed`, `hidden_step`, `score_items`), sin modificar |
| sha256 de `gru4rec_pytorch.py` | `d5b52620f76f36b13e229a53d11176d822d472270f3183fda25a034051e61278` (coincide con el archivo de la copia local y con `SHA256_MODELO` en `gru4rec_baseline.py:19`) |

```bash
# desde la raíz del repositorio
git clone https://github.com/hidasib/GRU4Rec_PyTorch_Official GRU4Rec_PyTorch_Official
git -C GRU4Rec_PyTorch_Official checkout d1fc31105577665d2f105de1c8d9a38d223a9a22
sha256sum GRU4Rec_PyTorch_Official/gru4rec_pytorch.py
```

La carpeta `GRU4Rec_PyTorch_Official/` está en `.gitignore` (no se versiona). `inicializar_proceso` verifica el sha256 antes de importar el archivo y se detiene si no coincide. Ese archivo importa solo `math`, `numpy`, `pandas`, `torch`, `collections` y `time`.

Las corridas guardadas usaron una copia local sin `.git`. Se identificó el commit comparando el hash git de sus 19 archivos con el árbol de cada commit en GitHub: coinciden todos con `d1fc311` (comprobación hecha fuera del notebook, sin output guardado).

## Configuración

| Parámetro | Valor | Fuente |
|---|---|---|
| Datos | Una muestra por fila de `train/`: historial (hasta 10 ítems) -> `item_id`. Filas de train / valid / test: CDs 49251 / 6156 / 6157, Instruments 66500 / 8312 / 8313, Toys 53898 / 6737 / 6738, Games 75175 / 9397 / 9397 | medido, celda 7 (D1) |
| Catálogo (ítems) | CDs 5841, Instruments 5030, Toys 6299, Games 5308 | medido, celda 7 |
| Largo de contexto | Máximo 10 ítems en entrenamiento y evaluación | paper, App. A |
| Optimizador | `torch.optim.Adam` con `weight_decay` (L2 sumado al gradiente, no AdamW) | paper, App. B; lectura literal de la decisión del proyecto |
| Grilla | lr en {1e-2, 1e-3, 1e-4} x weight decay en {1e-4, 1e-5, 1e-6}: 9 combinaciones por variante y dataset | decisión del proyecto, celdas 11 y 15 |
| Selección | Mejor combinación por NDCG@10 por id en `valid` (D6); las métricas de test se calculan para todas pero no participan en la elección | decisión del proyecto, `gru4rec_baseline.py:183` |
| Épocas y parada | Tope 200, paciencia 10 sobre NDCG@10 por id en `valid`, se conserva la mejor época (D7) | `../baseline_comun.py:18-19`, `gru4rec_baseline.py:159` |
| Batch | 64 | el paper no lo especifica; valor por defecto del GRU4Rec oficial |
| Arquitectura | Una capa GRU de 100, sin dropout, embedding compartido entre entrada y salida | el paper no lo especifica; valores por defecto del GRU4Rec oficial (`gru4rec_baseline.py:73`) |
| Pérdida BCE (paper) | BCE del objetivo contra un negativo uniforme del catálogo por fila, con la pérdida en el último paso | paper, App. B ("randomly sampled negative items"); el número de negativos no se especifica, uno es la convención de SASRec |
| Semilla | `SEED = 42`, fijada al inicio de cada combinación, así que el resultado no depende del orden ni del paralelismo | `../baseline_comun.py:16` |
| Paralelismo | `N_PROCESOS = 6` combinaciones a la vez, compartiendo la GPU | celda 3 |
| GPU | AMD Radeon RX 9070 XT | medido, celda 3 |
| CPU | Ryzen 5 7600X | sin output guardado (dato del README anterior) |
| Memoria GPU y RAM | no medido | |

### Variantes

| Nombre | Qué cambia | Se compara con la Tabla 1 |
|---|---|---|
| BCE (paper) | Configuración de arriba | Sí |
| CE catálogo | Softmax sobre todo el catálogo (5-6 mil ítems, sin muestrear) en vez de un negativo; misma grilla. Mide cuánto deja de lado GRU4Rec con la pérdida del paper | No, solo referencia |
| BCE todas las posiciones | Cada posición del historial predice el ítem siguiente (en la última, el objetivo de la fila), con un negativo uniforme por posición; pérdida media sobre las posiciones (`codificar_todas`, `perdida_bce_todas`). Es la supervisión del GRU4Rec oficial con la pérdida del paper; mide el costo de entrenar solo en la última posición (D1) | No, solo referencia |
| CE todas las posiciones | Softmax sobre el catálogo en cada posición, media sobre las posiciones (`perdida_ce_todas`). Es la variante más cercana al GRU4Rec oficial que permite el protocolo | No, pendiente (sin resultados) |

Detalles de las variantes por posición: con historiales de largo 1 se reducen a BCE (paper) y a CE catálogo respectivamente (verificado por el proyecto fuera del notebook, sin output guardado). Los lotes agrupan filas del mismo largo, así que no hay padding. Las filas de un usuario se solapan, así que cada transición se cuenta hasta 10 veces. No hay fuga de datos: los ítems del historial de una fila de train son objetivos de filas anteriores del mismo usuario, que también están en train.

## Adaptaciones respecto de la fuente

Del código oficial se usa solo `GRU4RecModel`; el resto se escribió en `gru4rec_baseline.py`.

| Cambio | Motivo | Efecto esperado en los resultados |
|---|---|---|
| Lotes de filas con el mismo largo de historial, estado oculto en cero por fila (el oficial usa `SessionDataIterator`, con estado acumulado por toda la sesión) | Con el iterador no se puede limitar el contexto a 10 ítems sin perder transiciones o contexto | Contexto igual al de los métodos con LLM; difiere del GRU4Rec original |
| Retropropagación completa por los hasta 10 pasos (el oficial la trunca a un paso, `h.detach_()`, `gru4rec_pytorch.py:470`) | Con 10 pasos no hace falta truncar | Posible mejora del gradiente respecto del oficial |
| BCE con un negativo uniforme por fila (el oficial usa 2048 negativos por popularidad compartidos en el lote, `SampleCache`) | El paper pide BCE con negativos aleatorios | Menos señal por paso que la pérdida oficial |
| Pérdida BCE (paper) o cross-entropy sobre el catálogo (el oficial usa cross-entropy o BPR-max con negativos muestreados) | BCE por el paper; CE como referencia | CE catálogo supera a BCE (ver Resultados) |
| Adam con embeddings densos y `weight_decay` (el oficial usa Adagrad disperso, sin weight decay) | Lo pide el paper; Adam no acepta gradientes dispersos | Distinto régimen de lr que el oficial (Adagrad con lr 0.05) |
| Early stopping en `valid`, tope de 200 épocas (el oficial corre 10 épocas fijas y se queda con la última) | Las 10 épocas oficiales están pensadas para Adagrad con lr altas y millones de eventos | Las ganadoras de BCE necesitan entre 60 y 195 épocas |
| Catálogo completo de `info/`, el id del ítem es su índice en el modelo (el oficial usa los ítems de train y descarta los desconocidos en evaluación) | Mismos candidatos que los LLM; ninguna fila de test se descarta | Ítems ausentes de train tienen embedding sin entrenar (ver limitaciones comunes en `../README.md`) |
| Inicialización en torch de `GRU4RecModel` (el oficial ofrece `compatibility_mode` con numpy) | `compatibility_mode` falla con numpy 2 | Ninguno esperado |
| Evaluación: HR y NDCG @5 y @10 por fila, por id y por título (el oficial reporta Recall y MRR @20 por sesión) | Métricas y formato del paper | No comparable con las cifras del GRU4Rec original |

El repo oficial trae hiperparámetros ajustados por dataset en `paramfiles/`, ninguno para Amazon.

## Limitaciones y desviaciones respecto del paper

- **Datos de entrenamiento:** el paper no dice cómo adapta GRU4Rec a las muestras de los LLM. La adaptación (una muestra por fila, pérdida en el último paso) es decisión del proyecto; BCE todas las posiciones muestra el efecto de otra opción.
- **Pérdida, negativos, batch y arquitectura:** el paper no los especifica; se usaron los valores de la tabla de Configuración. Las cifras dependen de esas elecciones.
- **Tope de épocas en CDs:** la ganadora de BCE (paper) en CDs (lr 1e-3, wd 1e-5) eligió la época 195 y corrió 200 de 200 (celda 11), por lo que seguía mejorando al llegar al tope y puede estar subentrenada. La ganadora de BCE todas las posiciones en CDs corrió 145 épocas (celda 15), sin llegar al tope.
- **Combinaciones que llegan al tope** (no ganadoras): CE catálogo en CDs con lr 1e-4 y wd 1e-4 (época 200, celda 11) y BCE todas las posiciones en Games con lr 1e-4 y wd 1e-5 (época 200 de 200, celda 15).
- **Una sola semilla:** con diferencias chicas entre configuraciones, la elección de la grilla puede cambiar con otra semilla. Con una versión anterior del notebook se observaron diferencias de hasta ~14 % en NDCG@10 entre corridas (sin output guardado).
- **Reproducibilidad en GPU** (prueba hecha fuera del notebook, sin output guardado; RX 9070 XT, CDs, lr 1e-3, wd 1e-5, 2 épocas, dos corridas seguidas en el mismo proceso): BCE, CE y BCE todas las posiciones se repitieron exactas; CE todas las posiciones dio las mismas métricas, con pesos que difirieron hasta 6,56e-7. En CPU, dos corridas dieron resultados idénticos en las 4 variantes.
- **Ítems ausentes de train:** ver las limitaciones comunes en el [README de `notebooks/`](../README.md).

## Cómo correrlo

1. Entorno `.venv-rocm` o `.venv-cuda` (ver el README de la raíz).
2. Clonar `GRU4Rec_PyTorch_Official/` en la raíz del repositorio y fijar el commit (ver Fuente y versión). En Colab, la carpeta va en `GRU4Rec_PyTorch_Official/` dentro del directorio de trabajo, y hay que subir `gru4rec_baseline.py` y `../baseline_comun.py` al directorio de trabajo.
3. Ejecutar el notebook con `notebooks/gru4rec/` como directorio de trabajo, desde la celda 2 en orden. Los datos se leen de `repo_latentr3/data`. El notebook agrega `notebooks/` a `sys.path` para importar `baseline_comun`.
4. Celda 11: grilla de 72 combinaciones (BCE (paper) y CE catálogo x 4 datasets x 9). Celda 13: tabla de comparación con la Tabla 1 (necesita `resultados` de la celda 11). Celdas 15 y 17: BCE todas las posiciones y CE todas las posiciones, 36 combinaciones cada una; no dependen de la celda 11, basta con las celdas de importación y datos.
5. Si la GPU tiene poca memoria o la CPU pocos núcleos, bajar `N_PROCESOS` (celda 3).

El notebook imprime una línea por combinación y no escribe archivos. `corridas` y `corridas_todas` guardan todas las combinaciones en memoria.

## Costos medidos

| Etapa | GPU | Memoria GPU | RAM | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|---|---|
| Grilla BCE (paper) + CE catálogo, 72 combinaciones, 6 procesos | RX 9070 XT | no medido | no medido | 27,2 min | medido | celda 11 |
| BCE todas las posiciones, 36 combinaciones, 6 procesos | RX 9070 XT | no medido | no medido | 29,0 min | medido | celda 15 |
| CE todas las posiciones, 36 combinaciones | RX 9070 XT | no medido | no medido | no medido (interrumpida) | | celda 17 |
| Una época de CDs, 6 procesos frente a 1 | RX 9070 XT, Ryzen 5 7600X | no medido | no medido | 6 procesos dan unas 3,6 veces más épocas por segundo | medido fuera del notebook, sin output guardado | |

El modelo es tan chico que, con un solo proceso, la GPU espera casi todo el tiempo a Python; por eso se usan 6 procesos.

## Resultados

Cifras redondeadas a 4 decimales por el notebook. Las métricas son de `test`; la configuración se eligió con NDCG@10 por id en `valid`. `wd` no sale en las tablas del notebook (aparece como 0.0000 por el redondeo) y se recuperó de la línea que el notebook imprime por cada combinación, buscando la que tiene el mismo lr, la misma época y el mismo NDCG@10 en `valid`.

### Mejor configuración por variante y dataset

Fuente: medido, celdas 11 y 13 (BCE (paper) y CE catálogo) y celda 15 (BCE todas las posiciones). "Épocas corridas": en la celda 15 es el valor impreso por el notebook; en BCE (paper) y CE catálogo es deducción del proyecto (base: época elegida + paciencia 10, con tope 200, según el bucle de `entrenar`), porque la celda 11 no las imprime.

| Variante | Dataset | lr | wd | Época | Épocas corridas | NDCG@10 valid | HR@5 id | NDCG@5 id | HR@10 id | NDCG@10 id | HR@5 título | NDCG@5 título | HR@10 título | NDCG@10 título |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BCE (paper) | cd_data | 1e-3 | 1e-5 | 195 | 200 (tope, deducción) | 0.0638 | 0.0754 | 0.0555 | 0.1020 | 0.0641 | 0.0797 | 0.0580 | 0.1095 | 0.0676 |
| BCE (paper) | music_data | 1e-3 | 1e-5 | 60 | 70 (deducción) | 0.0892 | 0.0811 | 0.0685 | 0.1007 | 0.0748 | 0.0812 | 0.0686 | 0.1009 | 0.0749 |
| BCE (paper) | toys_data | 1e-3 | 1e-5 | 131 | 141 (deducción) | 0.0526 | 0.0499 | 0.0331 | 0.0703 | 0.0396 | 0.0499 | 0.0331 | 0.0703 | 0.0396 |
| BCE (paper) | games_data | 1e-3 | 1e-5 | 79 | 89 (deducción) | 0.0512 | 0.0384 | 0.0257 | 0.0629 | 0.0336 | 0.0384 | 0.0257 | 0.0630 | 0.0336 |
| CE catálogo | cd_data | 1e-2 | 1e-4 | 7 | 17 (deducción) | 0.0755 | 0.0884 | 0.0716 | 0.1117 | 0.0791 | 0.0940 | 0.0753 | 0.1200 | 0.0836 |
| CE catálogo | music_data | 1e-3 | 1e-5 | 5 | 15 (deducción) | 0.1093 | 0.0948 | 0.0825 | 0.1134 | 0.0885 | 0.0949 | 0.0826 | 0.1137 | 0.0886 |
| CE catálogo | toys_data | 1e-3 | 1e-5 | 11 | 21 (deducción) | 0.0740 | 0.0698 | 0.0539 | 0.0850 | 0.0588 | 0.0699 | 0.0541 | 0.0853 | 0.0590 |
| CE catálogo | games_data | 1e-3 | 1e-4 | 17 | 27 (deducción) | 0.0760 | 0.0597 | 0.0413 | 0.0897 | 0.0510 | 0.0597 | 0.0413 | 0.0897 | 0.0510 |
| BCE todas las posiciones | cd_data | 1e-3 | 1e-6 | 135 | 145 | 0.0639 | 0.0827 | 0.0632 | 0.0994 | 0.0686 | 0.0861 | 0.0655 | 0.1046 | 0.0715 |
| BCE todas las posiciones | music_data | 1e-3 | 1e-5 | 66 | 76 | 0.0931 | 0.0842 | 0.0711 | 0.1022 | 0.0769 | 0.0842 | 0.0711 | 0.1022 | 0.0769 |
| BCE todas las posiciones | toys_data | 1e-3 | 1e-5 | 126 | 136 | 0.0636 | 0.0545 | 0.0382 | 0.0793 | 0.0462 | 0.0545 | 0.0382 | 0.0794 | 0.0462 |
| BCE todas las posiciones | games_data | 1e-3 | 1e-5 | 82 | 92 | 0.0545 | 0.0420 | 0.0278 | 0.0666 | 0.0357 | 0.0420 | 0.0278 | 0.0666 | 0.0357 |
| CE todas las posiciones | todos | pendiente | | | | | | | | | | | | |

### Tabla 1 del paper y comparación

Fila GRU4Rec, paper, Tabla 1 (celda 13, diccionario `PAPER`). La fila comparable de este trabajo es "BCE (paper), título"; las demás son referencia.

| Dataset | Fuente | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| cd_data | paper, Tabla 1 | 0.0481 | 0.0669 | 0.0365 | 0.0425 |
| | BCE (paper), id | 0.0754 | 0.1020 | 0.0555 | 0.0641 |
| | BCE (paper), título | 0.0797 | 0.1095 | 0.0580 | 0.0676 |
| | CE catálogo, título | 0.0940 | 0.1200 | 0.0753 | 0.0836 |
| music_data | paper, Tabla 1 | 0.0766 | 0.0960 | 0.0630 | 0.0692 |
| | BCE (paper), id | 0.0811 | 0.1007 | 0.0685 | 0.0748 |
| | BCE (paper), título | 0.0812 | 0.1009 | 0.0686 | 0.0749 |
| | CE catálogo, título | 0.0949 | 0.1137 | 0.0826 | 0.0886 |
| toys_data | paper, Tabla 1 | 0.0417 | 0.0564 | 0.0305 | 0.0352 |
| | BCE (paper), id | 0.0499 | 0.0703 | 0.0331 | 0.0396 |
| | BCE (paper), título | 0.0499 | 0.0703 | 0.0331 | 0.0396 |
| | CE catálogo, título | 0.0699 | 0.0853 | 0.0541 | 0.0590 |
| games_data | paper, Tabla 1 | 0.0322 | 0.0517 | 0.0207 | 0.0270 |
| | BCE (paper), id | 0.0384 | 0.0629 | 0.0257 | 0.0336 |
| | BCE (paper), título | 0.0384 | 0.0630 | 0.0257 | 0.0336 |
| | CE catálogo, título | 0.0597 | 0.0897 | 0.0413 | 0.0510 |

### Lectura de los resultados

- Con BCE (paper) y el criterio por título, GRU4Rec supera a la Tabla 1 en las 4 métricas de los 4 datasets. La mayor diferencia está en CDs: NDCG@10 de 0.0676 frente a 0.0425.
- CE catálogo supera a BCE (paper) en todos los datasets. NDCG@10 por título: 0.0836 frente a 0.0676 (CDs), 0.0886 frente a 0.0749 (Instruments), 0.0590 frente a 0.0396 (Toys) y 0.0510 frente a 0.0336 (Games).
- BCE todas las posiciones mejora a BCE (paper) en NDCG@10 por título en los 4 datasets: 0.0715 frente a 0.0676 (CDs), 0.0769 frente a 0.0749 (Instruments), 0.0462 frente a 0.0396 (Toys) y 0.0357 frente a 0.0336 (Games). Queda por debajo de CE catálogo en los 4, y por encima de la Tabla 1 (0.0425, 0.0692, 0.0352 y 0.0270).
- Pisos MostPop: ver la sección de pisos del [README de `notebooks/`](../README.md). Este notebook no tiene la celda de pisos.

## Dónde quedan los resultados

Las celdas se numeran desde 0 (posición en el JSON del notebook).

| Qué | Dónde | Celda |
|---|---|---|
| Líneas por combinación, grilla BCE (paper) y CE catálogo (72) | output del notebook | 11 |
| Tabla "mejor por variante" y métricas por id y por título | output del notebook | 11 |
| Comparación con la Tabla 1 | output del notebook | 13 |
| BCE todas las posiciones (36 combinaciones, tabla de la mejor y épocas corridas) | output del notebook | 15 |
| CE todas las posiciones | sin output (`KeyboardInterrupt`) | 17 |

No hay archivos de resultados: el notebook no escribe a `checkpoints/` y no existe `resultados/gru4rec/`. Los números solo están en los outputs de `baseline_gru4rec.ipynb`.

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| La celda 13 muestra "The Kernel crashed" y no tiene número de ejecución | El kernel se cayó de VS Code después de generar la salida | La tabla de la celda 13 está completa y sus cifras coinciden con la celda 11 | Ninguno |
| `wd` aparece como 0.0000 en las tablas de las celdas 11 y 13 | `round(4)` en el notebook | Se recupera de la línea impresa por combinación (ver Resultados) | Ninguno; solo afecta la lectura |
| La celda 11 no imprime las épocas corridas | El módulo con el que se corrió no las devolvía | Se deducen (época + paciencia 10, tope 200); la celda 15 sí las imprime | Ninguno en las métricas |
| CE todas las posiciones (celda 17) interrumpida | `KeyboardInterrupt` | Pendiente de correr | Sin resultados para esa variante |
| Aviso `rocSHMEM Could not open libnuma` en los outputs | Falta `libnuma1` en el sistema (ver README de la raíz) | No afecta con una sola GPU; se elimina con `sudo apt install libnuma1` | Ninguno |
