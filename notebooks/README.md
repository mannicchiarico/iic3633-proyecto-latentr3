# Notebooks

Una carpeta por modelo (notebook, `.py` y README), más el protocolo común de los baselines tradicionales, que se define en este archivo.

| Carpeta | Modelo | Estado | README |
|---|---|---|---|
| `gru4rec/` | GRU4Rec (código oficial `d1fc311`) | Corrido en las 4 categorías, salvo CE todas las posiciones (pendiente) | [README](gru4rec/README.md) |
| `sasrec/` | SASRec (port de pmixer `9f6a02b`) | Corrido en las 4 categorías | [README](sasrec/README.md) |
| `caser/` | Caser (`caser_pytorch` `c4e10e4`) | Corrido en las 4 categorías | [README](caser/README.md) |
| `bigrec/` | BIGRec (Qwen2.5-1.5B con el código de D3 `585c842`) | Corrido solo en Toys | [README](bigrec/README.md) |
| `d3/` | D3 sobre el modelo de BIGRec | Corrido solo en Toys | [README](d3/README.md) |
| `base_cot/` | Base y COT (Qwen2.5-1.5B sin ajustar) | Base corrido en las 4 categorías; COT solo en Toys | [README](base_cot/README.md) |
| `latentr3/` | LatentR3 (SFT + LR-GRPO; código de los autores `c558b44` con una copia parcheada) | Implementado, sin entrenamiento completo (solo una prueba de tiempo en Toys) | [README](latentr3/README.md) |

Otros archivos:
- `baseline_comun.py`: constantes, semillas, carga de datos, métricas y carga verificada del código oficial. Lo importan GRU4Rec, SASRec, Caser, BIGRec, D3, Base/COT y LatentR3.
- `datos.ipynb`: descarga y preprocesamiento de los datos (ver el [README de la raíz](../README.md)).
- `modelos_exploracion.ipynb`: análisis exploratorio y baselines simples (Random, MostPop, LastItem).

## Protocolo común de los baselines tradicionales

Aplica a GRU4Rec, SASRec y Caser. Lo propio de cada modelo (arquitectura, pérdida exacta, adaptaciones al código original) está en el README de su carpeta.

El paper describe sus baselines tradicionales (Caser, GRU4Rec y SASRec) en una sola frase del Apéndice B:

> *"For traditional recommendation models, we use the Adam optimizer and perform grid searches over learning rates 1e-2, 1e-3, 1e-4 and weight decay values 1e-4, 1e-5, 1e-6, and all models are trained using Binary Cross-Entropy loss with randomly sampled negative items."*

A eso se suman el largo máximo de secuencia de 10 (Apéndice A) y las métricas HR y NDCG @5 y @10 (sección 4.1). El resto no está especificado, y el repositorio de los autores no incluye código de los baselines.

| Aspecto | Paper | Implementación | Justificación | Fuente |
|---|---|---|---|---|
| Optimizador | Adam (Apéndice B) | `torch.optim.Adam` con `weight_decay` (L2 sumado al gradiente); las betas de cada modelo, en su README | Lectura literal de "Adam optimizer ... weight decay"; no es AdamW | paper, App. B; decisión del proyecto (D5) |
| Búsqueda | lr {1e-2, 1e-3, 1e-4} × weight decay {1e-4, 1e-5, 1e-6} | La misma grilla; se elige por NDCG@10 por id en `valid` | `test` no se usa para elegir | paper, App. B; decisión del proyecto (D6) |
| Pérdida | BCE con negativos aleatorios | BCE: el objetivo contra un negativo uniforme del catálogo por fila | El número de negativos no se especifica; uno es la convención de SASRec | paper, App. B; decisión del proyecto (D4) |
| Datos de entrenamiento | No dice cómo se adaptan los modelos | Una muestra por fila de `train/`: historial (≤10 ítems) → `item_id`, con la pérdida sobre el objetivo de la fila | Son las mismas muestras de la Tabla 3 con las que se entrenan los LLM, con el mismo contexto | paper, Tabla 3; decisión del proyecto (D1) |
| Largo de secuencia | Máximo 10 | ≤10 ítems en entrenamiento y evaluación | Coincide | paper, App. A |
| Épocas y parada | No especificadas para los baselines (el SFT de los LLM usa early stopping con paciencia 1) | Hasta 200 épocas, early stopping por NDCG@10 (id) en `valid` con paciencia 10, mejor época | Protocolo común de los baselines tradicionales del proyecto | decisión del proyecto (D7); `MAX_EPOCAS` y `PACIENCIA` en `baseline_comun.py` |
| Candidatos | "Full ranking" (Apéndice B, al describir BIGRec) | Todo el catálogo de `info/` | Los mismos candidatos que la decodificación restringida de los LLM | paper, App. B; decisión del proyecto (D3) |
| Criterio de acierto | No especificado; `calc.py` de los autores compara títulos | Se reportan los dos: por id y por título | Por título es lo comparable con la Tabla 1; por id es lo estándar en modelos de ids | repo de los autores, `calc.py`; decisión del proyecto (D9) |
| Historial en el ranking | No especificado | Los ítems del historial no se excluyen | Igual que `calc.py`, donde el LLM puede generar un ítem del historial | decisión del proyecto (D8) |
| Corridas | Una corrida sin fuentes de aleatoriedad (Apéndice E) | Una semilla fija (42) | Igual que el paper; la varianza se discute en Limitaciones comunes | paper, App. E; `SEED` en `baseline_comun.py` |

### Decisiones comunes (D1–D17)

Los README de cada carpeta usan estos códigos.

| Código | Decisión | Fuente |
|---|---|---|
| D1 | Datos: los CSV de `repo_latentr3/data/` sin re-split; una muestra por fila de `train` (historial de 1 a 10 ítems → `item_id`), con la pérdida solo sobre el objetivo de la fila | paper, Tabla 3; decisión del proyecto |
| D2 | Contexto: ≤10 ítems en entrenamiento y evaluación; cada fila usa su historial real | paper, App. A |
| D3 | Catálogo e ids: todo el catálogo de `info/`; los ids son `0..n-1` (los asigna `process.py`) y se usan como índice del modelo, verificado con un `assert` en `cargar_catalogo`; ningún ítem ni fila se descarta, aunque no aparezca en train | decisión del proyecto |
| D4 | Pérdida BCE con 1 negativo: el objetivo contra un negativo uniforme del catálogo por fila; puede coincidir con el positivo (probabilidad 1/n); suma de los 2 términos por fila dividida por el tamaño del lote | paper, App. B (el número de negativos no está especificado); decisión del proyecto |
| D5 | Adam con L2: `torch.optim.Adam(lr, weight_decay)`, no AdamW; embeddings densos | paper, App. B |
| D6 | Grilla de 9 combinaciones (lr × wd); se elige por NDCG@10 por id en `valid`, y solo la elegida se evalúa en `test` | paper, App. B; decisión del proyecto |
| D7 | Hasta 200 épocas, early stopping por NDCG@10 (id) en `valid` con paciencia 10, y se recupera la mejor época; si la configuración ganadora llega a 200, se anota como limitación | decisión del proyecto |
| D8 | Ranking top-50 sin excluir el historial: puntaje para todo el catálogo, top-50 ids por fila | repo de los autores, `calc.py`; decisión del proyecto |
| D9 | Métricas por id y por título: HR y NDCG @5 y @10 con un solo relevante, 1/log2(pos+2); por id sobre el top-10 de ids; por título, ids → títulos sin repetir, en orden, top-10 | repo de los autores, `calc.py`; decisión del proyecto |
| D10 | Normalización de títulos como `calc.py`: `.strip().strip('"').strip()` (el id se separa desde la derecha del tabulador) | repo de los autores, `calc.py`; `cargar_catalogo` en `baseline_comun.py` |
| D11 | Arquitectura por defecto del código oficial de cada modelo, salvo lo que exija el protocolo (largo 10, catálogo completo); los valores se toman del código oficial | el paper no lo especifica; decisión del proyecto |
| D12 | Código oficial sin modificar: se importa el modelo y el entrenamiento se escribe en el notebook; el código va en la raíz del repositorio, ignorado por git. Excepción en Caser: las convoluciones y el puntaje de todo el catálogo se calculan con multiplicaciones de matrices equivalentes, sin tocar el archivo original (ver el README de Caser) | decisión del proyecto |
| D13 | Semilla 42, con `fijar_semillas()` (random, numpy, torch) antes de cada combinación | paper, App. E (una corrida); `SEED` en `baseline_comun.py` |
| D14 | Variante CE catálogo: cross-entropy sobre todo el catálogo, misma grilla; se reporta aparte y no se compara con la Tabla 1 | decisión del proyecto |
| D15 | Formato de reporte: por variante y dataset, lr, wd, época, NDCG@10 en valid y las 8 métricas de test; además una tabla "Paper / BCE por id / BCE por título / CE por título" | decisión del proyecto |
| D16 | Entorno local y Colab (`IN_COLAB`): cada notebook corre en su carpeta de `notebooks/` y lee los datos desde `../../repo_latentr3/data` (en Colab, `data_amazon_2018/`); los notebooks de los baselines tradicionales no escriben archivos | decisión del proyecto |
| D17 | Cálculo paralelo de la grilla: entrenamiento en un `.py` junto al notebook y combinaciones repartidas con `ProcessPoolExecutor` (contexto `spawn`, `N_PROCESOS = 6` sobre la GPU, un hilo de CPU por proceso); si el modelo ya satura la GPU con un proceso, se deja secuencial. GRU4Rec y Caser corren en 6 procesos | decisión del proyecto |

### Criterio de acierto: por id y por título

Los métodos con LLM generan títulos y `calc.py` cuenta un acierto cuando el título generado coincide con el del ítem objetivo. Varios ítems del catálogo comparten título; en CDs, por ejemplo, hay discos distintos llamados "Greatest Hits". Si el objetivo tiene un título compartido, el LLM acierta con cualquiera de los ítems homónimos, mientras que un modelo de ids solo acierta con el id exacto.

| Dataset | Objetivos de test con título compartido |
|---|---|
| CDs | 13,7 % |
| Games | 2,4 % |
| Toys | 1,0 % |
| Instruments | 0,3 % |

(Medido fuera del notebook, sin output guardado.)

Por eso las métricas se calculan de dos formas:
- **Por id:** sobre los 10 ids con mayor puntaje.
- **Por título:** los ids se convierten a títulos, se quitan los repetidos manteniendo el orden y se toman los 10 primeros. Así queda una lista como la del beam search de los LLM, que nunca repite un título. Se compara con el título del objetivo.

Un acierto por id siempre lo es también por título, así que las métricas por título nunca son menores; `evaluar` (en `baseline_comun.py`) lo verifica con un `assert`.

### Pisos de referencia (MostPop)

MostPop (los 10 ítems más frecuentes en train), HR@10 / NDCG@10 en `test`:

| Dataset | HR@10 | NDCG@10 |
|---|---|---|
| CDs | 0.0197 | 0.0091 |
| Instruments | 0.0659 | 0.0330 |
| Toys | 0.0138 | 0.0082 |
| Games | 0.0208 | 0.0111 |

Fuente: constantes `PISOS` escritas en la celda 29 de `caser/baseline_caser.ipynb` (también en la celda 27 de `sasrec/baseline_sasrec.ipynb`), contra las que esas celdas comparan la variante BCE (paper) y todas dan OK. El output de esas celdas no imprime el cálculo de MostPop: estos valores se midieron fuera del notebook, sin output guardado. Un baseline por debajo de ellos indica un error.

`modelos_exploracion.ipynb` (celda 33) calcula otro MostPop que cuenta solo los objetivos de las filas de train (celda 29) y da valores distintos (HR@10 / NDCG@10): CDs 0.0192 / 0.0101, Instruments 0.0659 / 0.0348, Toys 0.0137 / 0.0083, Games 0.0208 / 0.0111. La diferencia viene de cómo se cuenta la frecuencia; la forma usada en los pisos no tiene output que la respalde.

### Datos

Tamaños de las particiones que escribe `gao` (log de la celda 9 de `datos.ipynb`). La ventana final sale del nombre de los CSV de `repo_latentr3/data/`.

| Categoría | Ventana final | Train | Valid | Test |
|---|---|---|---|---|
| Instruments | 2015-1 a 2018-11 | 66500 | 8312 | 8313 |
| Games | 2015-4 a 2018-11 | 75175 | 9397 | 9397 |
| CDs | 2015-10 a 2018-11 | 49251 | 6156 | 6157 |
| Toys | 2017-1 a 2018-11 | 53898 | 6737 | 6738 |

Son las cifras de la Tabla 3 del paper (App. A), salvo CDs test: el paper reporta 6158 y los CSV tienen 6157.

### Limitaciones comunes

- **Ítems ausentes de train** (4 a 57 por dataset): tienen embedding, pero solo aparecen como negativos, así que casi nunca se recomiendan. Son el objetivo en el 0,2–9,2 % de las filas de test (9,2 % en Games). Es una limitación de cualquier modelo basado en ids. Cifras medidas fuera del notebook, sin output guardado.
- **Una sola semilla**, como el paper. Con diferencias chicas entre configuraciones, la elección de la grilla puede cambiar con otra semilla. La reproducibilidad de cada modelo en GPU está en su README.
- **Protocolo de los autores desconocido:** no se sabe cómo entrenaron sus baselines, así que una diferencia con la Tabla 1 puede deberse al protocolo y no a la implementación.
- **Títulos compartidos:** el criterio por título favorece a los modelos que generan títulos (ver arriba); por eso se reportan ambos criterios.
- **Errores en la Tabla 1 del paper:** en Instruments, Base y COT reportan NDCG mayor que HR, lo que es imposible con un solo ítem relevante. No afecta a los baselines tradicionales, pero muestra que la tabla tiene errores de transcripción.
- El paper no informa varianza para los baselines; solo hay 5 corridas para BIGRec (Tabla 7).

## Uso común

- **Entorno:** `.venv-rocm` o `.venv-cuda` según la GPU (ver el [README de la raíz](../README.md)).
- **Directorio de trabajo:** cada notebook se ejecuta desde `notebooks/<carpeta>/` (lo normal en Jupyter y VS Code) y lee los datos desde `../../repo_latentr3/data`. Agrega `notebooks/` a `sys.path` para importar `baseline_comun.py`; los procesos de la grilla heredan ese `sys.path`.
- **Colab:** subir al directorio de trabajo el notebook, su `.py` y `baseline_comun.py`. Los datos se leen desde `data_amazon_2018/` y el código oficial desde su carpeta en el mismo directorio.
- **Fuentes externas:** el código oficial de cada modelo se clona en la raíz del repositorio (ignorado por git), con el commit fijado; ver [Fuentes externas](../README.md#fuentes-externas). Cada `.py` verifica el sha256 de los archivos que carga antes de importarlos y se detiene si no corresponden al commit.
- **Celdas:** en todos los README las celdas se numeran desde 0 (posición en el JSON del notebook).
- Los notebooks de los baselines tradicionales no escriben archivos. BIGRec escribe checkpoints y predicciones en `checkpoints/bigrec/`, y LatentR3 en `checkpoints/latentr3/` (ver sus README).
