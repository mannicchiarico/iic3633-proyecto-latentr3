# Notebooks

## GRU4Rec (`baseline_gru4rec.ipynb`)

Baseline GRU4Rec con la implementación oficial en PyTorch ([hidasib/GRU4Rec_PyTorch_Official](https://github.com/hidasib/GRU4Rec_PyTorch_Official)), entrenada con el protocolo de LatentR3. Los autores no publican el código de sus baselines.

### Lo que fija el paper

- **Entrenamiento (Apéndice B):** *"we use the Adam optimizer and perform grid searches over learning rates 1e-2, 1e-3, 1e-4 and weight decay values 1e-4, 1e-5, 1e-6, and all models are trained using Binary Cross-Entropy loss with randomly sampled negative items."*
- **Contexto (Apéndice A):** largo máximo de secuencia de 10.
- **Evaluación (sección 4.1):** HR y NDCG @5 y @10.

### Lo que el paper no especifica

| Aspecto | Valor usado | Fuente |
|---|---|---|
| Capa oculta, batch, dropout | 100, 64, sin dropout | Valores por defecto del GRU4Rec oficial |
| Embeddings | Compartidos entre entrada y salida (`constrained_embedding=True`) | Valor por defecto oficial |
| Negativos por ejemplo | 1, uniforme | Decisión del proyecto (como SASRec) |
| Épocas y criterio de parada | Hasta 100, early stopping por NDCG@10 en `valid` con paciencia 10, mejor época | Protocolo común del proyecto para SASRec, GRU4Rec y Caser |
| Elección de la configuración | Mejor NDCG@10 en `valid` por dataset | Decisión del proyecto |
| Ranking | Todo el catálogo de `info/`, desempate por índice, sin excluir el historial | Protocolo común: mismos candidatos que los métodos con LLM; no excluir el historial sigue a `calc.py` de los autores |

### Adaptaciones al código oficial

Los archivos oficiales no se modifican: el notebook importa `GRU4Rec`, `GRU4RecModel`, `SessionDataIterator` y `SampleCache`, y la subclase `GRU4RecPaper` sobrescribe la pérdida y el ciclo de entrenamiento.

| Aspecto | Oficial | Aquí | Por qué |
|---|---|---|---|
| Pérdida | Cross-entropy o BPR-max | BCE (`bce_loss`) | Lo pide el paper |
| Catálogo | Ítems de train; en evaluación descarta los eventos con ítems desconocidos | `itemidmap` con todos los ítems de `info/` | Mismos candidatos que los métodos con LLM; ninguna fila de test se descarta |
| Negativos | 2048 por popularidad, compartidos en el lote | Uno uniforme por fila sobre todo el catálogo (`SampleCache` con distribución uniforme, `n_sample = batch_size`) | "Randomly sampled" del paper; el muestreo por popularidad del iterador no admite ítems ausentes de train |
| Optimizador | Adagrad disperso, sin weight decay | `torch.optim.Adam(lr, weight_decay)` con embeddings densos | Lo pide el paper; Adam no acepta gradientes dispersos |
| Épocas | 10 fijas, se queda con la última | Early stopping en `valid` | Las 10 épocas oficiales están pensadas para Adagrad con lr altas y datasets grandes |
| Búsqueda | Optuna | Grilla de lr × weight decay | Lo pide el paper |
| Contexto | Sesiones completas | Bloques de hasta 11 ítems por usuario, cortados desde el final | Respeta el máximo de 10 sin duplicar transiciones |
| Inicialización | `compatibility_mode` (numpy) | Inicialización en torch de `GRU4RecModel` | `compatibility_mode` falla con numpy 2 |
| Evaluación | Recall y MRR @20 (`batch_eval`) | `recomendar`: top-10 sobre todo el catálogo desde el historial ≤10 de cada fila | Métricas del paper |

**Pérdida y negativos.** En entrenamiento, los puntajes `O` tienen forma `(filas, filas + n_sample)`. Los positivos están en la diagonal del primer bloque, y la fila *i* usa como negativo la columna *i* del bloque muestreado. `bce_loss` devuelve `−Σ [log σ(positivo) + log σ(−negativo)]`, y el ciclo la divide por `batch_size`, como en las pérdidas oficiales.

**Adam.** El `weight_decay` de `torch.optim.Adam` es regularización L2 sumada al gradiente, que es la lectura literal del paper. No es el weight decay desacoplado de AdamW.

**Contexto.** Cada fila de `train` es un prefijo de la secuencia de un usuario, así que la secuencia completa se reconstruye con el historial de la primera fila más todos los `item_id`. `a_sesiones` la corta en bloques de hasta 11 ítems desde el final. Cada bloque empieza en el último ítem del anterior, y el estado oculto vuelve a cero al inicio de cada bloque. Así, cada transición se entrena una vez por época, ninguna predicción usa más de 10 ítems previos y las más recientes tienen el contexto completo. `Time` es la posición dentro del bloque, porque los timestamps diarios tienen empates.

**Ciclo de entrenamiento.** `GRU4RecPaper.fit` es el ciclo oficial con estos cambios: Adam con embeddings densos, evaluación de NDCG@10 en `valid` al final de cada época con early stopping, y sin `compatibility_mode`, `logq`, registro de la pérdida ni chequeo de NaN. El iterador se crea sin negativos (`n_sample=0`) y con el `itemidmap` del catálogo, y después se le asigna un `SampleCache` oficial con distribución uniforme. El resto se mantiene igual: `SessionDataIterator`, `GRU4RecModel`, `_adjust_hidden` y `detach_` en cada paso.

### Limitaciones

- **Contexto al inicio de cada bloque:** las primeras predicciones de cada bloque ven menos ítems que la ventana de 10 del paper. El contexto medio en entrenamiento queda entre 3,5 y 4,1 ítems, frente a 3,8–5,1 de las filas del paper.
- **Retropropagación:** truncada a un paso (`detach_`), como en el oficial.
- **Ítems ausentes de train:** tienen embedding, pero solo se entrenan como negativos, así que casi nunca entran al top-10. Son pocos (4 a 57 por dataset), pero aparecen como objetivo en el 0,2–9,2 % de test, filas que en la práctica cuentan como fallo. Cualquier modelo por IDs entrenado con `train` tiene esta limitación.
- **Negativos:** el muestreador oficial no excluye el positivo (~0,02 % de probabilidad con unos 5.000 ítems).
- **Entrenamiento final:** solo con `train`, sin volver a entrenar con train + validación.
- **Una sola semilla.**

### Uso

- El código oficial debe estar en `../GRU4Rec_PyTorch_Official` en local, o en `GRU4Rec_PyTorch_Official/` dentro del directorio de trabajo en Colab. Esa carpeta está en `.gitignore`, así que hay que descargarla aparte. La licencia permite el uso libre en investigación y docencia.
- El notebook corre las 36 combinaciones (4 datasets × 9) en secuencia, fijando las semillas antes de cada una. Imprime una línea por combinación y al final la tabla con la mejor por dataset, evaluada en `test`. No escribe archivos. Tarda unos 30 minutos en una RX 9070 XT.