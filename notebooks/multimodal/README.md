# Multimodal (`multimodal.ipynb`)

Primer paso de la versión multimodal del proyecto: medir qué ítems tienen imagen, representar las imágenes de Toys con un codificador visual preentrenado y comparar un baseline secuencial con y sin esas features. Es la base sobre la que se mediría el aporte de las imágenes a LatentR3.

**Estado:** `multimodal.ipynb` corrido en Colab por el compañero M (2026-11-25; resultados importados a `resultados/multimodal/` el 2026-11-27). Corre en Colab ([`../../colab/README.md`](../../colab/README.md)) o en local. **LatentR3-SFT + SigLIP** (`latentr3_mm.ipynb`, al final de este README): implementado y verificado en CPU el 2026-11-28; falta correrlo en la GPU.

## Fuente y versión

- **Imágenes:** campos `imageURLHighRes` e `imageURL` de la metadata de Amazon Reviews 2018 (`raw/meta_<categoría>.json.gz`, descargada por `../datos.ipynb`). sha256 de cada archivo en `items/fuente.json`.
- **Codificador:** SigLIP `google/siglip-so400m-patch14-384` (Zhai et al., ICCV 2023), en fp32, con `transformers==4.51.3`. La revisión de Hugging Face que se usó queda en `manifiesto_embeddings.json`. El tokenizer necesita `sentencepiece`.
- **Baseline:** SASRec del port de pmixer ([SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch), commit `9f6a02b`, sin modificar; sha256 verificado por `../sasrec/sasrec_baseline.py`), con el protocolo común de `../baseline_comun.py`.

## Datos

`extraer_items.py` (se corre una vez en local, donde está `raw/`) escribe `items/<categoría>.csv` con `item_id`, `asin`, título y las listas de URLs. El cruce asin → id sale de los CSV de `train`, `valid` y `test` (`item_asin`/`item_id`, `item_asins`/`history_item_id`) y cubre todos los ids de `info/`. Si un asin aparece repetido en la metadata, queda la última línea, como hace `gao()` con los títulos.

Ítems con al menos una URL de imagen en la metadata (2026-11-24; `items/fuente.json`):

| Dataset | Ítems | Con URL | Sin URL | Líneas de metadata | asin repetidos |
|---|---|---|---|---|---|
| CDs | 5841 | 2991 (51,2 %) | 2850 (48,8 %) | 516 914 | 755 |
| Instruments | 5030 | 3926 (78,1 %) | 1104 (21,9 %) | 120 310 | 819 |
| Toys | 6299 | 5698 (90,5 %) | 601 (9,5 %) | 633 883 | 137 |
| Games | 5308 | 4781 (90,1 %) | 527 (9,9 %) | 84 819 | 539 |

En los 4 datasets, los ítems con `imageURLHighRes` son exactamente los que tienen `imageURL`. Que la URL exista no significa que responda: eso lo mide el paso de cobertura, con fecha.

## Configuración

| Paso | Valor | Fuente |
|---|---|---|
| URL elegida | La primera de `imageURLHighRes`; si no hay, la primera de `imageURL` (el campo queda registrado) | decisión del proyecto |
| Cobertura | GET con cabecera de navegador genérica, 16 hilos, timeout de 10 s, 2 reintentos ante errores 5xx o de red. "Responde con imagen" = estado 200 y `Content-Type` `image/*`; solo se leen las cabeceras | decisión del proyecto |
| Descarga | La misma URL; la imagen se guarda solo si PIL la abre; manifiesto con sha256, tamaño y dimensiones | decisión del proyecto |
| Embeddings | Imagen: `get_image_features` sobre la imagen en RGB (las PNG con transparencia o en paleta pasan a RGB). Título: `get_text_features` con `padding="max_length"` y 64 tokens, como se entrenó SigLIP. Ambos normalizados a norma 1; 1152 dimensiones; fp32 | decisión del usuario (SigLIP so400m-384, 2026-11-24); uso del procesador según la ficha del modelo |
| Ítems sin imagen | Embedding de imagen en cero y `tiene_imagen = False` | decisión del proyecto |
| Representación de un ítem en SASRec | `e(i) = id(i) + W·f(i)`, con `f(i)` el embedding SigLIP congelado y `W` una proyección lineal 1152→50 entrenable (inicialización xavier, como el resto del port) | decisión del usuario (SASRec + features SigLIP, 2026-11-24) |
| Variantes | **id + imagen** (los ítems sin imagen suman un vector aprendido); **id + imagen (título de respaldo)** (sin imagen: embedding del título); **id + título** (control: separa el aporte de la imagen del de la proyección extra); **solo id** (SASRec de siempre, combinación ganadora lr 1e-4, wd 1e-6, vuelto a entrenar en la misma GPU) | decisión del proyecto |
| Entrenamiento | BCE (paper) con 1 negativo uniforme, Adam (0,9, 0,98), batch 128, d = 50, 2 bloques, dropout 0,5, ≤ 200 épocas, paciencia 10 por NDCG@10 por id en `valid`; grilla lr {1e-2, 1e-3, 1e-4} × wd {1e-4, 1e-5, 1e-6} por variante (28 combinaciones en total); semilla 42 | protocolo común (`../README.md`, D1–D17) |
| Métricas | HR y NDCG @5 y @10 por id y por título sobre todo el catálogo (`evaluar`), y separadas por popularidad del objetivo: ≤ 5 interacciones en `train` frente al resto | protocolo común; separación: decisión del proyecto |
| Interacciones en `train` | Usuarios distintos que tienen el ítem en algún historial u objetivo de `train` | decisión del proyecto |

## Adaptaciones respecto de la fuente

- `EmbItemMM` (`sasrec_mm.py`) reemplaza `modelo.item_emb` después de crear el modelo con `sasrec_baseline.nuevo_modelo`. Se comporta como `nn.Embedding` (llamada con índices, `weight` y `embedding_dim`), así que `log2feats` y `forward` del port, y `perdida_bce` y `puntajes` del baseline, funcionan sin cambios. La fila 0 (padding) vale siempre cero.
- `sasrec_baseline.entrenar` y `correr_combinacion` aceptan `crear_modelo` (por defecto el modelo de siempre) y `con_recs`. Con los valores por defecto, las métricas son idénticas bit a bit a las de antes del cambio (comprobado en CPU con Toys, BCE, lr 1e-3, wd 1e-5, 2 épocas, 2026-11-24).

## Limitaciones

- La cobertura vale para la fecha de la revisión: los enlaces de Amazon se caen con el tiempo.
- La cobertura solo lee las cabeceras: un archivo que dice ser imagen pero no lo es cuenta como "responde"; la descarga lo detecta con PIL.
- Solo Toys tiene embeddings y SASRec + SigLIP (decisión del usuario: cobertura en los 4 datasets, embeddings en Toys).
- "Solo id" se vuelve a entrenar solo con la combinación ganadora de la corrida local, no con la grilla completa.
- Las imágenes no se versionan; el manifiesto (URL + sha256) permite verificar una nueva descarga, pero no garantiza volver a obtenerla.

## Cómo correrlo

- **Colab (compañero M):** ver [`../../colab/README.md`](../../colab/README.md). Primera sesión con `PRUEBA = True` (100 ítems por paso, 1 combinación de 2 épocas, carpetas `*_prueba`); después, `PRUEBA = False`.
- **Local:** desde `notebooks/multimodal/`, con `.venv-rocm` o `.venv-cuda` más `sentencepiece` (`pip install sentencepiece`), `SASRec.pytorch/` en la raíz (ver el README de la raíz) y acceso a internet para la cobertura, la descarga y bajar SigLIP. Escribe en `checkpoints/multimodal/` (ignorado por git).
- **Retomar:** volver a correr todo. Cada paso se salta lo ya hecho: ítems revisados o descargados (CSV), embeddings (`siglip.npz`, verificado contra su manifiesto) y combinaciones de SASRec (un JSON por combinación).

Archivos que escribe, en `checkpoints/multimodal/` (Colab: `MyDrive/IIC3633/checkpoints/multimodal/`):
- `cobertura/cobertura_<categoría>.csv` (un ítem por fila: campo, URL, estado, tipo, bytes, error, fecha) y `cobertura/resumen.json`;
- `Toys_and_Games/manifiesto_imagenes.csv`, `imagenes.zip` (solo Colab), `siglip.npz` (`imagen`, `tiene_imagen`, `titulo`) y `manifiesto_embeddings.json`;
- `Toys_and_Games/sasrec_mm/<variante>__lr<lr>_wd<wd>.json` (una combinación) y `Toys_and_Games/sasrec_mm.json` (la mejor de cada variante, con popularidad).

## Costos

| Paso | Medido o estimado |
|---|---|
| `extraer_items.py`, 4 datasets | Medido en local: 34 s, 282 MB de RAM, 1 núcleo (2026-11-24) |
| SASRec + SigLIP, 2 épocas en CPU (1 hilo) | Medido en local: 15,8 s (prueba con features aleatorias, 2026-11-24) |
| Cobertura, descarga, embeddings y grilla en una T4 | Sin medir; estimaciones en `../../colab/README.md` |

## Resultados

Sin resultados propios: el notebook no se ha corrido. La referencia es el SASRec BCE (paper) de Toys de la corrida local (`../sasrec/baseline_sasrec.ipynb`, celda 25): HR@5 / HR@10 / NDCG@5 / NDCG@10 por id 0.0561 / 0.0772 / 0.0364 / 0.0432; por título 0.0564 / 0.0773 / 0.0368 / 0.0435 (lr 1e-4, wd 1e-6, época 138).

## Dónde quedan los resultados

Copias versionadas (las importa `colab/importar_resultados.py`): `resultados/multimodal/cobertura/` y `resultados/multimodal/Toys_and_Games/` (manifiestos y resultados de SASRec + SigLIP). No se versionan las imágenes, `imagenes.zip` ni `siglip.npz`.

## LatentR3-SFT + SigLIP (`latentr3_mm.ipynb`)

Extensión propia: no es parte del paper y se reporta aparte de la replicación. Es LatentR3 con una imagen por ítem del historial, en Toys. Es el mismo SFT y la misma evaluación de [`../latentr3/latentr3.ipynb`](../latentr3/README.md); el RL es opcional. El código está en `latentr3_mm.py`, que reutiliza `../latentr3/latentr3.py`.

### Método

- **Prompt:** el de LatentR3, con el marcador `<|image_pad|>` después de cada título del historial: `"título 1" <|image_pad|>, "título 2" <|image_pad|>, ...`.
- **Por qué ese token:** `<|image_pad|>` ya existe en el vocabulario de Qwen2.5 (id 151655; viene de Qwen2-VL y el modelo de texto no lo usa). No se agrega un token nuevo, porque el código de los autores reconoce el token latente como el último del vocabulario (`autores/model.py`, `generate_embs`; `LogitProcesser.py`, `get_prefix_data`).
- **Imágenes:** el embedding de cada marcador se reemplaza por `proyector(SigLIP(imagen))`, con un proyector lineal de 1152 a 1536. Los ítems sin imagen (9,5 % de Toys) usan un vector aprendido `sin_imagen`, compartido, así que lo único propio de cada ítem que entra por esos tokens es visual. Los embeddings son los de `siglip.npz`, congelados; sus filas coinciden con `item_id`.
- **Las dos pasadas:** el reemplazo se hace en las dos pasadas del modelo, así que LatentRATT calcula el token latente viendo las imágenes. Es el mismo mecanismo que los autores usan para `<|Thought|>` (sobrescribir la fila de un marcador) y el que usa Qwen2-VL con `<|image_pad|>` (`masked_scatter` en `transformers/models/qwen2_vl/modeling_qwen2_vl.py`).
- **Entrenamiento:** el SFT de LatentR3 sin cambios (lr 3e-4, batch 128, hasta 3 épocas, paciencia 1, F3, bf16), con el proyector y `sin_imagen` entrenados junto con el modelo.
- **Evaluación:** 10 beams restringidos por el trie, `length_penalty` 1,0, lotes de 12.
- **Ablación:** se evalúa también con todas las imágenes reemplazadas por `sin_imagen`, para saber si el modelo usa las imágenes.
- **RL (opcional, `CORRER_RL`):** el de LatentR3 con lr 1e-5; solo entrena LatentRATT y deja congelados el proyector y `sin_imagen`.

Inspiración (decisiones del proyecto del 2026-11-28):
- el token del ítem junto a su título viene del "hybrid prompt" de LLaRA ([arXiv 2312.02445](https://arxiv.org/abs/2312.02445));
- el proyector lineal, de LLaVA ([arXiv 2304.08485](https://arxiv.org/abs/2304.08485)). LLaVA-1.5 ([arXiv 2310.03744](https://arxiv.org/abs/2310.03744)) y Honeybee ([arXiv 2312.06742](https://arxiv.org/abs/2312.06742)) reportan que un MLP mejora poco;
- un token por imagen desde el embedding global, de ClipCap ([arXiv 2111.09734](https://arxiv.org/abs/2111.09734));
- la ablación, de NoteLLM-2 ([arXiv 2405.16789](https://arxiv.org/abs/2405.16789)), que reporta que los modelos multimodales ajustados de punta a punta tienden a ignorar lo visual.

### Cambios en `../latentr3/latentr3.py`

`entrenar_sft`, `evaluar_categoria` y `entrenar_rl` aceptan funciones alternativas para el tokenizer, el dataset, el modelo, el congelamiento y la clase del trainer. `generar` pasa `item_ids` a `generate` si los encodings los traen. Con los valores por defecto, el comportamiento es el de siempre: V5 y V6 dan lo mismo que antes (medido fuera del notebook el 2026-11-28, sin output guardado).

### Verificaciones (celda 5, solo CPU)

Medidas fuera del notebook el 2026-11-28, en CPU (sin output guardado; la celda 5 las repite):

| | Qué comprueba | Resultado |
|---|---|---|
| VM1 | `<|image_pad|>` = 151655, un solo token; `<|Thought|>` = 151665, el último; ningún título contiene un token especial; ningún prefijo del trie usa 151655 | pasa |
| VM2 | En 200 filas por split, sin los marcadores, el prompt es el de `LatentRDataset` y el de `D3Dataset`; marcadores = ítems del historial | pasa; ~11–15 tokens extra por fila |
| VM3 | Con el embedding del propio marcador en lugar de la imagen, los logits son los de `LatentModel`; el gradiente llega al proyector y a `sin_imagen` | dif. máx. 0.0 |
| VM4 | `generate` con 3 beams recibe `item_ids` una sola vez, con forma [B·3, 10]; guardar y cargar el modelo | forma (6, 10); dif. de logits 2,4e-7 |
| VM6 | El trainer del RL pasa los `item_ids` de cada grupo (8 filas iguales) en `_puntuar_grupo` y `compute_loss`; tras un paso solo cambia `attention.*` | pasa |

Largos de los splits completos con los marcadores (medidos fuera del notebook): media / p99 / máximo de 167,33 / 311 / 396 tokens en `train` (sin marcadores, la media es 155,92), 180,38 / 316 / 363 en `valid` y 175,45 / 299 / 370 en `test`. Ninguna fila llega a 512 (`CUTOFF_LEN`). Si alguna llegara, el dataset falla en vez de truncar.

### Cómo correrlo

1. Entorno `.venv-rocm` o `.venv-cuda`. Directorio de trabajo: `notebooks/multimodal/`. Hacen falta `checkpoints/multimodal/Toys_and_Games/siglip.npz` (se verifica su sha256), `repo_latentr3/` en el commit `c558b44` y Qwen2.5-1.5B en la caché de Hugging Face.
2. Correr las celdas 2–3 (entorno) y 5 (verificaciones, segundos).
3. Celda 7 (opcional): prueba rápida con 1024 filas, evaluación de 100 filas normal y sin imágenes, y 2 pasos de RL. Duración estimada de 15–20 min (sin medir). Escribe en `checkpoints/latentr3_mm/Toys_and_Games/_prueba/` y lo borra al final.
4. Celda 9: SFT, ~7,2 h estimadas (sin medir). Son las 6,73 h del SFT de texto escaladas por tokens por fila, 167,33 frente a 155,92. GPU ~10 GB, RAM ~11 GB.
5. Celdas 11 y 12: evaluación normal y ablación, ~40 min cada una (estimado).
6. Celda 14: métricas verificadas con `calc.py`, comparación pareada con el only SFT de texto, BIGRec y D3, y grupos por popularidad y por imagen en el historial. También copia los resultados a `resultados/`.
7. Celda 16 (opcional): RL con `CORRER_RL = True`, ~7,2 h más ~1,4 h de evaluación (estimado).

Retomar: volver a correr la celda. El SFT y el RL continúan desde su último checkpoint, y las evaluaciones se saltan las partes ya escritas.

### Dónde quedan los resultados

- **Local (no versionado):** `checkpoints/latentr3_mm/Toys_and_Games/`, con `sft/modelo/`, `sft/entrenamiento.json`, `sft/predicciones/test/` y `sft/predicciones/test_sin_imagenes/` (y, si se corre el RL, `rl_1e-5/`).
- **Versionado** (lo copia la celda 14): `resultados/multimodal/Toys_and_Games/latentr3_mm/`, con `entrenamiento.json`, `predicciones/{test,test_sin_imagenes}/` (`final_result.json` y `registro.json`) y `analisis.json`.

