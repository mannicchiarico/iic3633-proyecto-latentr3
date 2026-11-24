# Multimodal (`multimodal.ipynb`)

Primer paso de la versión multimodal del proyecto: medir qué ítems tienen imagen, representar las imágenes de Toys con un codificador visual preentrenado y comparar un baseline secuencial con y sin esas features. Es la base sobre la que se mediría el aporte de las imágenes a LatentR3.

**Estado:** implementado y sin correr. Solo está hecha la extracción de las URLs desde la metadata (`items/`, ver "Datos"). Corre en Colab (compañero M, ver [`../../colab/README.md`](../../colab/README.md)) o en local.

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
