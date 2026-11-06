# iic3633-proyecto-latentr3

Replicación de [LatentR3](https://github.com/xuwenxinedu/R3) (*Reinforced Latent Reasoning*) para recomendación secuencial, con sus baselines, sobre cuatro categorías de Amazon Reviews 2018: `CDs_and_Vinyl`, `Musical_Instruments`, `Toys_and_Games` y `Video_Games`.

El repositorio contiene los notebooks, el código de los autores, el código adaptado y los resultados del proyecto semestral del curso de Sistemas Recomendadores.

## Estado

| Modelo | Estado | README |
|---|---|---|
| GRU4Rec | Corrido en las 4 categorías, salvo CE todas las posiciones | [notebooks/gru4rec/README.md](notebooks/gru4rec/README.md) |
| SASRec | Corrido en las 4 categorías | [notebooks/sasrec/README.md](notebooks/sasrec/README.md) |
| Caser | Corrido en las 4 categorías | [notebooks/caser/README.md](notebooks/caser/README.md) |
| BIGRec | Solo Toys_and_Games | [notebooks/bigrec/README.md](notebooks/bigrec/README.md) |
| D3 | Solo Toys_and_Games | [notebooks/d3/README.md](notebooks/d3/README.md) |
| Base | Corrido en las 4 categorías | [notebooks/base_cot/README.md](notebooks/base_cot/README.md) |
| COT | Solo Toys_and_Games | [notebooks/base_cot/README.md](notebooks/base_cot/README.md) |
| LatentR3 | Implementado, sin entrenamiento completo; solo hay una prueba de tiempo en Toys_and_Games | [notebooks/latentr3/README.md](notebooks/latentr3/README.md) |

Los protocolos y decisiones comunes de los baselines tradicionales están en [notebooks/README.md](notebooks/README.md).

## Estructura

```plaintext
.
├── docs/
│   └── 12149_Reinforced_Latent_Reason.pdf   Paper de LatentR3
├── notebooks/
│   ├── README.md                   Índice de baselines y protocolo común
│   ├── baseline_comun.py           Datos, métricas y semillas comunes a los baselines
│   ├── datos.ipynb                 Descarga y preprocesamiento de los datos
│   ├── modelos_exploracion.ipynb   Análisis exploratorio (Random, MostPop, LastItem)
│   ├── gru4rec/, sasrec/, caser/, bigrec/, d3/   Un baseline por carpeta: notebook, .py y README
│   ├── base_cot/                   Baselines Base y COT (Qwen2.5-1.5B sin ajustar)
│   └── latentr3/                   LatentR3 completo (SFT + LR-GRPO) en 1 GPU de 16 GB
│       └── autores/                Copia parcheada del código de los autores
├── resultados/      Copias versionadas de los JSON de métricas, tiempos y predicciones finales
├── checkpoints/     Local, no versionada: pesos y salidas completas
├── raw/             No versionada: archivos .json.gz de Amazon 2018 (los genera datos.ipynb)
├── repo_latentr3/   Código original de los autores de LatentR3
│   ├── data/        Datos procesados: info/ (título → item_id), train/, valid/, test/
│   ├── scripts/     Scripts de entrenamiento y evaluación (SFT y GRPO)
│   ├── src/         Código del modelo
│   └── environment.yaml
└── GRU4Rec_PyTorch_Official/, SASRec.pytorch/, caser_pytorch/, DecodingMatters/
                     Fuentes externas: no versionadas, se clonan (ver abajo)
```

## Fuentes externas

Se clonan en la raíz del repositorio y no se versionan. Los notebooks comprueban el sha256 de los archivos que cargan de ellas.

| Carpeta | Repo | Commit | Usada por |
|---|---|---|---|
| `GRU4Rec_PyTorch_Official` | https://github.com/hidasib/GRU4Rec_PyTorch_Official | `d1fc31105577665d2f105de1c8d9a38d223a9a22` | gru4rec |
| `SASRec.pytorch` | https://github.com/pmixer/SASRec.pytorch | `9f6a02b86a95e616fe815a0edc92dafa9911773d` | sasrec |
| `caser_pytorch` | https://github.com/graytowne/caser_pytorch | `c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1` | caser |
| `DecodingMatters` | https://github.com/SAI990323/DecodingMatters | `585c8427dd327a33155b5566f32bc22e21b5236c` | bigrec, d3, base_cot, latentr3 |

```bash
git clone https://github.com/hidasib/GRU4Rec_PyTorch_Official
git -C GRU4Rec_PyTorch_Official checkout d1fc31105577665d2f105de1c8d9a38d223a9a22
git clone https://github.com/pmixer/SASRec.pytorch
git -C SASRec.pytorch checkout 9f6a02b86a95e616fe815a0edc92dafa9911773d
git clone https://github.com/graytowne/caser_pytorch
git -C caser_pytorch checkout c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1
git clone https://github.com/SAI990323/DecodingMatters
git -C DecodingMatters checkout 585c8427dd327a33155b5566f32bc22e21b5236c
```

## Requisitos

Los notebooks y baselines necesitan Python 3.12 y las librerías de `requirements.txt`. `torch` no está en ese archivo a propósito: se instala aparte desde el índice que corresponde a la GPU (ROCm, CUDA o CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch    # o con el índice ROCm/CUDA de abajo
pip install -r requirements.txt
```

### Entorno para entrenar LatentR3 (GPU AMD o NVIDIA)

El código de LatentR3 usa la API `torch.cuda`, que PyTorch mantiene igual en NVIDIA (CUDA) y en AMD (ROCm), así que corre sin cambios en ambas. Solo cambia de dónde se instala `torch`; el resto de las librerías está en `requirements.txt`:

```bash
# AMD Radeon (ROCm)
python3 -m venv .venv-rocm
source .venv-rocm/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14

# NVIDIA (CUDA): usar el índice cuXXX que indique https://pytorch.org/get-started/locally/
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX

# Ambos
pip install -r requirements.txt
```

Las versiones de `transformers`, `trl` y `accelerate` son las del `environment.yaml` de los autores. No se instalan `flash-attn`, `vllm`, `xformers` ni `bitsandbytes`: el código no las usa con Qwen. Los wheels de PyTorch incluyen el runtime de ROCm o CUDA, así que no hace falta instalarlo en el sistema.

Para verificar que se detecta la GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Probado con una AMD Radeon RX 9070 XT (gfx1201, 16 GB) en Ubuntu 24.04, donde muestra `2.14.1+rocm7.14 AMD Radeon RX 9070 XT`.

- **NVIDIA:** el entrenamiento usa bf16, que requiere una GPU Ampere o posterior (RTX 30xx, RTX 40xx, A100, etc.). Una T4 o una RTX 20xx no sirve sin cambiar a fp16. Como alternativa más fiel al paper, se puede usar el entorno original de los autores (CUDA 12.1, `torch 2.6`): `conda env create -n your_env -f repo_latentr3/environment.yaml`.
- **AMD:** se requiere ROCm 6.4 o superior para las RX 9070. El aviso `rocSHMEM Could not open libnuma` no afecta con una sola GPU; se elimina con `sudo apt install libnuma1`. Si el equipo tiene además una GPU integrada, la dedicada es el índice `0`. No se debe combinar `HIP_VISIBLE_DEVICES` con `CUDA_VISIBLE_DEVICES`, porque el proceso se cae.

## Datos

Los datos ya procesados están en `repo_latentr3/data/`, por lo que `notebooks/datos.ipynb` es opcional y no hace falta ejecutarlo para correr los demás notebooks.

`datos.ipynb` puede ejecutarse desde `notebooks/` (por defecto en Jupyter y VS Code) o desde la raíz del repositorio. Si se vuelve a ejecutar, conviene reiniciar el kernel antes, porque la última celda cambia el directorio de trabajo a `repo_latentr3/`.

Qué hace:

1. Clona el repositorio de LatentR3 en `repo_latentr3/` si no existe e importa `gao` desde `process.py`.
2. Descarga las reseñas y la metadata de las cuatro categorías en `raw/` (varios GB; se omiten los archivos que ya existen).
3. Filtra las reseñas desde 2010 y llama a `gao` para cada categoría, con la zona horaria fijada en UTC. `gao` parte con la ventana 2017-10 a 2018-11 y retrocede el inicio de a 3 meses hasta tener al menos 5000 ítems.

Output esperado: logs de `gao` con el número de usuarios, ítems, reseñas y la densidad de cada categoría, y los archivos `info/*.txt` y `train|valid|test/*.csv` escritos en `repo_latentr3/data/`. Los nombres de archivo llevan la ventana final de cada categoría (por ejemplo, `Video_Games_5_2015-4-2018-11.csv`), así que coinciden con los ya incluidos y se sobrescriben.

## Cómo correr los notebooks

### `notebooks/modelos_exploracion.ipynb`

Análisis exploratorio de los datos con los modelos Random, MostPop y LastItem. Los baselines están en sus propias carpetas. Funciona tanto en local como en Google Colab:

- **Local:** se ejecuta con `notebooks/` como directorio de trabajo, que es el comportamiento por defecto de Jupyter y de VS Code. Lee los datos desde `../repo_latentr3/data`. Las celdas de Google Drive y del zip no hacen nada.
- **Colab:** monta Google Drive, descomprime `IIC3633/data_amazon_2018.zip` en `/content/` y lee los datos desde `data_amazon_2018/`.

### Baselines y LatentR3

Cada baseline se corre desde su carpeta (`notebooks/<modelo>/`), siguiendo la sección "Cómo correrlo" de su README (ver la tabla de [Estado](#estado)). Los que usan código externo requieren clonar antes las [fuentes externas](#fuentes-externas).

## Entrenamiento de LatentR3 con los scripts de los autores

### Modelo base

El paper usa `Qwen/Qwen2.5-1.5B` (versión base, no Instruct). Se descarga una vez (~3,1 GB) a la caché de Hugging Face (`~/.cache/huggingface/hub/`):

```bash
hf download Qwen/Qwen2.5-1.5B
```

Si la descarga de `model.safetensors` falla con `No CA certificates were loaded from the system`, la variable `SSL_CERT_DIR` apunta a una ruta inválida. Se corrige con `export SSL_CERT_DIR=/etc/ssl/certs` antes de descargar.

### Cambios en los scripts

Los scripts de `repo_latentr3/scripts/` venían con las GPUs de los autores fijas en el código (por ejemplo, `CUDA_VISIBLE_DEVICES=6,7`). Ahora las GPUs se eligen con la variable `GPUS`, que por defecto es `0`:

| Script | Cambio |
|---|---|
| `latent_train.sh` (SFT) | `GPUS` define `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, `array` y `nn`; `BASE_MODEL=Qwen/Qwen2.5-1.5B` |
| `attention_grpo.sh` (GRPO) | `GPUS` define `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, `array` y `nn` |
| `grpo_eval.sh` (evaluación) | `GPUS` define `array` y `nn` |

El número de procesos de entrenamiento y de particiones de la evaluación es la cantidad de GPUs en `GPUS`. Usar una GPU en vez de dos es una desviación respecto del paper: los autores advierten que la ventaja por batch del GRPO depende de usar 2 GPUs.

### Ejecución

Los scripts usan rutas relativas, así que se ejecutan desde `repo_latentr3/` con el entorno activado:

```bash
cd repo_latentr3
source ../.venv-rocm/bin/activate    # o ../.venv-cuda
bash scripts/latent_train.sh         # una GPU (índice 0)
GPUS=0,1 bash scripts/latent_train.sh  # dos GPUs, como en el paper
```

Los checkpoints y resultados se escribirán en `repo_latentr3/output_dir/<categoría>/`. Esa ruta aún no existe: la crean los scripts al ejecutarse.

## Problemas conocidos

- `src/grpo_attention_tuning/model.py` importa `noisy_layer`, que no existe en el repositorio. Por eso `train_noise_grpo.py` y `noise_eval.py` fallan con `ModuleNotFoundError`. La función importada solo se usa en código comentado.
- `grpo_eval.sh` llama a `src/grpo_full/noise_eval.py`, que no existe; la ruta correcta es `src/grpo_attention_tuning/noise_eval.py`.
- `attention_grpo.sh` no calcula las métricas al final; hay que ejecutar `src/utils/calc.py` sobre el `final_result.json`.
- **En una GPU de 16 GB** el SFT no cabe desde el segundo paso: sin los estados de AdamW el primer paso ya llega a 11,70 GB, y al crearlos se agota la memoria (14,13 GiB asignados de 15,92 GiB; `resultados/latentr3/_tiempo/sft2.json` y `tiempos.json`), y el RL con 1 GPU falla en `swap_adjacent_blocks`.
- **En ROCm 7.14 (gfx1201)**, hipBLASLt tiene un defecto en las multiplicaciones por lotes con k = 1 que mata el proceso ("Memory access fault"); esa operación aparece en el backward de LatentRATT.
- [notebooks/latentr3/](notebooks/latentr3/README.md) resuelve lo anterior con una copia parcheada del código (`notebooks/latentr3/autores/`) y AdamW con los estados en la RAM, sin cambiar el método. El detalle está en su README.


NOTA: Se utilizó apoyo de Claude Code para la documentación, entendimiento de las particularidades del código oficial de las distintos métodos implementados. También apoyo en la sugerencia y apliación de distintas alternativas para compatibilizar el código con las especificaciones del computador utilizado y aprovechar recursos del computador para reducir los tiempos de ejecución, pero fueron exigidas explicaciones y las decisiones de proyecto tomadas fueron conversadas como grupo. 