# iic3633-proyecto-latentr3
Repositorio en donde se suben los notebooks de jupyter, el código original y los resultados del proyecto semestral del curso de Sistemas Recomendadores.

El proyecto estudia [LatentR3](https://github.com/xuwenxinedu/R3) (*Reinforced Latent Reasoning*) para recomendación secuencial sobre cuatro categorías de Amazon Reviews 2018: `CDs_and_Vinyl`, `Musical_Instruments`, `Toys_and_Games` y `Video_Games`.

## Estructura

```plaintext
.
├── docs/            Paper de LatentR3, enunciado del proyecto y propuesta (H1)
├── notebooks/
│   ├── datos.ipynb                 Descarga y preprocesamiento de los datos
│   └── modelos_exploracion.ipynb   Análisis exploratorio y baselines
├── raw/             Archivos .json.gz de Amazon 2018 (no versionado, lo genera datos.ipynb)
└── repo_latentr3/   Código original de los autores de LatentR3
    ├── data/
    │   ├── process.py   Preprocesamiento (función gao)
    │   ├── info/        Mapeo título → item_id por categoría (.txt)
    │   ├── train/       Splits de entrenamiento (.csv)
    │   ├── valid/       Splits de validación (.csv)
    │   └── test/        Splits de prueba (.csv)
    ├── scripts/     Scripts de entrenamiento y evaluación (SFT y GRPO)
    ├── src/         Código del modelo
    └── environment.yaml
```

Los datos ya procesados están incluidos en `repo_latentr3/data/`, por lo que no es necesario ejecutar `datos.ipynb` para correr `modelos_exploracion.ipynb`.

## Requisitos

Para los notebooks basta un entorno con Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy matplotlib seaborn torch fire loguru tqdm
```

Para entrenar LatentR3 se usa el entorno de los autores, que requiere una GPU NVIDIA con CUDA 12.1:

```bash
conda env create -n your_env -f repo_latentr3/environment.yaml
```

### Entorno para GPU AMD Radeon (ROCm)

Alternativa al entorno de los autores, probada con una AMD Radeon RX 9070 XT (gfx1201, 16 GB) en Ubuntu 24.04. Usa PyTorch para ROCm, que mantiene la API `torch.cuda`, por lo que el código de LatentR3 corre sin cambios:

```bash
python3 -m venv .venv-rocm
source .venv-rocm/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14
pip install transformers==4.51.3 trl==0.16.1 accelerate==1.6.0 datasets fire loguru tqdm \
            pandas numpy matplotlib seaborn jupyter ipykernel
```

Las versiones de `transformers`, `trl` y `accelerate` son las del `environment.yaml`. No se instalan `flash-attn`, `vllm`, `xformers` ni `bitsandbytes`, que son exclusivas de CUDA y el código no las usa con Qwen. Los wheels de PyTorch incluyen el runtime de ROCm, así que no hace falta instalar ROCm en el sistema.

Para verificar que se detecta la GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Debe mostrar `2.14.1+rocm7.14 AMD Radeon RX 9070 XT`. El aviso `rocSHMEM Could not open libnuma` no afecta con una sola GPU; se elimina con `sudo apt install libnuma1`.

ROCm respeta `CUDA_VISIBLE_DEVICES`. Si el equipo tiene además una GPU integrada, la dedicada es el índice `0`. No se debe combinar `HIP_VISIBLE_DEVICES` con `CUDA_VISIBLE_DEVICES`, porque el proceso se cae.

## Cómo correr los notebooks

### 1. `notebooks/datos.ipynb` (opcional)

Puede ejecutarse desde `notebooks/` (por defecto en Jupyter y VS Code) o desde la raíz del repositorio. Si se vuelve a ejecutar, conviene reiniciar el kernel antes, porque la última celda cambia el directorio de trabajo a `repo_latentr3/`.

Qué hace:

1. Clona el repositorio de LatentR3 en `repo_latentr3/` si no existe e importa `gao` desde `process.py`.
2. Descarga las reseñas y la metadata de las cuatro categorías en `raw/` (varios GB; se omiten los archivos que ya existen).
3. Filtra las reseñas desde 2010 y llama a `gao` para cada categoría, con la zona horaria fijada en UTC. `gao` parte con la ventana 2017-10 a 2018-11 y retrocede el inicio de a 3 meses hasta tener al menos 5000 ítems.

Output esperado: logs de `gao` con el número de usuarios, ítems, reseñas y la densidad de cada categoría, y los archivos `info/*.txt` y `train|valid|test/*.csv` escritos en `repo_latentr3/data/`. Los nombres de archivo llevan la ventana final de cada categoría (por ejemplo, `Video_Games_5_2015-4-2018-11.csv`), así que coinciden con los ya incluidos y se sobrescriben.

### 2. `notebooks/modelos_exploracion.ipynb`

Funciona tanto en local como en Google Colab:

- **Local:** se ejecuta con `notebooks/` como directorio de trabajo, que es el comportamiento por defecto de Jupyter y de VS Code. Lee los datos desde `../repo_latentr3/data`. Las celdas de Google Drive y del zip no hacen nada.
- **Colab:** monta Google Drive, descomprime `IIC3633/data_amazon_2018.zip` en `/content/` y lee los datos desde `data_amazon_2018/`.

## Entrenamiento de LatentR3

### Modelo base

El paper usa `Qwen/Qwen2.5-1.5B` (versión base, no Instruct). Se descarga una vez (~3,1 GB) a la caché de Hugging Face (`~/.cache/huggingface/hub/`):

```bash
hf download Qwen/Qwen2.5-1.5B
```

Si la descarga de `model.safetensors` falla con `No CA certificates were loaded from the system`, la variable `SSL_CERT_DIR` apunta a una ruta inválida. Se corrige con `export SSL_CERT_DIR=/etc/ssl/certs` antes de descargar.

### Cambios en los scripts

Los scripts de `repo_latentr3/scripts/` venían configurados para 2 GPUs NVIDIA de los autores. Se adaptaron a una sola GPU (índice `0`):

| Script | Cambio |
|---|---|
| `latent_train.sh` (SFT) | `CUDA_VISIBLE_DEVICES=0`, `BASE_MODEL=Qwen/Qwen2.5-1.5B`, `--nproc_per_node=1`, `array=(0)`, `nn=1` |
| `attention_grpo.sh` (GRPO) | `CUDA_VISIBLE_DEVICES=0`, `--nproc_per_node=1`, `array=(0)`, `nn=1` |
| `grpo_eval.sh` (evaluación) | `array=(0)`, `nn=1` |

Usar una GPU en vez de dos es una desviación respecto del paper: los autores advierten que la ventaja por batch del GRPO depende de usar 2 GPUs.

### Ejecución

Los scripts usan rutas relativas, así que se ejecutan desde `repo_latentr3/` con el entorno activado:

```bash
cd repo_latentr3
source ../.venv-rocm/bin/activate
bash scripts/latent_train.sh
```

Los checkpoints y resultados quedan en `repo_latentr3/output_dir/<categoría>/`.

### Problemas conocidos

- `src/grpo_attention_tuning/model.py` importa `noisy_layer`, que no existe en el repositorio. Por eso `train_noise_grpo.py` y `noise_eval.py` fallan con `ModuleNotFoundError`. La función importada solo se usa en código comentado.
- `grpo_eval.sh` llama a `src/grpo_full/noise_eval.py`, que no existe; la ruta correcta es `src/grpo_attention_tuning/noise_eval.py`.
- `attention_grpo.sh` no calcula las métricas al final; hay que ejecutar `src/utils/calc.py` sobre el `final_result.json`.

> **Nota:** Esta documentación fue realizada por Claude, aunque se realizaron correcciones. [Ver prompt y respuesta original](https://claude.ai/artifact/Mskz7UPkqXJ9g2GdCSh9pk). Las secciones del entorno para GPU AMD y del entrenamiento de LatentR3 salen de [esta conversación](https://claude.ai/artifact/2ruQ4c1UMsBqamCmVii6wY).