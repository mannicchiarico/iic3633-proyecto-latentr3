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

## Cómo correr los notebooks

### 1. `notebooks/datos.ipynb` (opcional)

Debe ejecutarse con la **raíz del repositorio** como directorio de trabajo, porque las rutas se construyen desde `os.path.abspath(".")`:

```bash
jupyter notebook --notebook-dir=. notebooks/datos.ipynb
```

Qué hace:

1. Clona el repositorio de LatentR3 en `repo_latentr3/` si no existe e importa `gao` desde `process.py`.
2. Descarga las reseñas y la metadata de las cuatro categorías en `raw/` (varios GB; se omiten los archivos que ya existen).
3. Filtra las reseñas desde 2010 y llama a `gao` para cada categoría, con la zona horaria fijada en UTC.

Output esperado: logs de `gao` con el número de usuarios, ítems, reseñas y la densidad de cada categoría, y los archivos `info/*.txt` y `train|valid|test/*.csv` escritos en `repo_latentr3/data/`.

> **Nota:** `gao` se llama con su ventana de tiempo por defecto (2017-10 a 2018-11), así que los archivos generados llevan el sufijo `_5_2017-10-2018-11`. Los archivos incluidos en el repositorio usan otras ventanas por categoría (por ejemplo, `Video_Games_5_2015-4-2018-11`). Si se generan archivos nuevos, conviene dejar un solo archivo por categoría en cada carpeta, porque `modelos_exploracion.ipynb` carga todos los archivos que encuentra.

### 2. `notebooks/modelos_exploracion.ipynb`

Funciona tanto en local como en Google Colab:

- **Local:** se ejecuta con `notebooks/` como directorio de trabajo, que es el comportamiento por defecto de Jupyter y de VS Code. Lee los datos desde `../repo_latentr3/data`. Las celdas de Google Drive y del zip no hacen nada.
- **Colab:** monta Google Drive, descomprime `IIC3633/data_amazon_2018.zip` en `/content/` y lee los datos desde `data_amazon_2018/`.

> **Nota:** Esta documentación fue realizada por Claude. [Ver prompt y respuesta](https://claude.ai/artifact/Mskz7UPkqXJ9g2GdCSh9pk).