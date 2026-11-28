# Colab: guía para correr los experimentos asignados

Dos tareas que corren en Colab gratis, con una GPU T4, mientras la GPU local entrena los modelos:

| Quién | Tarea | Notebook |
|---|---|---|
| **Compañero E** | Evaluar BIGRec y D3 en Video_Games, Musical_Instruments y CDs_and_Vinyl, con modelos ya entrenados | [`colab/evaluar_bigrec_t4.ipynb`](evaluar_bigrec_t4.ipynb) |
| **Compañero M** | Cobertura de imágenes (4 datasets), embeddings SigLIP de Toys y el baseline SASRec + SigLIP | [`notebooks/multimodal/multimodal.ipynb`](../notebooks/multimodal/multimodal.ipynb) |

No hace falta instalar nada en tu computador ni clonar el repo: el notebook lo hace solo en Colab. Todo lo que importa queda en tu Google Drive, en `Mi unidad/IIC3633/`, así que si Colab se desconecta no se pierde lo ya terminado.

## Antes de empezar (una vez)

1. Abre el notebook desde GitHub. En Colab: **Archivo > Abrir cuaderno > GitHub**, pega `https://github.com/mannicchiarico/iic3633-proyecto-latentr3` y elige el notebook de tu tarea. Ábrelo siempre desde ahí, no desde una copia vieja.
2. **Solo E:** quien coordina te comparte una carpeta de Drive llamada `bigrec_compartido`, con una subcarpeta por dataset (`Toys_and_Games`, `Video_Games`, `Musical_Instruments` y, cuando se suba, `CDs_and_Vinyl`), cada una con `modelo/` y `entrenamiento.json`. En Drive, ve a **Compartido conmigo**, haz clic derecho en `bigrec_compartido` > **Organizar > Agregar acceso directo** y créalo dentro de `Mi unidad/IIC3633/checkpoints/`. Después renombra el acceso directo a `bigrec`. El notebook espera encontrar `Mi unidad/IIC3633/checkpoints/bigrec/Video_Games/modelo/`, etc.
   - *Supuesto:* Colab ve los accesos directos como carpetas normales. Si la tabla de estado del notebook dice `modelo listo: False` para un modelo que ya está subido, haz una copia de esa subcarpeta en `Mi unidad/IIC3633/checkpoints/bigrec/<dataset>/`, de a un dataset a la vez (cada modelo pesa 3,1 GB y Drive gratis tiene 15 GB).

## En cada sesión

1. **Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU.** Colab puede asignar otra máquina al reconectar, así que hay que revisarlo en cada sesión. Si dice que no hay GPU disponible, prueba más tarde: **no corras en CPU**. El notebook de E se detiene solo si no hay GPU. (M puede hacer la cobertura y la descarga sin GPU; ver más abajo.)
2. **Entorno de ejecución > Ejecutar todas.** La primera celda prepara la sesión: clona el repo en el commit fijado, instala las versiones correctas, monta Drive (acepta el permiso) y revisa la GPU. Debe imprimir una línea como `[preparación] GPU Tesla T4 | 14.7 GiB | capacidad [7, 5] | ...`.
3. Deja la pestaña abierta y el computador sin suspenderse. *Supuesto:* Colab gratis corta las sesiones inactivas y limita las horas de GPU por día, en una cantidad que varía.
4. **Si se corta:** vuelve a abrir el notebook y repite los pasos 1 y 2. Todo se retoma solo: las evaluaciones por partes de 256 filas, la cobertura y la descarga por ítem, y SASRec por combinación.
5. **Al terminar la sesión:** corre la última celda (**Exportar**). Descarga un zip y deja una copia en `Mi unidad/IIC3633/export/`. Envía ese zip a quien coordina. Después vacía la papelera de Drive: lo que se borra va a la papelera y sigue ocupando espacio.

**No modifiques el código ni los parámetros**, salvo los que esta guía indica. Si algo falla, no lo arregles: exporta y avisa con el mensaje de error.

## Compañero E: evaluación de BIGRec y D3

**Qué hace:** con cada modelo de BIGRec, genera el top-10 de todas las filas de `test` con BIGRec (`length_penalty` 1,0) y con D3 (0,0). La T4 no tiene bf16 en hardware, así que se evalúa en **fp16**: es una desviación respecto de los autores y queda registrada sola.

| Sesión | Qué corre | Duración en una T4 (medida en la sesión 1: 0,92 s por fila, 3,6× la GPU local) |
|---|---|---|
| 1 | **Validación:** BIGRec Toys, comparado con la corrida local | 1,7 h (medido; aprobada el 2026-11-26) |
| 2 | Video_Games: BIGRec y D3 | ≈ 4,8 h (estimado; si se corta, retoma por partes) |
| 3 | Musical_Instruments: BIGRec y D3 | ≈ 4,2 h (estimado) |
| 4 | CDs_and_Vinyl: BIGRec y D3 (cuando quien coordina suba el modelo) | ≈ 3,1 h (estimado: 6157 filas × 0,92 s × 2) |

Si un modelo aún no está en Drive, el notebook se salta ese dataset; vuelve a correrlo cuando quien coordina avise que lo subió.

**Qué revisar:**
- La celda de validación imprime `'ok': True`. Si dice `False`, el notebook se detiene: corre la celda **Exportar** y envía el zip. Solo sigue con Games, Instruments y CDs si quien coordina te pide poner `CONTINUAR_SIN_VALIDAR = True`.
- En la tabla de métricas, la columna `fuera del catálogo` es 0, y la celda termina sin error (si las métricas no coincidieran con `calc.py` de los autores, fallaría).
- `Mi unidad/IIC3633/colab/evaluaciones/gpu_log.csv` (una línea por minuto): durante la evaluación, `memory.used` debería rondar 9.000–11.000 MiB. *Supuesto:* la utilización oscila entre 40 y 90 %, porque en cada paso la CPU recorre el árbol de títulos. Si se queda bajo 20 % por más de 10 minutos, o la memoria baja a casi 0, revisa que la celda siga avanzando (imprime `parte k/n`). Si está detenida, reinicia la sesión y vuelve a correr todo.

## Compañero M: multimodal

**Qué hace:**
1. Revisa qué ítems de los 4 datasets tienen imagen y si el enlace sigue funcionando.
2. Descarga las imágenes de Toys.
3. Calcula sus embeddings con SigLIP (`google/siglip-so400m-patch14-384`).
4. Entrena SASRec con esos embeddings (3 variantes + SASRec normal) y lo compara con SASRec sin imágenes.

| Sesión | Parámetro | Qué corre | Duración estimada (supuesto) |
|---|---|---|---|
| 1 | `PRUEBA = True` (como viene) | Todo con 100 ítems y 1 combinación de 2 épocas, para detectar errores | 15–30 min (incluye bajar SigLIP, ~3,5 GB) |
| 2 | `PRUEBA = False` | Cobertura de los 4 datasets (~22.500 ítems) y descarga de Toys (~5.700 imágenes) | 20–60 min. Puede ser **sin GPU** (entorno "CPU"): ahorra cuota de GPU. Si el notebook se detiene en los embeddings por falta de GPU, sigue en la sesión 3 |
| 3 | `PRUEBA = False`, con T4 | Embeddings (15–30 min) y SASRec + SigLIP (28 combinaciones, 1–3 h) | 1,5–3,5 h |

**Qué revisar:**
- La tabla de cobertura. Según la metadata, Toys tiene URL de imagen en el 90,5 % de los ítems, CDs en el 51,2 %, Instruments en el 78,1 % y Games en el 90,1 %. `url_responde_imagen` dice cuántas de esas URLs siguen vivas **en la fecha de la revisión**. Por eso la cobertura no se vuelve a correr después: al retomar, conserva lo ya revisado.
- La descarga: `fallidas` incluye enlaces caídos y archivos que no son una imagen. Es normal que haya algunos.
- El manifiesto de embeddings: `items_con_imagen` cercano a las descargadas.
- La tabla final de SASRec: una fila por variante, más la referencia local.
- `gpu_log.csv` en `Mi unidad/IIC3633/colab/multimodal/`: en la cobertura y la descarga la GPU no se usa (0 % es lo esperable). En SASRec el modelo es chico y la utilización baja-media es normal (supuesto).

## Checklists

**Compañero E**
- [ ] Acceso directo `Mi unidad/IIC3633/checkpoints/bigrec` → `bigrec_compartido` creado.
- [ ] Sesión 1: T4, ejecutar todas, validación `ok: True`; zip enviado.
- [ ] Sesión 2: Video_Games BIGRec y D3 terminados (estado `terminada`); `fuera del catálogo` = 0; zip enviado.
- [ ] Sesión 3: Musical_Instruments BIGRec y D3 terminados; zip enviado.
- [ ] Sesión 4: CDs_and_Vinyl BIGRec y D3 terminados; zip enviado.
- [ ] Papelera de Drive vaciada.

**Compañero M**
- [ ] Sesión 1 (`PRUEBA = True`) sin errores; zip enviado.
- [ ] Sesión 2: cobertura de los 4 datasets y descarga de Toys; zip enviado.
- [ ] Sesión 3 (T4): embeddings y las 28 combinaciones de SASRec; tabla final; zip enviado.
- [ ] Papelera de Drive vaciada.

## Para quien coordina

- **Fijar la versión:** después del commit que agrega estos archivos, escribir ese hash en `COMMIT` en la primera celda de los dos notebooks (otro commit). Mientras diga `'PENDIENTE'`, el notebook se detiene en Colab.
- **Subir los modelos de BIGRec para E:** en Drive, `bigrec_compartido/<dataset>/` con `checkpoints/bigrec/<dataset>/modelo/` y `entrenamiento.json` (desde drive.google.com, "Subir carpeta"), y compartir `bigrec_compartido` con E como Editor. Toys ya existe; Games, Instruments y CDs se suben a medida que terminan sus SFT locales (CDs: celda 27 de `notebooks/bigrec/baseline_bigrec.ipynb`). Los modelos cuentan en el cupo de quien los sube (3,1 GB cada uno); las predicciones, en el de E.
- **Importar los resultados:** desde la raíz del repo, `python colab/importar_resultados.py <zip>` (con `--seco` para solo verificar). Rechaza cualquier archivo fuera de la lista blanca ([`lista_blanca.py`](lista_blanca.py)) o con otro sha256 que el del manifiesto, no pisa archivos distintos sin `--sobrescribir` y corre `calc.py` sobre cada `final_result.json`.
- **Qué entra a `resultados/`:** las predicciones y los `registro.json` de BIGRec/D3 (Games, Instruments, CDs y `bigrec_t4` de Toys), la validación, la cobertura, los manifiestos de imágenes y embeddings, los resultados de SASRec + SigLIP y `gpu_log.csv`/`etapas.json`. Nunca pesos, imágenes, embeddings ni partes intermedias.

## Archivos de esta carpeta

| Archivo | Qué es |
|---|---|
| `evaluar_bigrec_t4.ipynb` | Notebook de E |
| `preparar.py` | Preparación de la sesión (la llama la primera celda de los notebooks; se puede repetir) |
| `monitor.py` | Registro de `nvidia-smi` cada 60 s y de cada etapa (duración, estado, pico de memoria) en Drive |
| `exportar.py`, `lista_blanca.py` | Zip con los resultados permitidos y su manifiesto sha256 |
| `importar_resultados.py` | Importación local a `resultados/`, con verificación |
