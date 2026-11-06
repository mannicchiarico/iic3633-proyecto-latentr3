# LatentR3 (latentr3.ipynb)

LatentR3 completo: Qwen2.5-1.5B con **LatentRATT**, una capa de atención que produce un token latente de razonamiento a partir de los estados ocultos del prompt. Se entrena en dos etapas, un SFT de todo el modelo y un RL (LR-GRPO) que solo ajusta LatentRATT, y se evalúa con decodificación restringida al catálogo, como BIGRec. Se evalúan tres variantes por dataset: "only SFT" (Tabla 2 del paper), LatentR3 (BIGRec + LatentR3, Tabla 1) y D3 + LatentR3 (Tabla 1). Cada dataset tiene su propio modelo, entrenado desde Qwen2.5-1.5B base.

**Estado:** implementado y sin correr. El notebook no tiene outputs y no hay métricas propias. Solo existe la prueba de tiempo en Toys, hecha con scripts aparte (ver "Costos medidos"); sus JSON están en [`../../resultados/latentr3/`](../../resultados/latentr3/).

## Fuente y versión

- Repo de origen: [xuwenxinedu/R3](https://github.com/xuwenxinedu/R3), commit `c558b44`, copia local en `repo_latentr3/`.
- Clonado, desde la raíz del repo: `git clone https://github.com/xuwenxinedu/R3 repo_latentr3 && git -C repo_latentr3 checkout c558b44`.
- `cargar_autores` (en `latentr3.py`) verifica los sha256 de los originales y de las copias parcheadas antes de importar; si alguno no coincide, el notebook falla.
- Dos archivos se usan desde una **copia parcheada** en [`autores/`](autores/), con el diff contra el original en [`autores/cambios.diff`](autores/cambios.diff). El resto se carga sin modificar.

| Archivo de `repo_latentr3/` | sha256 (original) | Uso |
|---|---|---|
| `src/latent/reasoning_dataset.py` | `304eb670ae6a07e7a9107ea6c63d6e98834fcee82ab754a6dd7978927d1821cf` | Dataset del SFT, sin modificar |
| `src/latent/LogitProcesser.py` | `b55afbf044b8439fd4104df9810a69c37f094545581b7a0a5664118e6f236fdf` | Trie y procesador de logits, sin modificar |
| `src/latent/layers.py` | `0f25b70259ad4e83cc2d15096b5a742a76e255300b9286d6f0cf67a528dfa890` | Referencia de LatentRATT para las verificaciones, sin modificar |
| `src/grpo_attention_tuning/latent_grpo_dataset.py` | `da22298274630b4caadff9bad14c40291f39d4195a8b91f5aa9a09222996efa3` | Dataset del RL, sin modificar |
| `src/grpo_attention_tuning/latent_grpo_processor.py` | `12b66dc81bbfeaae2173cde70a853a02961e6a6929b7f1d063331552e0fa6862` | Sin modificar |
| `src/utils/calc.py` | `da8236af2ae792a2adaf2055cb10879baa1b67adc1e09721750cff3be2853e14` | Métricas de verificación, sin modificar |
| `src/grpo_attention_tuning/model.py` | `df344ddda8c0ea369791c8aee0d5b7072596ae2806f2674524b5fbd82bc8b46b` | Original de la copia parcheada |
| `src/grpo_attention_tuning/grpo_trainer.py` | `13ad92b83222bccdb44caf1ba481229ef63d646afbdcc1c9c6c401ab31cc0193` | Original de la copia parcheada |

| Copia parcheada | sha256 |
|---|---|
| `autores/model.py` | `fcff9e4d2e6a6e9b6cd21ca3133ec4d720360e574d5cfd98c44d7cc04d106081` |
| `autores/grpo_trainer.py` | `fb60ebcb62a8064f13feee2cbe47da04c377b36bc631b454b382ce695678aee6` |

Los argumentos de los scripts de entrenamiento y evaluación de los autores están copiados en `latentr3.py` con su línea de origen. El notebook importa `../baseline_comun.py` y `../bigrec/bigrec_baseline.py` (los hiperparámetros del SFT son los de BIGRec).

## Configuración

Fuentes: "paper" cita sección o apéndice; "repo" es el código de los autores (`latent_attention_train.py`, `train_noise_grpo.py`, `attention_grpo.sh`, `latent_train.sh`); "el paper no lo especifica" indica valores que se toman del código de los autores.

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B` (base, no Instruct), bf16 | paper, App. B |
| Token latente | 1 token `<|Thought|>`, id `len(tokenizer)-1` | paper, Sec. 4.2 (N = 1); repo `reasoning_dataset.py:122` |
| LatentRATT | Una cabeza d×d con Q, K y V con bias, RoPE base 1000; query = último token del prompt; 3(d²+d) parámetros | el paper no la especifica; repo `layers.py:32-120`; el autor la confirma en el issue #1 |
| Datos | Los mismos CSV de `repo_latentr3/data/` que los baselines, sin re-split | decisión del proyecto |
| Prompt | `LatentRDataset` (SFT) = `D3Dataset` (RL), con sus erratas; `cutoff_len` 512 | el paper no lo especifica; repo |
| **SFT** | | |
| Loss | Entropía cruzada sobre `"título"\n` + EOS, modelo completo; el gradiente pasa por los dos pases | paper, Sec. 3.2.1 (Eq. 4); repo `layers.py:145-158` |
| lr | 3e-4, sin grilla | decisión del proyecto; el autor lo confirma en el issue #2 (grilla del paper: {3e-3, 3e-4, 3e-5}, App. B) |
| Batch | 128 filas efectivas; micro-batch 2 con acumulación 64 | batch: repo, issue #2; micro-batch: decisión del proyecto por memoria |
| Épocas y parada | ≤ 3 épocas, loss de `valid` por época, paciencia 1, se queda con la mejor época | paper, App. B (paciencia 1); repo (3 épocas) |
| Scheduler | Lineal con warmup de 20 pasos | repo (default de HF) |
| Optimizador | AdamW con los estados en la RAM (`AdamWEnCPU`, F3) | paper, App. B (AdamW); F3: decisión del proyecto por memoria |
| Memoria | Gradient checkpointing no reentrante; evaluación de `valid` en lotes de 4 como los autores | decisión del proyecto; repo |
| **RL (LR-GRPO)** | | |
| Parámetros entrenables | Solo LatentRATT (`model.model`, embeddings y `lm_head` congelados); 7 082 496 parámetros | paper, Sec. 3.2.2; repo `train_noise_grpo.py:73-76`; medido, `resultados/latentr3/_tiempo/rl.json` |
| Muestreo | K = 8 muestras del token latente; la primera sin ruido; σ = 1,5 | el paper no especifica K ni σ (Eq. 5); repo `grpo_trainer.py:129-133` |
| Recompensa | −PPL del título objetivo con la política actual, sin generar | paper, Eq. 6; repo `grpo_trainer.py:156` |
| Ventaja | Recompensa menos la media de las muestras sin ruido de los grupos del micro-paso, dividida por la norma | paper, Eq. 7 ("batch" sin tamaño); repo |
| KL y clip | β = 0; clip 0,2 / 0,28, inactivo con `num_iterations` 1 | paper (KL cero); repo, script |
| lr | 5e-4 en los 4 datasets, sin grilla | decisión del proyecto; el autor lo sugiere para CDs (issue #2). Grilla del paper: {1e-5, 1e-4, 5e-4}, App. B |
| Batch | 256 filas por paso (32 prompts × 8); micro-batch 16 (2 grupos) con acumulación 16 | el paper no lo especifica; repo; micro-batch: decisión del proyecto (2 GPUs simuladas) |
| Scheduler y épocas | Coseno con warmup 200, 1 época, modelo final; checkpoint cada 200 pasos | el paper no lo especifica; repo `train_noise_grpo.py:128,151` |
| Validación | Ninguna (`eval_strategy="no"`) | decisión del proyecto, como los autores |
| Longitudes | `max_prompt_length` 512, `max_completion_length` 256, temperatura 1,0 | el paper no lo especifica; repo |
| Gradient checkpointing | No | decisión del proyecto: cabe en 16 GB (ver "Costos medidos") |
| **Evaluación** | | |
| Decodificación | Beam search de 10 beams restringido al catálogo con un trie de 5 tokens, `max_new_tokens` 128 | paper, App. B (restringida); repo `latent_attention_eval.py`, `noise_eval.py` |
| Lotes | 24 (SFT) y 12 (RL) | repo `latent_train.sh`, `attention_grpo.sh` |
| LatentR3 | `length_penalty` 1,0 | repo `latent_attention_eval.py:32`, `noise_eval.py` |
| D3 + LatentR3 | `length_penalty` 0,0 sobre el modelo del RL, sin ensemble con un modelo tradicional | paper, App. B (D3 sin ensemble); el valor 0,0 es una deducción del proyecto (base: `evaluate2.sh` de D3 sin `guidance_scale`) |
| Criterio de acierto | Por título; cada `final_result.json` se verifica con `gao` de `calc.py` | es el criterio de la Tabla 1; decisión del proyecto |
| **General** | | |
| Semilla | 42; una sola corrida, no determinista (kernels de GPU) | decisión del proyecto; el paper reporta una corrida determinista (App. E) |
| Hardware | 1 AMD Radeon RX 9070 XT de 16 GB (el paper usa 2 A100) | medido, `resultados/latentr3/_tiempo/tiempos.json`; paper, App. B |
| Entorno | torch 2.14.1+rocm7.14, `transformers==4.51.3`, `trl==0.16.1` | medido, `resultados/latentr3/_tiempo/tiempos.json`; versiones: repo `environment.yaml` |

El paper no especifica σ, K, el batch, las épocas, el scheduler, la arquitectura interna de LatentRATT ni el lr elegido de la grilla.

## Adaptaciones respecto de la fuente

La numeración salta de P3 a P5 porque P4 se probó y se revirtió (era una sola búsqueda de embeddings en `generate_embs`; no cambió el pico de memoria, 14,43 GB antes y después, según medición fuera del notebook, sin output guardado).

| Cambio | Archivo | Motivo | Efecto esperado en los resultados |
|---|---|---|---|
| P1: se quita `from noisy_layer import …` | `model.py` | El módulo no existe en el repo y el RL no arranca | Ninguno: solo lo usa código comentado |
| P2: `logits_to_keep=1` en el primer pase de `generate_embs` | `model.py` | Ese pase solo usa los estados ocultos; los logits de todo el vocabulario no caben en 16 GB con lotes de 24 × 10 beams (~15 GB, medición fuera del notebook, sin output guardado) | Ninguno: los estados ocultos no cambian (verificación V5) |
| P3: **2 GPUs simuladas**. Cada micro-paso trae 2 grupos; cada grupo se colaciona y puntúa por separado con el código original, la ventaja se calcula sobre los 2 grupos con las líneas originales y se hace el backward de cada grupo | `grpo_trainer.py` | El paper usa 2 A100 y los autores advierten que la ventaja por batch depende de las 2 GPUs; aquí hay una sola GPU | El mismo gradiente que con 2 GPUs (V7, V8). Cambian el RNG del ruido (con 2 GPUs cada proceso se siembra con `seed + process_index`; con uno, los ruidos de los dos grupos salen del mismo flujo con semilla 42) y el redondeo |
| P5: con gradiente, los dos productos de atención de LatentRATT se calculan elemento a elemento y suma en fp32 | `model.py` | Defecto de hipBLASLt con k = 1 en ROCm 7.14 (gfx1201) que mata el proceso (ver "Problemas conocidos") | Salida idéntica; gradientes idénticos con autocast (SFT) y con diferencia relativa de 4e-5 en bf16 (RL), según V5 y medición fuera del notebook, sin output guardado |
| F3: `AdamWEnCPU` para el SFT, en `latentr3.py` | `latentr3.py` | Pesos, gradientes y estados de AdamW no caben en 16 GB | Ninguno: llama a `torch.optim.adam.adam` con `decoupled_weight_decay=True`, la misma función de `torch.optim.AdamW`, con los estados en bf16 como `adamw_torch`. Parámetros idénticos bit a bit tras 5 pasos (medición fuera del notebook, sin output guardado) |

Cómo funciona F3: por cada parámetro copia el gradiente y el parámetro a la CPU, actualiza ahí y devuelve el parámetro a la GPU. No guarda una copia permanente del modelo en la RAM, así que no puede desincronizarse al retomar desde un checkpoint; al cargar un checkpoint devuelve los estados a la CPU.

**Equivalencia de las 2 GPUs simuladas (P3).** Con 2 GPUs × micro-batch 8 × acumulación 16 y con 1 proceso × micro-batch 16 × acumulación 16, el batch efectivo es 256 y el `RepeatRandomSampler` de TRL (`batch_size` = 256 / K = 32, `seed=42`) genera la misma secuencia de índices. Cada micro-paso de 16 filas trae [grupo a × 8, grupo b × 8], que son las 8 filas que recibía cada GPU (V8). Con 2 GPUs, TRL divide la loss por la acumulación (16) y DDP promedia las 2 GPUs: (L_a + L_b)/32. P3 hace el backward de cada grupo con L_g/(2 · 16), y V7 comprueba que el gradiente coincide. La equivalencia se demuestra por argumento y con V7/V8, no contra una corrida con 2 GPUs.

**Código de los autores que no se usa:** `grpo_eval.sh` (ruta rota a `src/grpo_full/noise_eval.py`), `attention_grpo.sh` para métricas (no las calcula) y el recorte de la salida de `noise_eval.py` (puede perder un lote entero si los prompts difieren en exactamente 3 tokens; aquí se decodifican solo los tokens generados).

**Opciones descartadas porque cambian el método:** Adam de 8 bits, LoRA, offload parcial y entrenamiento parcial.

### Verificaciones sin GPU (V1–V9)

Celdas 7 a 10 del notebook; corren en CPU, con modelos diminutos (2 capas, dimensión 64) que sirven para comprobar el código, no la calidad.

| # | Qué comprueba |
|---|---|
| V1 | sha256 de los originales y de las copias; `cambios.diff` igual al diff actual |
| V2 | `<|Thought|>` es un solo token con id `len(tokenizer)-1`, última fila tras `resize_token_embeddings`, con los embeddings atados a `lm_head` |
| V3 | Prompt del SFT = prompt del RL en 200 filas por split (y = prompt de BIGRec + token, si está `DecodingMatters/`); un solo `<|Thought|>` justo después de `### Response: \n`; el prompt no lleva loss |
| V4 | Trie con raíz de 5 tokens; todos los prompts de `test` terminan en ella; los títulos generables son exactamente los del catálogo |
| V5 | LatentRATT: 3(d²+d) parámetros, el token latente no ve el objetivo, la query es la posición anterior al token, el padding a la izquierda solo cambia el redondeo, P2 no cambia los estados ocultos, P5 da la misma salida y gradientes que la original |
| V6 | En el RL solo `attention.*` es entrenable y un paso solo cambia esos parámetros |
| V7 | P3: collator por grupo, ruido nulo en la fila 0, ventaja de la Eq. 7, gradiente por grupo = gradiente en un solo grafo (diferencia < 1e-6); el original falla con 1 grupo |
| V8 | Cada micro-paso de 16 filas trae los mismos 2 grupos que las 2 GPUs de los autores |
| V9 | Largos: `cutoff_len` 512 nunca trunca. Máximo de 441 tokens, en Instruments (medido fuera del notebook, sin output guardado) |

## Limitaciones y desviaciones respecto del paper

- **Una GPU de 16 GB en vez de 2 A100:** el RL se simula con 2 grupos por micro-paso (P3); el gradiente es el mismo, pero no se contrastó con una corrida de 2 GPUs.
- **Entorno distinto** (ROCm 7.14, torch 2.14, transformers 4.51.3, trl 0.16.1 frente a CUDA 12.1 y torch 2.6 de los autores): el autor advierte que el resultado cambia con el entorno.
- **Un solo lr del RL (5e-4) para los 4 datasets**, sin grilla; el lr elegido por los autores no está especificado.
- **lr del SFT 3e-4 sin grilla**, como confirma el autor.
- **Sin validación en el RL:** se usa el modelo final de 1 época, sin elegir checkpoint.
- **Valores tomados del código de los autores** porque el paper no los especifica: K = 8, σ = 1,5, batch de 256 filas, coseno con warmup 200, `max_prompt_length` 512 y `max_completion_length` 256.
- **Una sola corrida con semilla 42, no determinista**, mientras el paper reporta una corrida determinista (App. E).
- **D3 + LatentR3 con `length_penalty` 0,0** es una deducción del proyecto, no un dato de los autores.
- **Nuestro BIGRec en Toys podría estar ya cerca del "only SFT" del paper**, así que la ganancia absoluta del RL puede ser menor que en el paper. No se ha comparado: aún no hay resultados propios de LatentR3.
- **Retomar pierde trabajo:** en el SFT, hasta 1 época (los checkpoints son por época); en el RL, hasta 200 pasos.
- **Sin métricas propias:** la implementación no se ha corrido; ninguna de las tablas de "Resultados" tiene cifras propias.

## Cómo correrlo

- **Entorno:** `.venv-rocm` o `.venv-cuda` (ver el README de la raíz), con `transformers==4.51.3` y `trl==0.16.1`; el trainer parcheado depende de esa versión de TRL. El modelo base `Qwen/Qwen2.5-1.5B` debe estar en la caché de Hugging Face. En local el notebook trabaja sin conexión (`HF_HUB_OFFLINE=1`) y deja visible solo la GPU 0.
- **Directorio de trabajo:** `notebooks/latentr3/`. Los datos se leen desde `../../repo_latentr3/data`.
- **Fuentes externas:** `repo_latentr3/` clonado en el commit `c558b44` (ver "Fuente y versión"). Opcional, para V3: `DecodingMatters/` en la raíz, como para BIGRec.
- **Variable que hay que editar:** `DATASETS = ['toys_data']` (celda 3) define qué se entrena y evalúa; agregar uno no toca los demás.
- **Orden de las celdas:** 2-4 entorno y configuración; 6 datos; 8-10 verificaciones V1–V9 (solo CPU); 12-16 prueba rápida (opcional, ver abajo); 18 y 20 SFT y su evaluación; 22 y 24 RL y su evaluación (LatentR3 y D3 + LatentR3); 26 métricas; 28 comparación con el paper.
- **Prueba rápida (celdas 12-16):** en Toys, con 1024 filas de `train` (8 pasos por época), 256 de `valid`, 100 de `test` y 64 prompts para el RL (2 pasos). Comprueba el pipeline completo (0 títulos fuera del catálogo, métricas iguales a las de `calc.py`, solo cambia LatentRATT en el RL), vuelve a medir tiempos y escribe en `checkpoints/latentr3/_prueba/`, que se borra al final (`BORRAR_PRUEBA`). Duración estimada de 12 a 16 minutos de GPU (estimada, sin medición).
- **Archivos que escribe** en `checkpoints/latentr3/<categoría>/` (ignorado por git):
  - `sft/trainer/`: checkpoints por época, máximo 2; se borran al terminar. `sft/modelo/`: mejor época. `sft/entrenamiento.json` y `sft/predicciones/test/`.
  - `rl_5e-4/trainer/`: checkpoint cada 200 pasos, máximo 2; se borran al terminar. `rl_5e-4/modelo/`. `rl_5e-4/entrenamiento.json`, `rl_5e-4/predicciones/test/` (LatentR3) y `rl_5e-4/predicciones/test_d3/` (D3 + LatentR3).
  - Tamaños, medidos fuera del notebook, sin output guardado: ~9 GB por checkpoint del SFT y ~3,1 GB por modelo.
- **Retomar:** volver a correr la celda. Cada etapa continúa desde su último checkpoint y se salta si `entrenamiento.json` dice que terminó; la evaluación se salta las partes de 256 filas ya escritas.
- **Otros procesos:** conviene cerrar los kernels de Jupyter de otros notebooks antes de entrenar; el escritorio y otros procesos ocupan ~1,4-1,9 GB de la GPU (medición fuera del notebook, sin output guardado).

## Costos medidos

Todos los tiempos son de la GPU RX 9070 XT (16 GB) en Toys, con la LatentRATT recién inicializada (el tiempo por paso no depende del entrenamiento). Las pruebas se hicieron con scripts aparte, hoy fuera de `latentr3.py`, por lo que el notebook no tiene outputs que las respalden; los JSON versionados son la fuente.

| Etapa | GPU | Memoria GPU | RAM | Tiempo | Medido o estimado | Fuente |
|---|---|---|---|---|---|---|
| SFT, micro-batch 1, AdamW en la GPU | RX 9070 XT | 11,70 GB fijos desde el paso 1, sin contar los estados de Adam | no medido | 22,318 s por paso de 128 filas | medido (Toys, 157,64 tokens por fila) | `resultados/latentr3/_tiempo/sft1.json` |
| SFT, micro-batch 2, AdamW en la GPU | RX 9070 XT | 11,70 GB sin los estados de Adam | no medido | 14,17 s por paso de 128 filas | medido | `resultados/latentr3/_tiempo/sft2.json` |
| SFT, AdamW en la GPU (sin F3) | RX 9070 XT | Falla por memoria: 14,13 GiB asignados de 15,92 GiB (error del backward) | no medido | no aplica | medido | `resultados/latentr3/_tiempo/tiempos.json` |
| SFT con F3 (configuración final) | RX 9070 XT | 9,8 GB | 10,6 GB (con las filas más largas de los 4 datasets, 448 tokens) | 15,1 s por paso (14,17 s + ~0,9 s de F3) | medido fuera del notebook, sin output guardado (el 15,1 s es esa suma) | no hay archivo versionado |
| Loss de `valid` (lotes de 4) | RX 9070 XT | no medido | no medido | 0,048 s por fila (0,051 con micro-batch 1) | medido | `resultados/latentr3/_tiempo/sft2.json`, `sft1.json` |
| RL, 3 pasos | RX 9070 XT | 7,82 GB (pico) | no medido | 9,40 s por paso de 256 filas (mediana de 3 pasos; el primero tarda 12,62 s) | medido | `resultados/latentr3/_tiempo/rl.json` |
| RL, 20 pasos | RX 9070 XT | 8,58 GB (máximo) | no medido | 9,70 s por paso (mediana de 20 pasos) | medido | `resultados/latentr3/_tiempo_rl/tiempos.json` |
| RL, mediana de 50 pasos con `train()` | RX 9070 XT | 8,8 GB | no medido | 9,62 s por paso | medido fuera del notebook, sin output guardado | no hay archivo versionado |
| Generación, lotes de 24 (SFT) | RX 9070 XT | 13,14 GB (pico) | no medido | 0,372 s por fila | medido | `resultados/latentr3/_tiempo/gen.json` |
| Generación, lotes de 12 (RL) | RX 9070 XT | 8,15 GB (pico) | no medido | 0,410 s por fila | medido | `resultados/latentr3/_tiempo/gen.json` |
| Entrenamiento y evaluación completos | RX 9070 XT | no medido | no medido | no medido | no corrido | no aplica |

Notas:
- La generación de `gen.json` se midió con el LLM de BIGRec en Toys más una LatentRATT nueva, porque el modelo base sin ajustar genera títulos largos y sería más lento (medición fuera del notebook, sin output guardado).
- Las memorias del RL distinguen asignada y máxima: 7,82 GB (`rl.json`, `pico_gb`) frente a 8,58 GB (`_tiempo_rl`, `memoria_max_gb`); los 8,8 GB de la fila de 50 pasos no tienen archivo.
- Memoria del SFT: los pesos (2,9 GB), los gradientes (2,9 GB) y los estados de AdamW (5,8 GB) ocupan 11,6 GB fijos, y el backward suma ~2,8 GB transitorios (copias fp32 de los logits y gradientes densos del tamaño de la matriz de embeddings). El pico es casi igual con micro-batch 2 (14,43 GB) y 1 (14,61 GB). Medición fuera del notebook, sin output guardado; el `sft1.json` y el `sft2.json` registran 11,70 GB sin los estados de Adam.
- F3 libera 5,8 GB de GPU y usa ~5,8 GB de RAM más una copia temporal del parámetro más grande.

### Horas estimadas (por dataset)

Estimación (cálculo: función `estimar` de `latentr3.py` con `MEDICION_TOYS`, que usa 14,17 s + ~0,9 s por paso del SFT con 3 épocas, 9,62 s por paso del RL, 0,048 s por fila de `valid` y 0,372 s y 0,410 s por fila de generación; el SFT y el RL se escalan por tokens por fila respecto de Toys, 157,64; la generación no se escala). No se recalculó con otras cifras. Con paciencia 1 el SFT puede parar antes de la tercera época; las horas del SFT suponen las 3 épocas.

| Dataset | SFT | RL | Evaluación (SFT + RL + D3) | Total |
|---|---|---|---|---|
| CDs | 4,1 h | 3,3 h | 2,0 h | 9,4 h |
| Toys | 5,5 h | 4,5 h | 2,2 h | 12,2 h |
| Games | 7,4 h | 6,0 h | 3,1 h | 16,5 h |
| Instruments | 8,4 h | 6,8 h | 2,8 h | 18,0 h |
| **Total** | **25,4 h** | **20,5 h** | **10,1 h** | **56,1 h** |

Estimación del RL con 9,70 s por paso (`resultados/latentr3/_tiempo_rl/tiempos.json`, campo `estimacion`):

| Dataset | RL (1 época) |
|---|---|
| CDs | 3,353074352704075 h |
| Toys | 4,539211722215017 h |
| Games | 6,109786340077773 h |
| Instruments | 6,942250098978548 h |

Las tablas del SFT y de la evaluación no tienen un JSON equivalente; el `estimacion` de `_tiempo/tiempos.json` quedó en 0 porque el SFT falló por memoria en esa prueba.

## Resultados

**Resultados propios: no hay.** El notebook no se ha corrido, así que no existen la loss de `valid` por época del SFT, la recompensa por paso del RL ni las métricas de "only SFT", LatentR3 y D3 + LatentR3. Cuando se corra, las celdas 18, 22 y 26 las producen y la celda 28 las compara con el paper y con nuestro BIGRec y D3 (de `checkpoints/bigrec/`, con la mejora relativa media de las 4 métricas).

| Dataset | Variante | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| Toys | only SFT | no corrido | no corrido | no corrido | no corrido |
| Toys | LatentR3 | no corrido | no corrido | no corrido | no corrido |
| Toys | D3 + LatentR3 | no corrido | no corrido | no corrido | no corrido |
| CDs, Games, Instruments | las tres variantes | no corrido | no corrido | no corrido | no corrido |

### Valores del paper

Fuente: celda 28 del notebook, que copia las Tablas 1 y 2 del paper y el issue #2 del repo de los autores. "Only SFT" es la Tabla 2 (solo Toys y CDs); BIGRec + LatentR3 y D3 + LatentR3 son la Tabla 1.

| Dataset | Fuente | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| Toys | Tabla 1: BIGRec + LatentR3 | 0.0821 | 0.1107 | 0.0600 | 0.0693 |
| Toys | Tabla 1: D3 + LatentR3 | 0.0898 | 0.1152 | 0.0670 | 0.0752 |
| Toys | Tabla 2: w/o RL (only SFT) | 0.0804 | 0.1067 | 0.0584 | 0.0669 |
| CDs | Tabla 1: BIGRec + LatentR3 | 0.0934 | 0.1160 | 0.0754 | 0.0826 |
| CDs | Tabla 1: D3 + LatentR3 | 0.1137 | 0.1327 | 0.0915 | 0.0977 |
| CDs | Tabla 2: w/o RL (only SFT) | 0.0830 | 0.1005 | 0.0662 | 0.0719 |
| CDs | Issue #2: only SFT reproducido por un usuario con 2 GPUs | 0.07698554 | 0.09079097 | 0.0639346 | 0.06841407 |
| Games | Tabla 1: BIGRec + LatentR3 | 0.0580 | 0.0870 | 0.0413 | 0.0506 |
| Games | Tabla 1: D3 + LatentR3 | 0.0716 | 0.1006 | 0.0507 | 0.0601 |
| Instruments | Tabla 1: BIGRec + LatentR3 | 0.1029 | 0.1214 | 0.0882 | 0.0941 |
| Instruments | Tabla 1: D3 + LatentR3 | 0.1066 | 0.1229 | 0.0920 | 0.0973 |

## Dónde quedan los resultados

Las celdas se numeran desde 0 (posición en el JSON del notebook).

| Qué | Ruta local (no versionada) | Copia versionada | Celda |
|---|---|---|---|
| Log del SFT (loss de `valid` por época, tiempos, memoria) | `checkpoints/latentr3/<categoría>/sft/entrenamiento.json` | aún no hay | 18 |
| Predicciones del SFT | `.../sft/predicciones/test/` (`final_result.json`, `registro.json`) | aún no hay | 20 |
| Log del RL (recompensa por paso, tiempos, memoria) | `.../rl_5e-4/entrenamiento.json` | aún no hay | 22 |
| Predicciones de LatentR3 y de D3 + LatentR3 | `.../rl_5e-4/predicciones/test/` y `test_d3/` | aún no hay | 24 |
| Métricas de las tres variantes (con 0 títulos fuera del catálogo y verificación con `calc.py`) | calculadas desde los `final_result.json` | aún no hay | 26 |
| Comparación con el paper y con nuestro BIGRec y D3 | calculada en el notebook | aún no hay | 28 |
| Prueba de tiempo en Toys | scripts aparte, hoy no están en el repo | [`../../resultados/latentr3/_tiempo/`](../../resultados/latentr3/_tiempo/) (`tiempos.json`, `sft1.json`, `sft2.json`, `gen.json`, `rl.json`) y [`../../resultados/latentr3/_tiempo_rl/tiempos.json`](../../resultados/latentr3/_tiempo_rl/tiempos.json) | no hay celda |

Los pesos (`modelo/`) y los estados del entrenador (`trainer/`) no se versionan.

## Problemas conocidos

| Problema | Causa | Solución o estado | Efecto en los resultados |
|---|---|---|---|
| `train_noise_grpo.py` y `noise_eval.py` fallan con `ModuleNotFoundError` | `model.py` importa `noisy_layer`, que no existe en el repo | P1 | Ninguno |
| `grpo_eval.sh` no corre | Llama a `src/grpo_full/noise_eval.py`, que no existe (la ruta correcta es `src/grpo_attention_tuning/noise_eval.py`) | No se usa; la evaluación está en `latentr3.py` | Ninguno |
| `attention_grpo.sh` no calcula métricas | El script termina sin llamar a `calc.py` | Las métricas se calculan en la celda 26 y se verifican con `gao` de `calc.py` | Ninguno |
| El SFT no cabe en 16 GB | Pesos, gradientes y estados de AdamW ocupan 11,6 GB fijos más ~2,8 GB transitorios en el backward; el error de `_tiempo/tiempos.json` muestra 14,13 GiB asignados de 15,92 GiB; achicar el micro-batch casi no ayuda (pico de 14,43 GB con micro-batch 2 y 14,61 GB con micro-batch 1, sin output guardado) | F3 (estados de AdamW en la RAM). El asignador sin fragmentación (`expandable_segments`) no basta (demanda ~15,1 GB) | Ninguno (bit a bit idéntico a AdamW tras 5 pasos); +~0,9 s por paso |
| El RL con 1 GPU falla en `swap_adjacent_blocks` | La ventaja de la Eq. 7 necesita 2 grupos por micro-paso | P3 | El gradiente es el mismo que con 2 GPUs; cambian el RNG del ruido y el redondeo |
| `Memory access fault` en el backward de LatentRATT | hipBLASLt (ROCm 7.14, gfx1201) tiene un defecto en multiplicaciones por lotes con k = 1 (kernel 107193, `-m 1536 -n 160 -k 1 --batch_count 8`): lee más allá del final de un operando y solo falla cuando ese tensor termina justo en un borde de asignación de múltiplo de 2 MiB, por lo que aparece al azar. Esa multiplicación solo está en el backward de LatentRATT: fallaba siempre en el RL (5 de 5 corridas con `train()`) y a veces en el SFT. Se reprodujo aislado con `torch.bmm` | P5. Usar rocBLAS para todo también lo evita, pero hace el RL 6,9 veces más lento (64,7 s por paso frente a 9,4 s; medición fuera del notebook, sin output guardado), por eso se descartó. Con P5: 50 pasos de RL, 20 pasos con otra disposición de memoria y 15 pasos de SFT sin fallos (medición fuera del notebook, sin output guardado) | Diferencia relativa de 4e-5 en bf16 en los gradientes del RL |
| Riesgo restante de hipBLASLt | Quedan multiplicaciones con k = 1 en el cálculo de las frecuencias de RoPE (Qwen y LatentRATT), en fp32, sin lotes, sin backward y con otros kernels (140715 y 140716) | No fallaron en ninguna prueba | Ninguno observado |
| La variable `TORCH_BLAS_PREFER_HIPBLASLT=0` no sirve | No existe en este torch (no aparece en `libtorch_hip.so`) | No usar | Ninguno |
| Retomar pierde trabajo | Los checkpoints del SFT son por época y los del RL cada 200 pasos | Aceptado | Solo tiempo: hasta 1 época del SFT o 200 pasos del RL (~30 min) |
| Conteo de `test` de CDs | El notebook tiene 6157 filas y la Tabla 3 del paper dice 6158 (celda 6) | Se usan los CSV del repo | Una fila de diferencia |
