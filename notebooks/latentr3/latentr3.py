"""LatentR3 completo (SFT con token latente + LR-GRPO) en una GPU de 16 GB (lo usa latentr3.ipynb).

Del código de los autores (repo_latentr3/, commit c558b44) se cargan sin modificar, verificados por sha256, el
dataset del SFT, el del RL, el trie y el procesador de logits. `model.py` y `grpo_trainer.py` se usan desde la copia
parcheada en autores/ (parches P1, P2, P3 y P5; diff en autores/cambios.diff). El SFT usa AdamWEnCPU (F3), el mismo
AdamW con los estados en la RAM. Las decisiones están en README.md.
"""
import os
import io
import sys
import json
import math
import time
import glob
import shutil
import hashlib
import difflib
import numpy as np
import torch

from baseline_comun import SEED, buscar_archivo, verificar_sha256, cargar_modulo
from bigrec_baseline import (BATCH_SIZE, EPOCAS, LR, CUTOFF_LEN, WARMUP, PACIENCIA, SAVE_TOTAL_LIMIT, NUM_BEAMS,
                             MAX_NEW_TOKENS, FILAS_POR_PARTE, CATEGORY_DICT, a_hf, largos_tokens, entrenado,
                             titulos_catalogo, metricas, verificar_con_calc)

BASE_MODEL = "Qwen/Qwen2.5-1.5B"   # paper, App. B
TOKEN_LATENTE = "<|Thought|>"
# SFT: latent_attention_train.py (los mismos argumentos que BIGRec, importados de bigrec_baseline)
MICRO_SFT = 2            # acumulación 64; el batch efectivo sigue en 128
# RL: scripts/attention_grpo.sh y train_noise_grpo.py
LR_RL = 5e-4             # decisión del proyecto: solo 5e-4; el autor lo sugiere para CDs (issue #2)
NUM_GENERATIONS = 8      # K
BATCH_RL = 256           # filas por paso de optimización = 32 prompts x 8
GRUPOS_POR_MICRO = 2     # 2 GPUs simuladas (parche P3): un grupo por GPU de los autores
MICRO_RL = NUM_GENERATIONS * GRUPOS_POR_MICRO
WARMUP_RL = 200          # train_noise_grpo.py L128
SAVE_STEPS_RL = 200      # train_noise_grpo.py L151
# Evaluación: lotes de latent_train.sh (SFT) y attention_grpo.sh (RL)
BATCH_GEN_SFT = 24
BATCH_GEN_RL = 12
LENGTH_PENALTY = {"latentr3": 1.0,   # latent_attention_eval.py L32 / noise_eval.py
                  "d3": 0.0}         # D3 + LatentR3 sin TFA (deducción del proyecto, base: evaluate2.sh de D3)
# Configuración de memoria (README.md, "Adaptaciones"). Ninguna opción cambia el método
CONFIG_SFT = {"micro_batch": MICRO_SFT, "checkpointing": True, "optim_cpu": True}   # S1 + F3
CONFIG_RL = {"checkpointing": False}                                              # 8,8 GB medido fuera del notebook

COMMIT = "c558b44"
# sha256 de los archivos que se cargan sin modificar, y de los originales de las copias parcheadas
SHA256_REPO = {
    os.path.join("src", "latent", "reasoning_dataset.py"): "304eb670ae6a07e7a9107ea6c63d6e98834fcee82ab754a6dd7978927d1821cf",
    os.path.join("src", "latent", "LogitProcesser.py"): "b55afbf044b8439fd4104df9810a69c37f094545581b7a0a5664118e6f236fdf",
    os.path.join("src", "latent", "layers.py"): "0f25b70259ad4e83cc2d15096b5a742a76e255300b9286d6f0cf67a528dfa890",
    os.path.join("src", "grpo_attention_tuning", "latent_grpo_dataset.py"): "da22298274630b4caadff9bad14c40291f39d4195a8b91f5aa9a09222996efa3",
    os.path.join("src", "grpo_attention_tuning", "latent_grpo_processor.py"): "12b66dc81bbfeaae2173cde70a853a02961e6a6929b7f1d063331552e0fa6862",
    os.path.join("src", "grpo_attention_tuning", "model.py"): "df344ddda8c0ea369791c8aee0d5b7072596ae2806f2674524b5fbd82bc8b46b",
    os.path.join("src", "grpo_attention_tuning", "grpo_trainer.py"): "13ad92b83222bccdb44caf1ba481229ef63d646afbdcc1c9c6c401ab31cc0193",
    os.path.join("src", "utils", "calc.py"): "da8236af2ae792a2adaf2055cb10879baa1b67adc1e09721750cff3be2853e14",
}
COPIAS = {"model.py": os.path.join("src", "grpo_attention_tuning", "model.py"),
          "grpo_trainer.py": os.path.join("src", "grpo_attention_tuning", "grpo_trainer.py")}
# sha256 de las copias parcheadas revisadas (cambian solo si se edita un parche; ver cambios.diff)
SHA256_COPIAS = {"model.py": "fcff9e4d2e6a6e9b6cd21ca3133ec4d720360e574d5cfd98c44d7cc04d106081",
                 "grpo_trainer.py": "fb60ebcb62a8064f13feee2cbe47da04c377b36bc631b454b382ce695678aee6"}
AUTORES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autores")

# Filas de cada split (Tabla 3 / CSV del repo) y tokens por fila de train con el token latente
# medido con LatentRDataset
FILAS = {"CDs_and_Vinyl": (49251, 6156, 6157), "Toys_and_Games": (53898, 6737, 6738),
         "Video_Games": (75175, 9397, 9397), "Musical_Instruments": (66500, 8312, 8313)}
TOKENS_POR_FILA = {"CDs_and_Vinyl": 126.22, "Toys_and_Games": 155.92, "Video_Games": 150.49,
                   "Musical_Instruments": 193.01}
# Prueba de tiempo en Toys, RX 9070 XT (medido fuera del notebook; ver README.md, "Costos medidos")
MEDICION_TOYS = {"tokens_por_fila": 157.64,       # media de las filas medidas
                 "s_por_paso_sft": 15.1,          # 14,17 s con AdamW en la GPU + ~0,9 s de F3
                 "s_valid_por_fila": 0.048,
                 "s_por_paso_rl": 9.62,           # medido fuera del notebook, sin archivo versionado
                 "s_test_por_fila_sft": 0.372,    # lotes de 24
                 "s_test_por_fila_rl": 0.410}     # lotes de 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LatentModel = None
LatentRDataset = None
D3DatasetRL = None
get_prefix_data = None
get_hash = None
CFEnhancedLogitsProcessor = None
NoiseGRPORecTrainer = None
swap_adjacent_blocks = None


# ---------- Código de los autores ----------

def sha256(ruta):
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def diff_copias(latentr3):
    """diff unificado de cada original contra su copia parcheada (el contenido de autores/cambios.diff)."""
    partes = []
    for copia, original in COPIAS.items():
        with open(os.path.join(latentr3, original), encoding="utf-8") as f:
            a = f.readlines()
        with open(os.path.join(AUTORES, copia), encoding="utf-8") as f:
            b = f.readlines()
        partes += difflib.unified_diff(a, b, fromfile=f"repo_latentr3/{original}", tofile=f"autores/{copia}")
    return "".join(partes)


def cargar_autores(latentr3):
    """Verifica los sha256 y carga el código de los autores sin agregar carpetas a sys.path."""
    global LatentModel, LatentRDataset, D3DatasetRL, get_prefix_data, get_hash, CFEnhancedLogitsProcessor
    global NoiseGRPORecTrainer, swap_adjacent_blocks
    verificar_sha256(latentr3, SHA256_REPO, COMMIT, "xuwenxinedu/R3")
    for copia, esperado in SHA256_COPIAS.items():
        if sha256(os.path.join(AUTORES, copia)) != esperado:
            raise ValueError(f"autores/{copia} no es la copia parcheada revisada (sha256 distinto). "
                             "Si se cambió un parche a propósito, actualizar SHA256_COPIAS y cambios.diff.")
    src = os.path.join(latentr3, "src")
    LatentRDataset = cargar_modulo("latentr3_reasoning_dataset",
                                   os.path.join(src, "latent", "reasoning_dataset.py")).LatentRDataset
    logits = cargar_modulo("latentr3_logits", os.path.join(src, "latent", "LogitProcesser.py"))
    get_prefix_data, get_hash = logits.get_prefix_data, logits.get_hash
    CFEnhancedLogitsProcessor = logits.CFEnhancedLogitsProcessor
    D3DatasetRL = cargar_modulo("latentr3_grpo_dataset",
                                os.path.join(src, "grpo_attention_tuning", "latent_grpo_dataset.py")).D3Dataset
    # grpo_trainer.py hace `from latent_grpo_processor import ...`: se registra el módulo antes de cargarlo
    sys.modules["latent_grpo_processor"] = cargar_modulo(
        "latent_grpo_processor", os.path.join(src, "grpo_attention_tuning", "latent_grpo_processor.py"))
    LatentModel = cargar_modulo("latentr3_model", os.path.join(AUTORES, "model.py")).LatentModel
    trainer = cargar_modulo("latentr3_grpo_trainer", os.path.join(AUTORES, "grpo_trainer.py"))
    NoiseGRPORecTrainer, swap_adjacent_blocks = trainer.NoiseGRPORecTrainer, trainer.swap_adjacent_blocks


# ---------- Modelo, tokenizer y datos ----------

def tokenizer_con_latente(ruta=BASE_MODEL):
    """Tokenizer con <|Thought|> como último token especial, como latent_attention_train.py L78-86.

    Si `ruta` es un modelo ya entrenado, su tokenizer ya trae el token y solo se comprueba.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ruta)
    if TOKEN_LATENTE not in tokenizer.additional_special_tokens:
        especiales = tokenizer.additional_special_tokens
        especiales.append(TOKEN_LATENTE)
        tokenizer.add_special_tokens({"additional_special_tokens": especiales})
    assert tokenizer(TOKEN_LATENTE)["input_ids"][0] == len(tokenizer) - 1
    return tokenizer


def para_generar(tokenizer):
    # Padding a la izquierda con EOS, como latent_attention_eval.py L51-53 y train_noise_grpo.py L87-89
    tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def modelo_base_latente(tokenizer, ruta=BASE_MODEL):
    """Qwen2.5-1.5B + LatentRATT recién inicializada, con los embeddings redimensionados (L64-86)."""
    modelo = LatentModel.from_pretrained(ruta, torch_dtype=torch.bfloat16)
    modelo.attention.end_k = -1
    modelo.config.loss_type = "ce"
    modelo.resize_token_embeddings(len(tokenizer))
    assert modelo.model.embed_tokens.num_embeddings - 1 == tokenizer.convert_tokens_to_ids(TOKEN_LATENTE)
    return modelo


def cargar_modelo_latente(dir_modelo, tokenizer):
    """Modelo entrenado (SFT o RL) listo para generar, como latent_attention_eval.py L38-63."""
    modelo = LatentModel.from_pretrained(dir_modelo, torch_dtype=torch.bfloat16).to(device)
    modelo.attention.end_k = -1
    modelo.config.pad_token_id = modelo.config.eos_token_id = tokenizer.eos_token_id
    modelo.config.bos_token_id = tokenizer.bos_token_id
    modelo.config.use_cache = True   # se guardó en False para entrenar; solo cambia la velocidad
    return modelo.eval()


def dataset_sft(tokenizer, archivo, categoria, sample=-1, test=False, seed=0):
    # latent_attention_train.py L91-92; eval L54
    return LatentRDataset(train_file=archivo, tokenizer=tokenizer, max_len=2560 if test else CUTOFF_LEN,
                          sample=sample, test=test, seed=seed, category=CATEGORY_DICT[categoria], K=0)


def dataset_rl(tokenizer, archivo, categoria, sample=-1, seed=SEED, test=False):
    # train_noise_grpo.py L99-100
    return D3DatasetRL(train_file=archivo, tokenizer=tokenizer, max_len=2560 if test else CUTOFF_LEN, sample=sample,
                       seed=seed, category=CATEGORY_DICT[categoria], test=test)


# ---------- Optimizador del SFT (F3) ----------

class AdamWEnCPU(torch.optim.Optimizer):
    """AdamW de torch con los estados en la RAM en vez de la GPU (arreglo F3 para el SFT en 16 GB).

    Es el mismo algoritmo que `optim="adamw_torch"` de los autores (latent_attention_train.py L125): la actualización
    la hace la función `torch.optim.adam.adam` con `decoupled_weight_decay=True`, que es la que usa torch.optim.AdamW,
    y los estados quedan en el dtype del parámetro (bf16), como en el original. Por cada parámetro se copian a la CPU
    el gradiente y el parámetro, se actualiza ahí y el parámetro vuelve a la GPU. No se guarda una copia permanente
    del modelo en la RAM: así no puede quedar desincronizada al retomar desde un checkpoint.
    Libera los 5,8 GB de estados de la GPU; usa ~5,8 GB de RAM más una copia temporal del parámetro más grande.
    Medido fuera del notebook: parámetros idénticos bit a bit a torch.optim.AdamW tras 5 pasos.
    """

    def __init__(self, params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        from torch.optim.adam import adam
        perdida = closure() if closure is not None else None
        for grupo in self.param_groups:
            beta1, beta2 = grupo["betas"]
            for p in grupo["params"]:
                if p.grad is None:
                    continue
                estado = self.state[p]
                if not estado:   # como Adam._init_group sin capturable ni fused
                    estado["step"] = torch.tensor(0.0, dtype=torch.float32)
                    estado["exp_avg"] = torch.zeros(p.shape, dtype=p.dtype, device="cpu")
                    estado["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device="cpu")
                p_cpu, g_cpu = p.detach().to("cpu"), p.grad.detach().to("cpu")
                adam([p_cpu], [g_cpu], [estado["exp_avg"]], [estado["exp_avg_sq"]], [], [estado["step"]],
                     foreach=False, capturable=False, differentiable=False, fused=None, grad_scale=None,
                     found_inf=None, has_complex=False, decoupled_weight_decay=True, amsgrad=False, beta1=beta1,
                     beta2=beta2, lr=grupo["lr"], weight_decay=grupo["weight_decay"], eps=grupo["eps"], maximize=False)
                p.copy_(p_cpu)
                del p_cpu, g_cpu
        return perdida

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict lleva los estados al dispositivo del parámetro (la GPU); se devuelven a la CPU
        super().load_state_dict(state_dict)
        for estado in self.state.values():
            for clave in ("exp_avg", "exp_avg_sq", "step"):
                if clave in estado:
                    estado[clave] = estado[clave].to("cpu")


def optimizador_cpu(modelo, args):
    """AdamWEnCPU con los hiperparámetros que Trainer le daría a adamw_torch (lr, betas, eps y weight decay de args)."""
    return AdamWEnCPU([p for p in modelo.parameters() if p.requires_grad], lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon, weight_decay=args.weight_decay)


# ---------- Argumentos de entrenamiento ----------

def args_sft(salida, micro_batch=MICRO_SFT, checkpointing=True, **extra):
    """TrainingArguments de latent_attention_train.py L115-135, con la configuración de memoria."""
    from transformers import TrainingArguments
    base = dict(output_dir=salida, per_device_train_batch_size=micro_batch,
                per_device_eval_batch_size=4,   # 4 como los autores: eval_loss es una media de medias por lote
                gradient_accumulation_steps=BATCH_SIZE // micro_batch, warmup_steps=WARMUP, num_train_epochs=EPOCAS,
                learning_rate=LR, bf16=True, logging_steps=1, optim="adamw_torch", eval_strategy="epoch",
                save_strategy="epoch", save_total_limit=SAVE_TOTAL_LIMIT, load_best_model_at_end=True,
                group_by_length=False, report_to="none", remove_unused_columns=False, seed=SEED,
                gradient_checkpointing=checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None)
    base.update(extra)
    return TrainingArguments(**base)


def config_rl(salida, checkpointing=False, **extra):
    """GRPOConfig de train_noise_grpo.py L126-158 con los argumentos de attention_grpo.sh; micro-batch de 2 grupos."""
    import trl
    base = dict(warmup_steps=WARMUP_RL, num_generations=NUM_GENERATIONS, max_prompt_length=512, temperature=1.0,
                max_completion_length=256, use_vllm=False, learning_rate=LR_RL, beta=0.0, num_iterations=1,
                epsilon=0.2, epsilon_high=0.28, num_train_epochs=1, output_dir=salida,
                per_device_train_batch_size=MICRO_RL, per_device_eval_batch_size=MICRO_RL, seed=SEED,
                gradient_accumulation_steps=BATCH_RL // MICRO_RL, lr_scheduler_type="cosine", eval_strategy="no",
                save_steps=SAVE_STEPS_RL, save_total_limit=2, load_best_model_at_end=False, report_to="none", bf16=True,
                logging_steps=1, gradient_checkpointing=checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None)
    base.update(extra)
    return trl.GRPOConfig(**base)


def congelar_llm(modelo):
    # train_noise_grpo.py L73-76: solo LatentRATT se entrena
    for p in modelo.model.parameters():
        p.requires_grad = False
    for p in modelo.lm_head.parameters():
        p.requires_grad = False
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def recompensa_exact_match(completions, targets, **kwargs):
    # train_noise_grpo.py L112-117. El trainer la recibe pero no la llama (la recompensa es −PPL, L156)
    def strip_(a):
        return a.strip().strip('\n').strip('\"')
    return [1 if strip_(c) == strip_(t) else 0 for c, t in zip(completions, targets)]


def _cronometro():
    from transformers import TrainerCallback

    class Cronometro(TrainerCallback):
        # Tiempo de cada paso de optimización, sin contar la evaluación en valid
        def __init__(self):
            self.marcas = []
        def on_train_begin(self, args, state, control, **kwargs):
            self.marcas.append(time.time())
        def on_step_end(self, args, state, control, **kwargs):
            self.marcas.append(time.time())
    return Cronometro()


def _ultimo_checkpoint(dir_trainer):
    checkpoints = sorted(glob.glob(os.path.join(dir_trainer, "checkpoint-*")), key=lambda r: int(r.rsplit("-", 1)[1]))
    return checkpoints[-1] if checkpoints else None


def _escribir(ruta, datos):
    with open(ruta + ".tmp", "w") as f:
        json.dump(datos, f, indent=1, ensure_ascii=False)
    os.replace(ruta + ".tmp", ruta)


# ---------- Etapa 1: SFT ----------

def entrenar_sft(categoria, base, salida, sample_train=-1, sample_valid=-1, micro_batch=MICRO_SFT, checkpointing=True,
                 optim_cpu=True, borrar_intermedios=True, modelo_base=BASE_MODEL):
    """SFT de LatentR3 en una categoría (latent_attention_train.py). Retoma desde el último checkpoint y se salta si
    ya terminó.

    Guarda en `salida`: `trainer/` (checkpoints por época, se borran al final), `modelo/` (la época con menor
    eval_loss, con LatentRATT y el tokenizer con <|Thought|>) y `entrenamiento.json`. micro_batch, checkpointing y
    optim_cpu solo cambian la memoria y la velocidad, no el batch efectivo ni el método (README.md, "Adaptaciones").
    """
    from transformers import Trainer, EarlyStoppingCallback, DataCollatorForSeq2Seq
    import transformers
    dir_trainer, dir_modelo = os.path.join(salida, "trainer"), os.path.join(salida, "modelo")
    if entrenado(salida):
        print(f"{categoria}: SFT ya entrenado en {dir_modelo}; se salta.")
        with open(os.path.join(salida, "entrenamiento.json")) as f:
            return json.load(f)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Hace falta una GPU con bf16 (los autores entrenan en bf16; una T4 no sirve).")
    assert BATCH_SIZE % micro_batch == 0
    os.makedirs(salida, exist_ok=True)

    tokenizer = tokenizer_con_latente(modelo_base)
    train = dataset_sft(tokenizer, buscar_archivo(base, "train", categoria), categoria, sample_train)
    valid = dataset_sft(tokenizer, buscar_archivo(base, "valid", categoria), categoria, sample_valid)
    modelo = modelo_base_latente(tokenizer, modelo_base)
    args = args_sft(dir_trainer, micro_batch, checkpointing)
    cronometro = _cronometro()
    trainer = Trainer(
        model=modelo, args=args, train_dataset=a_hf(train), eval_dataset=a_hf(valid), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PACIENCIA), cronometro],
        optimizers=(optimizador_cpu(modelo, args) if optim_cpu else None, None),
    )
    modelo.config.use_cache = False   # L140

    checkpoint = _ultimo_checkpoint(dir_trainer)
    paso_inicial = int(checkpoint.rsplit("-", 1)[1]) if checkpoint else 0
    if checkpoint:
        print(f"{categoria}: se retoma el SFT desde {checkpoint}")
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    segundos = time.time() - inicio

    modelo.save_pretrained(dir_modelo)    # la mejor época, cargada por load_best_model_at_end
    tokenizer.save_pretrained(dir_modelo)
    pasos = np.diff(cronometro.marcas)
    evaluaciones = [h for h in trainer.state.log_history if "eval_loss" in h]
    registro = {
        "categoria": categoria, "etapa": "sft", "terminado": True,
        "mejor_checkpoint": trainer.state.best_model_checkpoint, "mejor_eval_loss": trainer.state.best_metric,
        "evaluaciones": [{"epoca": h["epoch"], "eval_loss": h["eval_loss"], "eval_runtime": h["eval_runtime"]}
                         for h in evaluaciones],
        "pasos_totales": trainer.state.global_step, "pasos_esta_sesion": trainer.state.global_step - paso_inicial,
        "filas_train": len(train), "filas_valid": len(valid), "segundos_esta_sesion": segundos,
        "s_por_paso_mediana": float(np.median(pasos[1:])) if len(pasos) > 1 else None,   # sin el primer paso
        "s_valid_por_fila": (float(np.mean([h["eval_runtime"] for h in evaluaciones])) / len(valid)
                             if evaluaciones else None),
        "memoria_max_gb": torch.cuda.max_memory_allocated() / 2**30,
        "memoria_reservada_max_gb": torch.cuda.max_memory_reserved() / 2**30,
        "micro_batch": micro_batch, "acumulacion": BATCH_SIZE // micro_batch, "gradient_checkpointing": checkpointing,
        "optimizador": "AdamWEnCPU (F3)" if optim_cpu else "adamw_torch",
        "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "log_history": trainer.state.log_history,
    }
    _escribir(os.path.join(salida, "entrenamiento.json"), registro)   # se escribe al final: marca el fin
    del trainer, modelo
    torch.cuda.empty_cache()
    if borrar_intermedios:
        shutil.rmtree(dir_trainer, ignore_errors=True)
    return registro


# ---------- Etapa 2: LR-GRPO ----------

def entrenar_rl(categoria, base, salida, dir_sft, sample=-1, checkpointing=False, borrar_intermedios=True):
    """LR-GRPO de LatentR3 (train_noise_grpo.py con el trainer parcheado, 2 GPUs simuladas) desde el modelo del SFT.

    Solo entrena LatentRATT. Retoma desde el último checkpoint (cada 200 pasos) y se salta si ya terminó. Guarda en
    `salida`: `trainer/` (se borra al final), `modelo/` (el modelo tras 1 época, como los autores) y
    `entrenamiento.json` con la recompensa por paso.
    """
    import transformers
    import trl
    dir_trainer, dir_modelo = os.path.join(salida, "trainer"), os.path.join(salida, "modelo")
    if entrenado(salida):
        print(f"{categoria}: RL ya entrenado en {dir_modelo}; se salta.")
        with open(os.path.join(salida, "entrenamiento.json")) as f:
            return json.load(f)
    if not os.path.isfile(os.path.join(dir_sft, "config.json")):
        raise FileNotFoundError(f"Falta el modelo del SFT en {dir_sft}: correr antes entrenar_sft.")
    os.makedirs(salida, exist_ok=True)

    tokenizer = para_generar(tokenizer_con_latente(dir_sft))   # padding a la izquierda (L87-89)
    train = dataset_rl(tokenizer, buscar_archivo(base, "train", categoria), categoria, sample)
    modelo = LatentModel.from_pretrained(dir_sft, torch_dtype=torch.bfloat16)   # L69-71
    modelo.attention.end_k = -1
    entrenables = congelar_llm(modelo)
    trie = get_prefix_data(buscar_archivo(base, "info", categoria), tokenizer)
    cronometro = _cronometro()
    trainer = NoiseGRPORecTrainer(
        prefix_allowed_tokens_fn=lambda _, ids: trie.get(get_hash(ids), []), model=modelo,
        reward_funcs=[recompensa_exact_match], args=config_rl(dir_trainer, checkpointing=checkpointing),
        train_dataset=a_hf(train), eval_dataset=None, processing_class=tokenizer, callbacks=[cronometro])
    modelo.config.use_cache = False   # L163

    checkpoint = _ultimo_checkpoint(dir_trainer)
    paso_inicial = int(checkpoint.rsplit("-", 1)[1]) if checkpoint else 0
    if checkpoint:
        print(f"{categoria}: se retoma el RL desde {checkpoint}")
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    segundos = time.time() - inicio

    modelo.save_pretrained(dir_modelo)   # L177
    tokenizer.save_pretrained(dir_modelo)
    pasos = np.diff(cronometro.marcas)
    historial = [h for h in trainer.state.log_history if "recompensa" in h]
    registro = {
        "categoria": categoria, "etapa": "rl", "terminado": True, "lr": LR_RL, "dir_sft": dir_sft,
        "pasos_totales": trainer.state.global_step, "pasos_esta_sesion": trainer.state.global_step - paso_inicial,
        "filas_train": len(train), "parametros_entrenables": entrenables, "segundos_esta_sesion": segundos,
        "s_por_paso_mediana": float(np.median(pasos[1:])) if len(pasos) > 1 else None,
        "recompensa_por_paso": [h["recompensa"] for h in historial],
        "recompensa_s1_por_paso": [h.get("recompensa_s1") for h in historial],
        "memoria_max_gb": torch.cuda.max_memory_allocated() / 2**30,
        "memoria_reservada_max_gb": torch.cuda.max_memory_reserved() / 2**30,
        "micro_batch": MICRO_RL, "acumulacion": BATCH_RL // MICRO_RL, "gradient_checkpointing": checkpointing,
        "torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
        "gpu": torch.cuda.get_device_name(0), "log_history": trainer.state.log_history,
    }
    _escribir(os.path.join(salida, "entrenamiento.json"), registro)
    del trainer, modelo
    torch.cuda.empty_cache()
    if borrar_intermedios:
        shutil.rmtree(dir_trainer, ignore_errors=True)
    return registro


def verificar_rl_congelado(dir_sft, dir_rl):
    """Compara tensor por tensor los pesos del SFT y del RL: solo attention.* debe haber cambiado."""
    from safetensors import safe_open
    rutas = {d: sorted(glob.glob(os.path.join(d, "*.safetensors"))) for d in (dir_sft, dir_rl)}
    assert len(rutas[dir_sft]) == len(rutas[dir_rl]) == 1, "se esperaba un solo archivo .safetensors por modelo"
    iguales, cambiados = [], []
    with safe_open(rutas[dir_sft][0], "pt") as a, safe_open(rutas[dir_rl][0], "pt") as b:
        assert set(a.keys()) == set(b.keys())
        for clave in a.keys():
            (iguales if torch.equal(a.get_tensor(clave), b.get_tensor(clave)) else cambiados).append(clave)
    assert cambiados and all(c.startswith("attention.") for c in cambiados), cambiados[:5]
    return {"tensores iguales": len(iguales), "tensores cambiados": cambiados}


# ---------- Decodificación restringida y evaluación ----------

@torch.no_grad()
def generar(modelo, tokenizer, encodings, trie, length_penalty, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS):
    """Top-`num_beams` títulos para un lote de prompts, como latent_attention_eval.py L65-131 / noise_eval.py.

    Padding a la izquierda con attention_mask; se decodifican solo los tokens generados (evita el recorte por
    minLen de noise_eval.py). No se piden los puntajes por paso: el orden de los beams es el mismo.
    """
    from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper
    largo = max(len(e["input_ids"]) for e in encodings)
    ids = [[tokenizer.pad_token_id] * (largo - len(e["input_ids"])) + e["input_ids"] for e in encodings]
    mascara = [[0] * (largo - len(e["input_ids"])) + e["attention_mask"] for e in encodings]
    config = GenerationConfig(num_beams=num_beams, length_penalty=length_penalty, num_return_sequences=num_beams,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                              max_new_tokens=max_new_tokens)
    restriccion = CFEnhancedLogitsProcessor(
        guidance_scale=1.0, cf_logits=None, cf_dict=None, unconditional_ids=None, model=modelo, tokenizer=tokenizer,
        prefix_allowed_tokens_fn=lambda _, prefijo: trie.get(get_hash(prefijo), []), num_beams=num_beams)
    salida = modelo.generate(torch.tensor(ids, device=modelo.device),
                             attention_mask=torch.tensor(mascara, device=modelo.device),
                             generation_config=config, return_dict_in_generate=True,
                             logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), restriccion]))
    textos = tokenizer.batch_decode(salida.sequences[:, largo:], skip_special_tokens=True)
    return [textos[i * num_beams:(i + 1) * num_beams] for i in range(len(encodings))]


def evaluar_categoria(categoria, base, dir_modelo, dir_pred, batch_size, length_penalty, sample=-1,
                      filas_por_parte=FILAS_POR_PARTE):
    """Genera el top-10 de cada fila de test con el modelo de `dir_modelo` y lo guarda por partes en `dir_pred`.

    Se salta las partes ya escritas, así que se puede retomar. Al final une las partes en `final_result.json`, con
    el formato que lee calc.py de los autores. El prompt es el de LatentRDataset (= D3Dataset del RL; V3).
    """
    final = os.path.join(dir_pred, "final_result.json")
    if os.path.isfile(final):
        print(f"{categoria}: ya evaluado en {final}; se salta.")
        return final
    if not os.path.isfile(os.path.join(dir_modelo, "config.json")):
        raise FileNotFoundError(f"Falta el modelo en {dir_modelo}.")
    os.makedirs(dir_pred, exist_ok=True)
    tokenizer = para_generar(tokenizer_con_latente(dir_modelo))
    test = dataset_sft(tokenizer, buscar_archivo(base, "test", categoria), categoria, sample=sample, test=True)
    encodings = [test[i] for i in range(len(test))]
    filas = test.get_all()
    trie = get_prefix_data(buscar_archivo(base, "info", categoria), tokenizer)
    ruta_registro = os.path.join(dir_pred, "registro.json")
    registro = {"filas": 0, "segundos": 0.0, "batch_size": batch_size, "length_penalty": length_penalty,
                "modelo": dir_modelo}
    if os.path.isfile(ruta_registro):
        with open(ruta_registro) as f:
            registro = json.load(f)

    modelo = None
    n_partes = math.ceil(len(encodings) / filas_por_parte)
    for p in range(n_partes):
        ruta = os.path.join(dir_pred, f"parte_{p:03d}.json")
        if os.path.isfile(ruta):
            continue
        if modelo is None:
            modelo = cargar_modelo_latente(dir_modelo, tokenizer)
            torch.cuda.reset_peak_memory_stats()
        ini, fin = p * filas_por_parte, min((p + 1) * filas_por_parte, len(encodings))
        inicio = time.time()
        predicciones = []
        for i in range(ini, fin, batch_size):
            predicciones += generar(modelo, tokenizer, encodings[i:min(i + batch_size, fin)], trie, length_penalty)
        segundos = time.time() - inicio
        parte = []
        for fila, prediccion in zip(filas[ini:fin], predicciones):
            fila = {k: v for k, v in fila.items() if k != "dedup"}   # eval L157-159
            fila["predict"] = prediccion
            parte.append(fila)
        _escribir(ruta, parte)   # una parte a medio escribir nunca queda con el nombre final
        registro["filas"] += fin - ini
        registro["segundos"] += segundos
        registro["memoria_max_gb"] = torch.cuda.max_memory_allocated() / 2**30
        _escribir(ruta_registro, registro)
        print(f"{categoria}: parte {p + 1}/{n_partes}, {segundos / (fin - ini):.3f} s por fila")

    datos = []
    for p in range(n_partes):   # como merge.py de los autores
        with open(os.path.join(dir_pred, f"parte_{p:03d}.json")) as f:
            datos.extend(json.load(f))
    assert len(datos) == len(encodings)
    _escribir(final, datos)
    if modelo is not None:
        del modelo
        torch.cuda.empty_cache()
    return final


def rutas(salida_categoria):
    """Carpetas de un dataset dentro de checkpoints/latentr3/<categoría>/."""
    sft = os.path.join(salida_categoria, "sft")
    rl = os.path.join(salida_categoria, "rl_" + f"{LR_RL:.0e}".replace("e-0", "e-"))   # rl_5e-4
    return {"sft": sft, "modelo_sft": os.path.join(sft, "modelo"), "pred_sft": os.path.join(sft, "predicciones", "test"),
            "rl": rl, "modelo_rl": os.path.join(rl, "modelo"), "pred_rl": os.path.join(rl, "predicciones", "test"),
            "pred_d3": os.path.join(rl, "predicciones", "test_d3")}


# ---------- Verificaciones sin GPU ----------

def verificar_codigo(latentr3):
    """V1: sha256 de los originales y de las copias, y que cambios.diff sea el diff actual de las copias."""
    verificar_sha256(latentr3, SHA256_REPO, COMMIT, "xuwenxinedu/R3")
    copias = {c: sha256(os.path.join(AUTORES, c)) for c in COPIAS}
    assert copias == SHA256_COPIAS, copias
    with open(os.path.join(AUTORES, "cambios.diff"), encoding="utf-8") as f:
        assert f.read() == diff_copias(latentr3), "cambios.diff no corresponde a las copias actuales"
    parches = [l for l in diff_copias(latentr3).splitlines() if l.startswith("+") and "PARCHE P" in l]
    return {"originales": len(SHA256_REPO), "copias verificadas": list(copias),
            "líneas con PARCHE en el diff": len(parches)}


def verificar_token(tokenizer):
    """V2: <|Thought|> es un solo token con id len(tokenizer)-1, y tras redimensionar es la última fila."""
    from transformers import Qwen2Config
    ids = tokenizer(TOKEN_LATENTE)["input_ids"]
    assert len(ids) == 1 and ids[0] == len(tokenizer) - 1
    config = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    diminuto = LatentModel(config)
    diminuto.resize_token_embeddings(len(tokenizer))   # como L85: 151 936 → len(tokenizer) filas
    assert diminuto.model.embed_tokens.num_embeddings - 1 == ids[0]
    assert diminuto.lm_head.weight.data_ptr() == diminuto.model.embed_tokens.weight.data_ptr()   # siguen atados
    return {"id de <|Thought|>": ids[0], "len(tokenizer)": len(tokenizer),
            "filas de embeddings tras el resize": diminuto.model.embed_tokens.num_embeddings}


def verificar_prompt_latente(base, categoria, tokenizer, filas=200):
    """V3: LatentRDataset (SFT) y D3Dataset del RL dan los mismos tokens, con un solo <|Thought|> justo después de
    '### Response: \\n'. La comparación con D3Dataset de BIGRec (prompt sin el token) la hace bb.verificar_prompt."""
    id_latente = len(tokenizer) - 1
    for split, test in [("train", False), ("valid", False), ("test", True)]:
        archivo = buscar_archivo(base, split, categoria)
        sft = dataset_sft(tokenizer, archivo, categoria, sample=filas, test=test, seed=SEED)
        rl = dataset_rl(tokenizer, archivo, categoria, sample=filas, seed=SEED, test=test)
        for i in range(len(sft)):
            a, b = sft[i], rl[i]
            for clave in ("input_ids", "attention_mask") + (() if test else ("labels",)):
                assert a[clave] == b[clave], (split, i, clave)
            ids = a["input_ids"]
            pos = [j for j, t in enumerate(ids) if t == id_latente]
            assert len(pos) == 1, (split, i, "se esperaba un solo token latente")
            assert tokenizer.decode(ids[pos[0] - 4:pos[0]]) == "### Response: \n", (split, i, "no sigue a '### Response: \\n'")
            if test:
                assert pos[0] == len(ids) - 1, (split, i, "en test el prompt debe terminar en el token latente")
            if not test:
                assert all(l == -100 for l in a["labels"][:pos[0] + 1]), (split, i, "el prompt no debe llevar loss")
    return tokenizer.decode(dataset_sft(tokenizer, buscar_archivo(base, "train", categoria), categoria, sample=1,
                                        seed=SEED)[0]["input_ids"])


def verificar_trie_latente(info_file, tokenizer, encodings_test):
    """V4: trie de get_prefix_data con raíz de 5 tokens; todos los prompts de test terminan en ella y los títulos
    generables son exactamente los del catálogo."""
    trie = get_prefix_data(info_file, tokenizer)
    with open(info_file, "r") as f:   # el mismo armado que get_prefix_data (LogitProcesser.py L15-40)
        info = ["### Response: \n" + "\"" + l.split("\t")[0].strip(" ") + "\"\n" for l in f.readlines()]
    prefijos = [tokenizer(t).input_ids for t in info]
    for ids in prefijos:
        ids.insert(4, len(tokenizer) - 1)
        ids.append(tokenizer.eos_token_id)
    raices = {tuple(ids[:5]) for ids in prefijos}
    assert len(raices) == 1, f"la raíz de 5 tokens se tokeniza de {len(raices)} formas"
    raiz = list(raices.pop())
    finales = {tuple(e["input_ids"][-5:]) for e in encodings_test}
    assert finales == {tuple(raiz)}, "algún prompt no termina en los 5 tokens de la raíz del trie"
    generables, pila = [], [[]]
    while pila:
        camino = pila.pop()
        for token in trie.get(get_hash(raiz) if not camino else get_hash(camino), []):
            if token == tokenizer.eos_token_id:
                generables.append(tokenizer.decode(camino).strip(" \n").strip("\"").strip(" "))
            else:
                pila.append(camino + [token])
    catalogo = set(titulos_catalogo(info_file))
    assert set(generables) == catalogo, (len(set(generables) - catalogo), len(catalogo - set(generables)))
    tokens = [len(ids) - 5 for ids in prefijos]
    return {"ítems": len(prefijos), "títulos únicos": len(catalogo), "caminos del trie": len(generables),
            "tokens del título más largo": max(tokens),
            f"títulos con más de {MAX_NEW_TOKENS} tokens": sum(t > MAX_NEW_TOKENS for t in tokens)}


def _diminuto(vocab, d=64):
    """LatentModel con un Qwen2 de 2 capas en fp32 y CPU, para verificar sin GPU."""
    from transformers import Qwen2Config
    torch.manual_seed(SEED)
    config = Qwen2Config(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         tie_word_embeddings=True, attn_implementation="eager")
    modelo = LatentModel(config).float().eval()
    modelo.attention.end_k = -1
    return modelo


def verificar_latentratt(latentr3, vocab=1000, d=64):
    """V5: LatentRATT y los parches P2 y P5 sobre un modelo diminuto en CPU (fp32)."""
    import transformers
    modelo = _diminuto(vocab, d)
    latente = vocab - 1
    torch.manual_seed(1)
    n = 40
    ids = torch.randint(0, vocab - 2, (1, n))
    ids[0, 30] = latente   # token latente en la posición 30; después, 9 tokens de "objetivo"
    mascara = torch.ones_like(ids)
    resultado = {"parámetros de LatentRATT": sum(p.numel() for p in modelo.attention.parameters()),
                 "esperado 3(d²+d)": 3 * (d * d + d)}
    assert resultado["parámetros de LatentRATT"] == resultado["esperado 3(d²+d)"]
    with torch.no_grad():
        r = modelo.generate_embs(ids, mascara)[0, 30]
        # (b) sin fuga: cambiar los tokens posteriores al latente no cambia r
        otro = ids.clone(); otro[0, 31:] = torch.randint(0, vocab - 2, (n - 31,))
        assert torch.equal(modelo.generate_embs(otro, mascara)[0, 30], r)
        # (c) la query es la posición anterior al latente: cambiar ese token sí cambia r
        otro = ids.clone(); otro[0, 29] = (ids[0, 29] + 1) % (vocab - 2)
        assert not torch.allclose(modelo.generate_embs(otro, mascara)[0, 30], r)
        # (d) padding a la izquierda (generación): r igual salvo redondeo (RoPE es relativo)
        pad = torch.cat([torch.zeros(1, 7, dtype=ids.dtype), ids], 1)
        mpad = torch.cat([torch.zeros(1, 7, dtype=mascara.dtype), mascara], 1)
        dif_pad = (modelo.generate_embs(pad, mpad)[0, 37] - r).abs().max().item()
        assert dif_pad < 1e-4, dif_pad
        # (e) P2: el pase con logits_to_keep=1 da los mismos estados ocultos que el original (sin logits_to_keep)
        completo = transformers.Qwen2ForCausalLM.forward(modelo, input_ids=ids, attention_mask=mascara,
                                                         output_hidden_states=True)
        recortado = transformers.Qwen2ForCausalLM.forward(modelo, input_ids=ids, attention_mask=mascara,
                                                          output_hidden_states=True, logits_to_keep=1)
        assert all(torch.equal(a, b) for a, b in zip(completo.hidden_states, recortado.hidden_states))
        h = completo.hidden_states[-1]
    # (f) P5: con gradiente (producto elemento a elemento) = sin gradiente (matmul) = LatentRATT original de layers.py
    original = cargar_modulo("latentr3_layers_original", os.path.join(latentr3, "src", "latent", "layers.py"))
    ref = original.SelfAttentionLayer(d).float()
    ref.load_state_dict(modelo.attention.state_dict())
    idx = torch.tensor([30])
    with torch.no_grad():
        sin_grad = modelo.attention(h, mascara, idx)
    h1, h2 = h.clone().requires_grad_(True), h.clone().requires_grad_(True)
    con_grad = modelo.attention(h1, mascara, idx)
    salida_ref = ref(h2, mascara, idx)
    g = torch.randn_like(con_grad)
    con_grad.backward(g)
    salida_ref.backward(g)
    resultado["P5: dif. máx. salida (con vs. sin gradiente)"] = (con_grad - sin_grad).abs().max().item()
    resultado["P5: dif. máx. salida vs. original"] = (con_grad - salida_ref).abs().max().item()
    resultado["P5: dif. máx. gradiente vs. original"] = max(
        (a.grad - b.grad).abs().max().item() for a, b in zip(modelo.attention.parameters(), ref.parameters()))
    resultado["padding: dif. máx. de r"] = dif_pad
    assert resultado["P5: dif. máx. salida (con vs. sin gradiente)"] < 1e-5
    assert resultado["P5: dif. máx. salida vs. original"] < 1e-5
    assert resultado["P5: dif. máx. gradiente vs. original"] < 1e-4
    return resultado


def verificar_congelamiento(vocab=1000, d=64):
    """V6: tras congelar_llm solo attention.* es entrenable, y un paso de AdamW solo cambia esos parámetros."""
    modelo = _diminuto(vocab, d).train()
    entrenables = congelar_llm(modelo)
    nombres = [n for n, p in modelo.named_parameters() if p.requires_grad]
    assert all(n.startswith("attention.") for n in nombres) and entrenables == 3 * (d * d + d)
    antes = {n: p.detach().clone() for n, p in modelo.named_parameters()}
    ids = torch.randint(0, vocab - 2, (4, 20)); ids[:, 12] = vocab - 1
    mascara = torch.ones_like(ids)
    embeds = modelo.generate_embs(ids, mascara)   # como compute_loss del trainer (L212)
    perdida = modelo(input_ids=None, inputs_embeds=embeds, attention_mask=mascara, use_cache=False).logits[:, -5:].logsumexp(-1).mean()
    perdida.backward()
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    cambiados = [n for n, p in modelo.named_parameters() if not torch.equal(p, antes[n])]
    assert cambiados and all(n.startswith("attention.") for n in cambiados), cambiados
    return {"parámetros entrenables": entrenables, "tensores cambiados tras un paso": cambiados}


def verificar_p3(base, categoria, tokenizer, salida_tmp):
    """V7: el trainer parcheado (2 GPUs simuladas) sobre un modelo diminuto en CPU con filas reales.

    (a) el collator por grupo da los mismos tensores que colacionar cada grupo solo; (b) la fila 0 de cada grupo no
    tiene ruido; (c) la ventaja es la Eq. 7 / L172-176 calculada a mano sobre los 2 grupos; (d) el gradiente del
    backward por grupo es el de (L_a + L_b) / 2 / acumulación en un solo grafo; (e) swap_adjacent_blocks falla con un
    solo grupo, que era el error original con 1 GPU.
    """
    from transformers import DataCollatorForSeq2Seq
    _reiniciar_accelerate()   # el trainer de V7 corre en la CPU; el estado de accelerate es global al proceso
    try:
        return _verificar_p3(base, categoria, tokenizer, salida_tmp, DataCollatorForSeq2Seq)
    finally:
        _reiniciar_accelerate()   # para que un entrenamiento posterior en el mismo kernel use la GPU
        shutil.rmtree(salida_tmp, ignore_errors=True)


def _reiniciar_accelerate():
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)


def _verificar_p3(base, categoria, tokenizer, salida_tmp, DataCollatorForSeq2Seq):
    tok = para_generar(tokenizer)
    modelo = _diminuto(len(tok)).train()
    congelar_llm(modelo)
    datos = dataset_rl(tok, buscar_archivo(base, "train", categoria), categoria, sample=2, seed=SEED)
    filas = [datos[0]] * NUM_GENERATIONS + [datos[1]] * NUM_GENERATIONS
    trainer = NoiseGRPORecTrainer(
        prefix_allowed_tokens_fn=lambda _, ids: [], model=modelo, reward_funcs=[recompensa_exact_match],
        args=config_rl(salida_tmp, bf16=False, use_cpu=True, max_steps=1, save_strategy="no"),
        train_dataset=a_hf(datos), eval_dataset=None, processing_class=tok)
    # (a)
    lote = trainer.data_collator(filas)
    colador = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    for g, grupo in enumerate(lote["grupos"]):
        solo = colador(filas[g * NUM_GENERATIONS:(g + 1) * NUM_GENERATIONS])
        assert all(torch.equal(grupo[k], solo[k]) for k in solo)
    # (b) y (c)
    salidas = trainer._generate_and_score_completions(lote)["grupos"]
    recompensas = []
    for s in salidas:
        assert torch.all(s["noise"][0] == 0) and torch.all(s["noise"][1:].abs().sum(-1) > 0)
        largo = s["completion_ids"].size(1)
        recompensas.append(-(-s["old_per_token_logps"].sum(-1) / largo).exp())
    S = torch.cat(recompensas)
    media_s1 = torch.stack([r[0] for r in recompensas]).mean()
    esperada = (S - media_s1) / (torch.norm(S - media_s1) + 1e-6)
    obtenida = torch.cat([s["advantages"] for s in salidas])
    assert torch.allclose(obtenida, esperada, atol=1e-5), (obtenida, esperada)
    # (d)
    trainer._prepare_inputs = lambda _: {"grupos": salidas}
    modelo.zero_grad()
    trainer.training_step(modelo, None)
    g_por_grupo = [p.grad.clone() for p in modelo.attention.parameters()]
    modelo.zero_grad()
    perdida = sum(trainer.compute_loss(modelo, s) for s in salidas) / (len(salidas) * trainer.args.gradient_accumulation_steps)
    perdida.backward()
    dif_grad = max((a - p.grad).abs().max().item() for a, p in zip(g_por_grupo, modelo.attention.parameters()))
    assert dif_grad < 1e-6, dif_grad
    # (e)
    try:
        swap_adjacent_blocks(torch.zeros(NUM_GENERATIONS), NUM_GENERATIONS)
        falla_con_un_grupo = False
    except RuntimeError:
        falla_con_un_grupo = True
    assert falla_con_un_grupo
    return {"grupos por micro-paso": len(salidas), "ventaja: dif. máx. vs. Eq. 7": (obtenida - esperada).abs().max().item(),
            "gradiente: dif. máx. por grupo vs. un grafo": dif_grad, "swap con 1 grupo falla": falla_con_un_grupo}


def verificar_sampler(n=1000):
    """V8: RepeatRandomSampler de TRL con el batch efectivo de 256 entrega, en cada micro-paso de 16 filas, dos
    grupos de 8 copias consecutivas: los mismos que las 2 GPUs de los autores (8 filas cada una) en ese micro-paso."""
    from trl.trainer.grpo_trainer import RepeatRandomSampler
    # 1 proceso: 16 × 16; 2 procesos: 8 × 2 × 16. En los dos casos el batch efectivo es 256 y TRL usa estos argumentos
    indices = list(RepeatRandomSampler(range(n), mini_repeat_count=NUM_GENERATIONS,
                                       batch_size=BATCH_RL // NUM_GENERATIONS, repeat_count=1, seed=SEED))
    micro = [indices[i:i + MICRO_RL] for i in range(0, len(indices), MICRO_RL)]
    for m in micro:
        a, b = m[:NUM_GENERATIONS], m[NUM_GENERATIONS:]
        assert len(set(a)) == 1 and len(set(b)) == 1   # cada mitad es un grupo (la GPU 0 y la GPU 1)
    return {"filas muestreadas": len(indices), "micro-pasos": len(micro),
            "pasos de optimización": len(indices) // BATCH_RL, "primer micro-paso": micro[0]}


# ---------- Estimación de horas ----------

def estimar(categoria, medicion=MEDICION_TOYS):
    """Horas por etapa de un dataset. El SFT y el RL se escalan por tokens por fila respecto de lo medido; la
    generación no se escala (depende más del largo de los títulos que del prompt)."""
    n_train, n_valid, n_test = FILAS[categoria]
    escala = TOKENS_POR_FILA[categoria] / medicion["tokens_por_fila"]
    epoca = (n_train / BATCH_SIZE * medicion["s_por_paso_sft"] + n_valid * medicion["s_valid_por_fila"]) * escala
    h = {f"h SFT ({EPOCAS} épocas)": EPOCAS * epoca / 3600,
         "h RL (1 época)": (n_train // (BATCH_RL // NUM_GENERATIONS)) * medicion["s_por_paso_rl"] * escala / 3600,
         "h evaluación (SFT + RL + D3)": n_test * (medicion["s_test_por_fila_sft"]
                                                  + 2 * medicion["s_test_por_fila_rl"]) / 3600}
    h["h total"] = sum(h.values())
    return h
