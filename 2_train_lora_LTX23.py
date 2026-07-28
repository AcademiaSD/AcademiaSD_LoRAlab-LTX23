# -*- coding: utf-8 -*-
"""
2_train_lora_LTX23.py

Trainer LoRA para LTX-2.3 (image-only / personaje) con:
- Descarga automática del modelo (aviso >100 GB).
- Pre-cache de texto y eliminación de connectors de VRAM.
- Padding de texto a múltiplo de learnable registers (128).
- Resume / stop / continue (signal handlers).
- Preview con seed / steps / CFG.
- Optimizaciones de VRAM (attention eficiente, grad checkpointing, cast bf16).
- EXPORT LoRA en formato estándar ComfyUI / Civitai:
    * prefijo configurable (default 'diffusion_model.'),
    * SIN el token '.default' del adapter de PEFT,
    * scaling (alpha/rank) 'horneado' en lora_B para que strength=1.0
      en ComfyUI reproduzca EXACTAMENTE el entrenamiento.
"""

import os
import platform

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("DIFFUSERS_NO_ADVISORY_WARNINGS", "1")

if platform.system() != "Windows":
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )
else:
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "garbage_collection_threshold:0.8",
    )

import gc
import math
import time
import random
import json
import signal
import sys
import inspect
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import DiffusionPipeline

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)

import bitsandbytes as bnb
from bitsandbytes.nn import (
    Linear4bit,
    Params4bit,
)

from safetensors import safe_open
from safetensors.torch import save_file, load_file

from PIL import Image


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================================
# CONFIG
# ============================================================================

DEFAULTS = {
    "model_id": "./LTX23-NF4",
    "cache_dir": "./cached_data_ltx23",
    "output_dir": "./ltx23_lora_output",
    "total_steps": 500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 8,
    "lora_alpha": 16,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 25,
    "seed": 42,
    "frame_rate": 24.0,
    "project_name": "",
    "trigger_word": "",

    # Optimización VRAM.
    "max_text_tokens": 256,
    "lora_only_attn": True,
    "cast_frozen_bf16": True,
    "use_audio_loss": False,

    # Preview.
    "preview_every": 0,
    "preview_steps": 8,
    "preview_cfg": 1.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "preview_vae_inverse_scale": False,

    # Formato de keys del LoRA exportado (ComfyUI / Civitai).
    # 'diffusion_model.' = loader nativo de ComfyUI.
    # 'transformer.'     = algunos nodos custom / formatos PEFT.
    # ''                 = sin prefijo.
    "lora_key_prefix": "diffusion_model.",
}

CONFIG_PATH = "train_settings.json"

HF_BASE_REPO_ID = "diffusers/LTX-2.3-Diffusers"
HF_NF4_REPO_ID = "AcademiaSD/LTX23_NF4"


if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuración cargada: {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] No existe {CONFIG_PATH}; usando valores por defecto.")


def cfg_get(key, default):
    if key in cfg:
        return cfg[key]
    if (key + " ") in cfg:
        return cfg[key + " "]
    return default


def _cfg_bool(key, default):
    value = cfg_get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")

    return bool(value)


MODEL_ID = str(cfg_get("model_id", DEFAULTS["model_id"])).strip()
TOTAL_STEPS = int(cfg_get("total_steps", DEFAULTS["total_steps"]))
BATCH_SIZE = int(cfg_get("batch_size", DEFAULTS["batch_size"]))
GRAD_ACCUM_STEPS = int(cfg_get("grad_accum_steps", DEFAULTS["grad_accum_steps"]))
LR = float(cfg_get("lr", DEFAULTS["lr"]))
MIN_LR_RATIO = float(cfg_get("min_lr_ratio", DEFAULTS["min_lr_ratio"]))
WARMUP_STEPS = int(cfg_get("warmup_steps", DEFAULTS["warmup_steps"]))
LORA_RANK = int(cfg_get("lora_rank", DEFAULTS["lora_rank"]))
LORA_ALPHA = int(cfg_get("lora_alpha", DEFAULTS["lora_alpha"]))
WEIGHT_DECAY = float(cfg_get("weight_decay", DEFAULTS["weight_decay"]))
MAX_GRAD_NORM = float(cfg_get("max_grad_norm", DEFAULTS["max_grad_norm"]))
SAVE_EVERY = int(cfg_get("save_every", DEFAULTS["save_every"]))
SEED = int(cfg_get("seed", DEFAULTS["seed"]))
FRAME_RATE = float(cfg_get("frame_rate", DEFAULTS["frame_rate"]))
TRIGGER_WORD = str(cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
PROJECT_NAME = str(cfg_get("project_name", DEFAULTS["project_name"])).strip()

MAX_TEXT_TOKENS = int(cfg_get("max_text_tokens", DEFAULTS["max_text_tokens"]) or 0)
LORA_ONLY_ATTN = _cfg_bool("lora_only_attn", DEFAULTS["lora_only_attn"])
CAST_FROZEN_BF16 = _cfg_bool("cast_frozen_bf16", DEFAULTS["cast_frozen_bf16"])
USE_AUDIO_LOSS = _cfg_bool("use_audio_loss", DEFAULTS["use_audio_loss"])

PREVIEW_EVERY = int(cfg_get("preview_every", DEFAULTS["preview_every"]))
PREVIEW_STEPS = int(cfg_get("preview_steps", DEFAULTS["preview_steps"]))
PREVIEW_CFG = float(cfg_get("preview_cfg", DEFAULTS["preview_cfg"]))
PREVIEW_CAPTION_MODE = str(cfg_get("preview_caption_mode", DEFAULTS["preview_caption_mode"])).strip().lower()
PREVIEW_CUSTOM_PROMPT = str(cfg_get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip()
PREVIEW_VAE_INVERSE_SCALE = _cfg_bool("preview_vae_inverse_scale", DEFAULTS["preview_vae_inverse_scale"])

LORA_KEY_PREFIX = str(cfg_get("lora_key_prefix", DEFAULTS["lora_key_prefix"]))


if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_ltx23_{PROJECT_NAME}"
    OUTPUT_DIR = f"./ltx23_lora_output_{PROJECT_NAME}"
else:
    CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()
    OUTPUT_DIR = str(cfg_get("output_dir", DEFAULTS["output_dir"])).strip()


os.makedirs(OUTPUT_DIR, exist_ok=True)

RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE = os.path.join(OUTPUT_DIR, "current_step.txt")


# ============================================================================
# BANNER
# ============================================================================

print()
print("=" * 80)
print(" LTX-2.3 LoRA TRAINER")
print("=" * 80)
print(f"  Model ID / ID Modelo        : {MODEL_ID}")
print(f"  Base Repo                   : {HF_BASE_REPO_ID}")
print(f"  NF4 Repo                    : {HF_NF4_REPO_ID}")
print(f"  Project / Proyecto          : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra      : {TRIGGER_WORD}")
print(f"  Cache Dir / Carpeta Caché   : {CACHE_DIR}")
print(f"  Output Dir / Salida         : {OUTPUT_DIR}")
print(f"  Total Steps / Pasos         : {TOTAL_STEPS}")
print(f"  Learning Rate / LR          : {LR}")
print(f"  LoRA Rank/Alpha             : {LORA_RANK}/{LORA_ALPHA}")
print(f"  Batch / Grad Accum          : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  Max Text Tokens             : {MAX_TEXT_TOKENS}")
print(f"  LoRA Only Attention         : {'ON' if LORA_ONLY_ATTN else 'OFF'}")
print(f"  Use Audio Loss              : {'ON' if USE_AUDIO_LOSS else 'OFF'}")
print(f"  Preview Mode / Prompt       : Mode={PREVIEW_CAPTION_MODE} | Custom='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  Preview Every / Steps / CFG : {PREVIEW_EVERY} / {PREVIEW_STEPS} / {PREVIEW_CFG}")
print(f"  LoRA Key Prefix / Prefijo   : '{LORA_KEY_PREFIX}'")
print(f"  Seed Configured / Semilla   : {SEED} ({'RANDOM' if SEED <= 0 else 'FIXED'})")
print("=" * 80)


# ============================================================================
# DESCARGA DEL MODELO
# ============================================================================

def get_hf_token():
    if os.path.exists("HF_token.json"):
        try:
            with open("HF_token.json", "r", encoding="utf-8") as f:
                token_data = json.load(f)

            token = token_data.get("token", "").strip()

            if token:
                return token

        except Exception:
            pass

    token = os.environ.get("HF_TOKEN", "").strip()

    if token:
        return token

    return None


def ensure_ltx23_model_downloaded(local_path):
    local_path = str(local_path or "./LTX23-NF4")

    has_base = os.path.exists(os.path.join(local_path, "model_index.json"))
    has_nf4 = os.path.exists(os.path.join(local_path, "index.json"))

    if has_base and has_nf4:
        print(f"[OK] Modelo local encontrado en / Local model found at: {local_path}")
        return local_path

    print()
    print("=" * 80)
    print("WARNING / ATENCIÓN")
    print("=" * 80)
    print("This will download more than 100 GB. This may take several minutes.")
    print("Esto descargará más de 100 GB. Esto puede tardar varios minutos.")
    print("=" * 80)

    auto = os.environ.get("LTX_AUTO_CONFIRM_DOWNLOAD", "0").strip().lower()

    if auto not in ("1", "true", "yes", "y", "on"):
        try:
            input("Press Enter to continue / Pulsa Enter para continuar...")
        except Exception:
            pass

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    token = get_hf_token()

    if token:
        print("✓ Using HF Token / Usando token de HF")

    os.makedirs(local_path, exist_ok=True)

    print()
    print("Downloading / Descargando:", HF_BASE_REPO_ID)

    snapshot_download(
        repo_id=HF_BASE_REPO_ID,
        local_dir=local_path,
        token=token,
        max_workers=4,
    )

    print()
    print("Downloading / Descargando:", HF_NF4_REPO_ID)

    snapshot_download(
        repo_id=HF_NF4_REPO_ID,
        local_dir=local_path,
        token=token,
        max_workers=4,
    )

    print()
    print(f"[OK] Modelo descargado en / Model downloaded to: {local_path}")

    return local_path


# ============================================================================
# UTILIDADES
# ============================================================================

def free_vram(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def pin_cpu_tensor(t):
    if (
        torch.cuda.is_available()
        and torch.is_tensor(t)
        and t.device.type == "cpu"
    ):
        try:
            if not t.is_pinned():
                return t.pin_memory()
        except Exception:
            pass

    return t


def get_parent_module(root, name):
    parts = name.split(".")
    parent = root

    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


def _round_to_multiple(x, multiple):
    x = int(x)
    multiple = max(1, int(multiple))

    return max(
        multiple,
        ((x + multiple - 1) // multiple) * multiple,
    )


# ============================================================================
# NF4 CACHE
# ============================================================================

def load_nf4_cache_(transformer, cache_dir):
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No existe index.json: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    unquantized = index.get("unquantized", {})
    weights_dir = os.path.join(cache_dir, "weights")

    replaced = 0

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No existe peso NF4: {filepath}")

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias_data = None

            if info.get("bias", False):
                bias_data = f.get_tensor("bias")

            qs_dict = {}

            for key in f.keys():
                if key.startswith("quant_state."):
                    qs_dict[key[len("quant_state."):]] = f.get_tensor(key)

        packed_qs = {}

        for key, value in qs_dict.items():
            packed_qs[key] = value

        new_layer = Linear4bit(
            int(info["in_features"]),
            int(info["out_features"]),
            bias=info.get("bias", False),
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )

        new_weight = Params4bit.from_prequantized(
            data=weight_data,
            quantized_stats=packed_qs,
            requires_grad=False,
            device="cuda",
            module=new_layer,
        )

        new_layer.weight = new_weight

        if bias_data is not None:
            new_layer.bias = torch.nn.Parameter(
                bias_data.to("cuda", dtype=torch.bfloat16),
                requires_grad=False,
            )

        setattr(parent, child_name, new_layer)
        replaced += 1

    for name, info in unquantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")
            bias = None

            if info.get("bias", False):
                bias = f.get_tensor("bias")

        layer = torch.nn.Linear(
            int(info["in_features"]),
            int(info["out_features"]),
            bias=info.get("bias", False),
        )

        layer.weight = torch.nn.Parameter(weight, requires_grad=False)

        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        setattr(parent, child_name, layer)

    verified = 0

    for _, module in transformer.named_modules():
        if isinstance(module, Linear4bit):
            if (
                getattr(module.weight, "bnb_quantized", False)
                and getattr(module.weight, "quant_state", None) is not None
            ):
                verified += 1

    print(f"Capas NF4 reconstruidas: {replaced}")
    print(f"Capas NF4 verificadas: {verified}")

    if verified != replaced:
        raise RuntimeError("La verificación NF4 no coincide.")

    print("[OK] Caché NF4 cargada correctamente.")

    return transformer


# ============================================================================
# LoRA TARGETS
# ============================================================================

def discover_lora_targets(transformer):
    targets = []
    only_attn = globals().get("LORA_ONLY_ATTN", True)

    for name, module in transformer.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue

        if not name:
            continue

        parts = name.split(".")

        excluded_parts = {
            "audio_attn1",
            "audio_attn2",
            "audio_ff",
            "audio_to_video_attn",
            "video_to_audio_attn",
        }

        if any(part in excluded_parts for part in parts):
            continue

        if any(part.startswith("audio_") for part in parts):
            continue

        if "transformer_blocks" not in parts:
            continue

        if only_attn:
            attn_markers = (
                "attn1",
                "attn2",
                "to_q",
                "to_k",
                "to_v",
                "to_out",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            )

            if not any(marker in name for marker in attn_markers):
                continue

        targets.append(name)

    targets = list(dict.fromkeys(targets))

    if not targets:
        raise RuntimeError("No se encontraron módulos Linear4bit visuales para LoRA.")

    return targets


# ============================================================================
# PROMPT / TEXT CONDITIONING
# ============================================================================

def load_prompt_structure(cache_dir, prefix):
    path = os.path.join(cache_dir, f"{prefix}_structure.json")

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    def recurse(node):
        if node["type"] == "tensor":
            return torch.load(
                os.path.join(cache_dir, node["file"]),
                map_location="cpu",
                weights_only=True,
            )

        if node["type"] == "dict":
            return {k: recurse(v) for k, v in node["items"].items()}

        if node["type"] == "tuple":
            return tuple(recurse(v) for v in node["items"])

        if node["type"] == "list":
            return [recurse(v) for v in node["items"]]

        return node.get("value")

    return recurse(structure)


def flatten_tensors(obj, prefix="root"):
    result = []

    if torch.is_tensor(obj):
        result.append((prefix, obj))

    elif isinstance(obj, dict):
        for k, v in obj.items():
            result.extend(flatten_tensors(v, f"{prefix}.{k}"))

    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            result.extend(flatten_tensors(v, f"{prefix}.{i}"))

    return result


def get_prompt_pair(result):
    tensors = flatten_tensors(result)

    if not tensors:
        raise RuntimeError("La caché de prompt no contiene tensores.")

    if (
        isinstance(result, (tuple, list))
        and len(result) >= 2
        and torch.is_tensor(result[0])
    ):
        return (
            result[0],
            result[1] if torch.is_tensor(result[1]) else None,
        )

    return (
        tensors[0][1],
        tensors[1][1] if len(tensors) > 1 else None,
    )


def valid_token_slice(mask):
    if mask is None or mask.ndim != 2:
        return None

    if mask.shape[0] != 1:
        return None

    m = mask[0].to(torch.int64)
    valid = int(m.sum().item())

    if valid <= 0:
        return slice(0, 1)

    if valid >= m.numel():
        return slice(None)

    if int(m[:valid].sum().item()) == valid:
        return slice(0, valid)

    if int(m[-valid:].sum().item()) == valid:
        return slice(-valid, None)

    idx = torch.nonzero(m, as_tuple=False).reshape(-1)

    if idx.numel() == valid and int(idx[-1] - idx[0] + 1) == valid:
        return slice(int(idx[0].item()), int(idx[-1].item()) + 1)

    return None


def get_text_cache_paths(cache_dir, base, max_text_tokens):
    tag = f"mt{int(max_text_tokens or 0)}_reg128_v2"

    video_text_path = os.path.join(cache_dir, f"{base}_video_text_{tag}.pt")
    audio_text_path = os.path.join(cache_dir, f"{base}_audio_text_{tag}.pt")

    return video_text_path, audio_text_path


def run_text_connectors(prompt_result, connectors, max_text_tokens=0):
    if connectors is None:
        raise RuntimeError("No hay text connectors cargados.")

    embeds, mask = get_prompt_pair(prompt_result)

    embeds = embeds.to("cuda", dtype=torch.bfloat16)

    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)

    if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.int64, device="cuda")
    else:
        mask = mask.to("cuda")

    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    if embeds.ndim != 3:
        raise RuntimeError("prompt_embeds debe tener forma [B, S, D].")

    if mask.ndim != 2:
        raise RuntimeError("prompt_attention_mask debe tener forma [B, S].")

    B, S, D = embeds.shape
    max_text_tokens = int(max_text_tokens or 0)

    register_multiple = 128

    for obj in (
        connectors,
        getattr(connectors, "video_connector", None),
        getattr(connectors, "audio_connector", None),
    ):
        if obj is None:
            continue

        for attr in (
            "num_learnable_registers",
            "num_registers",
            "num_register_tokens",
            "registers",
            "n_registers",
        ):
            val = getattr(obj, attr, None)

            if isinstance(val, int) and val > 0:
                register_multiple = val
                break

        if register_multiple != 128:
            break

    if B != 1:
        target_len = S

        if max_text_tokens > 0:
            target_len = min(target_len, max_text_tokens)

        target_len = _round_to_multiple(target_len, register_multiple)

        if target_len < S:
            embeds = embeds[:, :target_len, :]
            mask = mask[:, :target_len]
            S = target_len

        if target_len > S:
            new_embeds = torch.zeros((B, target_len, D), device=embeds.device, dtype=embeds.dtype)
            new_mask = torch.zeros((B, target_len), device=mask.device, dtype=mask.dtype)

            new_embeds[:, -S:, :] = embeds
            new_mask[:, -S:] = mask

            embeds = new_embeds
            mask = new_mask

        with torch.no_grad():
            out = connectors(embeds, mask, padding_side="left")

        if isinstance(out, (tuple, list)):
            video_text = out[0]
            audio_text = out[1]
        else:
            video_text = getattr(out, "video_text", None)

            if video_text is None:
                video_text = getattr(out, "video_embeds", None)

            if video_text is None:
                video_text = getattr(out, "video", None)

            audio_text = getattr(out, "audio_text", None)

            if audio_text is None:
                audio_text = getattr(out, "audio_embeds", None)

            if audio_text is None:
                audio_text = getattr(out, "audio", None)

        if video_text is None or audio_text is None:
            raise RuntimeError("connectors() no devolvió video_text/audio_text.")

        return video_text, audio_text

    sl = valid_token_slice(mask)

    if sl is not None:
        embeds = embeds[:, sl, :]
        mask = mask[:, sl]

    valid_len = int(embeds.shape[1])

    if valid_len <= 0:
        embeds = torch.zeros((1, 1, D), device=embeds.device, dtype=embeds.dtype)
        mask = torch.zeros((1, 1), device=mask.device, dtype=mask.dtype)
        valid_len = 1

    target_len = valid_len

    if max_text_tokens > 0:
        target_len = min(valid_len, max_text_tokens)

    target_len = _round_to_multiple(target_len, register_multiple)

    if target_len < valid_len:
        embeds = embeds[:, :target_len, :]
        mask = mask[:, :target_len]
        valid_len = target_len

    if target_len > valid_len:
        pad_len = target_len - valid_len

        new_embeds = torch.zeros((1, target_len, D), device=embeds.device, dtype=embeds.dtype)
        new_mask = torch.zeros((1, target_len), device=mask.device, dtype=mask.dtype)

        new_embeds[:, pad_len:, :] = embeds[:, :valid_len, :]
        new_mask[:, pad_len:] = mask[:, :valid_len]

        embeds = new_embeds
        mask = new_mask
    else:
        embeds = embeds.contiguous()
        mask = mask.contiguous()

    with torch.no_grad():
        out = connectors(embeds, mask, padding_side="left")

    if isinstance(out, (tuple, list)):
        video_text = out[0]
        audio_text = out[1]
    else:
        video_text = getattr(out, "video_text", None)

        if video_text is None:
            video_text = getattr(out, "video_embeds", None)

        if video_text is None:
            video_text = getattr(out, "video", None)

        audio_text = getattr(out, "audio_text", None)

        if audio_text is None:
            audio_text = getattr(out, "audio_embeds", None)

        if audio_text is None:
            audio_text = getattr(out, "audio", None)

    if video_text is None or audio_text is None:
        raise RuntimeError("connectors() no devolvió video_text/audio_text.")

    return video_text, audio_text


def prepare_text_conditioning(entries, connectors, max_text_tokens=0):
    max_text_tokens = int(max_text_tokens or 0)

    for entry in entries:
        base = entry["name"]

        video_text_path, audio_text_path = get_text_cache_paths(
            CACHE_DIR,
            base,
            max_text_tokens,
        )

        entry["_video_text_path"] = video_text_path
        entry["_audio_text_path"] = audio_text_path

    missing = [
        entry
        for entry in entries
        if not (
            os.path.exists(entry["_video_text_path"])
            and os.path.exists(entry["_audio_text_path"])
        )
    ]

    if missing:
        if connectors is None:
            raise RuntimeError("Faltan textos precomputados y no hay connectors.")

        print()
        print(f"Precomputando text conditioning para {len(missing)} entradas...")

        connectors.to("cuda", dtype=torch.bfloat16)
        connectors.eval()

        for param in connectors.parameters():
            param.requires_grad_(False)

        for entry in missing:
            base = entry["name"]

            if entry.get("prompt", None) is None:
                entry["prompt"] = load_prompt_structure(CACHE_DIR, f"{base}_prompt")

            if entry.get("prompt", None) is None:
                raise RuntimeError(f"No hay prompt cacheado para {base}.")

            video_text, audio_text = run_text_connectors(
                entry["prompt"],
                connectors,
                max_text_tokens=max_text_tokens,
            )

            video_text = video_text.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            audio_text = audio_text.detach().to("cpu", dtype=torch.bfloat16).contiguous()

            torch.save(video_text, entry["_video_text_path"])
            torch.save(audio_text, entry["_audio_text_path"])

            free_vram(video_text, audio_text)

        connectors.to("cpu")
        free_vram()

    for entry in entries:
        entry["video_text"] = pin_cpu_tensor(
            torch.load(entry["_video_text_path"], map_location="cpu", weights_only=True).to(torch.bfloat16)
        )

        entry["audio_text"] = pin_cpu_tensor(
            torch.load(entry["_audio_text_path"], map_location="cpu", weights_only=True).to(torch.bfloat16)
        )

        entry.pop("prompt", None)


def prepare_special_text_conditioning(connectors, max_text_tokens, preview_custom_prompt):
    special = {}
    prefixes = []

    neg_path = os.path.join(CACHE_DIR, "_neg_structure.json")
    custom_path = os.path.join(CACHE_DIR, "_custom_structure.json")

    if os.path.exists(neg_path):
        prefixes.append("_neg")

    if preview_custom_prompt and os.path.exists(custom_path):
        prefixes.append("_custom")

    if not prefixes:
        return special

    max_text_tokens = int(max_text_tokens or 0)

    paths = {
        prefix: get_text_cache_paths(CACHE_DIR, prefix, max_text_tokens)
        for prefix in prefixes
    }

    missing = [
        prefix
        for prefix in prefixes
        if not (
            os.path.exists(paths[prefix][0])
            and os.path.exists(paths[prefix][1])
        )
    ]

    if missing and connectors is not None:
        print()
        print("Precomputando textos especiales para preview:")

        for prefix in missing:
            print(f"  - {prefix}")

        connectors.to("cuda", dtype=torch.bfloat16)
        connectors.eval()

        for param in connectors.parameters():
            param.requires_grad_(False)

        for prefix in missing:
            prompt_result = load_prompt_structure(CACHE_DIR, prefix)

            if prompt_result is None:
                continue

            video_text, audio_text = run_text_connectors(
                prompt_result,
                connectors,
                max_text_tokens=max_text_tokens,
            )

            video_text = video_text.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            audio_text = audio_text.detach().to("cpu", dtype=torch.bfloat16).contiguous()

            torch.save(video_text, paths[prefix][0])
            torch.save(audio_text, paths[prefix][1])

            free_vram(video_text, audio_text)

        connectors.to("cpu")
        free_vram()

    for prefix in prefixes:
        video_text_path, audio_text_path = paths[prefix]

        if os.path.exists(video_text_path) and os.path.exists(audio_text_path):
            special[prefix] = (
                pin_cpu_tensor(
                    torch.load(video_text_path, map_location="cpu", weights_only=True).to(torch.bfloat16)
                ),
                pin_cpu_tensor(
                    torch.load(audio_text_path, map_location="cpu", weights_only=True).to(torch.bfloat16)
                ),
            )

    return special


def load_cached_entries(cache_dir, audio_channels, max_text_tokens=0):
    entries = []

    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith("_video_latent.pt"):
            continue

        base = filename[:-len("_video_latent.pt")]

        video_path = os.path.join(cache_dir, filename)
        audio_path = os.path.join(cache_dir, f"{base}_audio_latent.pt")

        if not os.path.exists(audio_path):
            continue

        video_text_path, audio_text_path = get_text_cache_paths(
            cache_dir,
            base,
            max_text_tokens,
        )

        prompt_result = None

        if not (
            os.path.exists(video_text_path)
            and os.path.exists(audio_text_path)
        ):
            prompt_result = load_prompt_structure(cache_dir, f"{base}_prompt")

            if prompt_result is None:
                continue

        video_latent = torch.load(video_path, map_location="cpu", weights_only=True)

        if video_latent is None:
            continue

        video_latent = pin_cpu_tensor(video_latent.to(torch.bfloat16))

        audio_latent_raw = torch.load(audio_path, map_location="cpu", weights_only=True)

        if audio_latent_raw is None:
            audio_latent = torch.zeros(
                (video_latent.shape[0], audio_channels, 1),
                dtype=torch.bfloat16,
            )
        else:
            audio_latent = audio_latent_raw.to(torch.bfloat16)

        audio_latent = pin_cpu_tensor(audio_latent)

        entries.append(
            {
                "name": base,
                "video": video_latent,
                "audio": audio_latent,
                "prompt": prompt_result,
                "_video_text_path": video_text_path,
                "_audio_text_path": audio_text_path,
            }
        )

    if not entries:
        raise RuntimeError("No se encontraron entradas válidas.")

    return entries


# ============================================================================
# PATCHIFY / TIMESTEP / LOSS
# ============================================================================

def patch_video_latent(latent, patch_size=1, patch_size_t=1):
    if latent.ndim != 5:
        raise RuntimeError("Video latent esperado [B,C,F,H,W].")

    B, C, Fm, H, W = latent.shape

    patch_size = max(1, int(patch_size))
    patch_size_t = max(1, int(patch_size_t))

    if Fm % patch_size_t != 0:
        Fm = (Fm // patch_size_t) * patch_size_t

    if H % patch_size != 0:
        H = (H // patch_size) * patch_size

    if W % patch_size != 0:
        W = (W // patch_size) * patch_size

    latent = latent[:, :, :Fm, :H, :W]

    x = latent.view(
        B,
        C,
        Fm // patch_size_t,
        patch_size_t,
        H // patch_size,
        patch_size,
        W // patch_size,
        patch_size,
    )

    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)

    return x.reshape(B, -1, C * patch_size_t * patch_size * patch_size)


def patch_audio_latent(latent):
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    if latent.ndim != 3:
        raise RuntimeError("Audio latent esperado [B,C,T] o [C,T].")

    return latent.transpose(1, 2).contiguous()


def unpack_video_latent(tokens, latent_shape, patch_size=1, patch_size_t=1):
    B, C, Fm, H, W = tuple(latent_shape)

    pt = max(1, int(patch_size_t))
    p = max(1, int(patch_size))

    Fp = Fm // pt
    Hp = H // p
    Wp = W // p

    expected_seq = Fp * Hp * Wp

    tokens = tokens[:, :expected_seq, :]

    x = tokens.view(B, Fp, Hp, Wp, C, pt, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)

    return x.reshape(B, C, Fp * pt, Hp * p, Wp * p)


def make_video_timestep(sigma, seq_len, device, dtype):
    multiplier = float(getattr(CURRENT_CONFIG, "timestep_scale_multiplier", 1000))

    return (
        sigma.view(-1, 1)
        .expand(-1, seq_len)
        * multiplier
    ).to(device=device, dtype=dtype)


def mse_loss_chunked(pred, target, chunk_elements=2_000_000):
    if pred.numel() == 0:
        return pred.new_zeros((), dtype=torch.float32)

    if pred.numel() <= chunk_elements:
        return F.mse_loss(pred.float(), target.float())

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    n = pred_flat.numel()

    loss_sum = torch.zeros((), device=pred.device, dtype=torch.float32)

    for start in range(0, n, chunk_elements):
        end = min(start + chunk_elements, n)

        p = pred_flat[start:end].float()
        t = target_flat[start:end].float()

        loss_sum = loss_sum + F.mse_loss(p, t, reduction="sum")

        del p, t

    return loss_sum / float(n)


# ============================================================================
# OPTIMIZACIONES
# ============================================================================

def enable_memory_efficient_attention(transformer):
    try:
        transformer.enable_xformers_memory_efficient_attention()
        return
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass


def enable_gradient_checkpointing_safe(transformer, model):
    for obj in (model, transformer):
        if hasattr(obj, "enable_gradient_checkpointing"):
            try:
                obj.enable_gradient_checkpointing()
                return
            except Exception:
                pass

    try:
        if hasattr(transformer, "gradient_checkpointing"):
            transformer.gradient_checkpointing = True
    except Exception:
        pass


def cast_frozen_to_bf16(root):
    for name, param in root.named_parameters():
        if isinstance(param, Params4bit):
            continue

        if param.requires_grad:
            continue

        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    for name, buf in root.named_buffers():
        if buf.is_floating_point() and buf.dtype != torch.bfloat16:
            lower = name.lower()

            if any(k in lower for k in ("norm", "ln", "layernorm")):
                continue

            buf.data = buf.data.to(torch.bfloat16)


# ============================================================================
# EXPORT LoRA (formato estándar ComfyUI / Civitai)
# ============================================================================

def save_lora(model, path, prefix=None):
    """
    Exporta el LoRA en formato estándar ComfyUI / Civitai:
      - prefijo configurable (default 'diffusion_model.'),
      - SIN el token '.default' del adapter de PEFT,
      - scaling (alpha/rank) 'horneado' en lora_B para que strength=1.0
        en ComfyUI reproduzca EXACTAMENTE el entrenamiento
        (ni el loader nativo ni los hooks de inyección aplican alpha/rank).

    Antes se guardaba 'transformer.X.lora_A.default.weight' y ComfyUI
    no encontraba NINGUNA key -> 'lora key not loaded' en todas.
    """
    if prefix is None:
        prefix = LORA_KEY_PREFIX
    if prefix is None:
        prefix = ""

    scaling = float(LORA_ALPHA) / float(max(1, LORA_RANK))

    state = {}

    for name, tensor in model.state_dict().items():
        if "lora_" not in name:
            continue

        # base_model.model.transformer_blocks.0.attn1.to_q.lora_A.default.weight
        clean = name.replace("base_model.model.", "")
        clean = clean.replace(".default.", ".")          # quita adapter PEFT

        t = tensor.detach().to(torch.float32).cpu()
        if ".lora_B." in clean:                          # hornea el scaling
            t = t * scaling
        t = t.to(torch.bfloat16).contiguous()

        state[prefix + clean] = t

    save_file(
        state,
        path,
        metadata={
            "format": "ltx23_lora",
            "lora_key_prefix": prefix,
            "baked_scaling": f"{scaling:.6f}",
        },
    )


# ============================================================================
# PREVIEW
# ============================================================================

class LTXVaeHolder:
    vae = None

    @classmethod
    def get(cls):
        if cls.vae is None:
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                transformer=None,
                text_encoder=None,
                audio_vae=None,
                tokenizer=None,
                processor=None,
                vocoder=None,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

            cls.vae = pipe.vae

            if cls.vae is None:
                raise RuntimeError("No se pudo obtener pipe.vae para preview.")

            cls.vae.requires_grad_(False)
            cls.vae.eval()

            del pipe
            gc.collect()

        return cls.vae


def set_scheduler_timesteps_for_preview(scheduler, steps, seq_len, device):
    kwargs = {"device": device}

    try:
        sig = inspect.signature(scheduler.set_timesteps)

        if "mu" in sig.parameters:
            base_seq = float(getattr(scheduler.config, "base_image_seq_len", 256))
            max_seq = float(getattr(scheduler.config, "max_image_seq_len", 6400))
            base_shift = float(getattr(scheduler.config, "base_shift", 0.5))
            max_shift = float(getattr(scheduler.config, "max_shift", 1.15))

            if max_seq > base_seq:
                m = (max_shift - base_shift) / (max_seq - base_seq)
                b = base_shift - m * base_seq
                kwargs["mu"] = float(seq_len) * m + b

    except Exception:
        pass

    try:
        scheduler.set_timesteps(steps, **kwargs)
    except TypeError:
        scheduler.set_timesteps(steps, device=device)


def preview_timestep_tensors(t, seq_len, device, dtype):
    multiplier = float(getattr(CURRENT_CONFIG, "timestep_scale_multiplier", 1000))

    if torch.is_tensor(t):
        t_val = float(t.item())
    else:
        t_val = float(t)

    if abs(t_val) > 10.0:
        sigma_val = t_val / multiplier
        timestep_val = t_val
    else:
        sigma_val = t_val
        timestep_val = t_val * multiplier

    sigma_val = min(max(sigma_val, 1e-4), 1.0)

    timestep = torch.full((1, int(seq_len)), timestep_val, device=device, dtype=dtype)
    audio_timestep = torch.tensor([timestep_val], device=device, dtype=dtype)
    sigma = torch.tensor([sigma_val], device=device, dtype=torch.float32)

    return timestep, audio_timestep, sigma


def preview_forward(
    model,
    video_tokens,
    audio_tokens,
    video_text,
    audio_text,
    t,
    latent_shape,
    audio_channels,
):
    B, C, Fm, H, W = tuple(latent_shape)

    timestep, audio_timestep, sigma = preview_timestep_tensors(
        t,
        video_tokens.shape[1],
        video_tokens.device,
        torch.bfloat16,
    )

    forward_kwargs = {
        "hidden_states": video_tokens,
        "audio_hidden_states": audio_tokens,
        "encoder_hidden_states": video_text,
        "audio_encoder_hidden_states": audio_text,
        "timestep": timestep,
        "audio_timestep": audio_timestep,
        "sigma": sigma,
        "audio_sigma": sigma,
        "num_frames": Fm,
        "height": H,
        "width": W,
        "fps": FRAME_RATE,
        "audio_num_frames": audio_tokens.shape[1],
        "return_dict": False,
    }

    if CURRENT_TRANSFORMER is not None:
        signature = inspect.signature(CURRENT_TRANSFORMER.forward)
    else:
        signature = inspect.signature(model.forward)

    forward_kwargs = {
        k: v
        for k, v in forward_kwargs.items()
        if k in signature.parameters
    }

    output = model(**forward_kwargs)

    if isinstance(output, tuple):
        if len(output) == 0:
            raise RuntimeError("Preview: forward devolvió tuple vacía.")

        pred_video = output[0]

    else:
        pred_video = getattr(output, "video", None)

        if pred_video is None:
            pred_video = getattr(output, "sample", None)

    if pred_video is None:
        raise RuntimeError("Preview: no se pudo obtener predicción de vídeo.")

    return pred_video


def decode_preview_latent(vae, latent):
    latent = latent.detach().to("cuda", dtype=vae.dtype)

    # IMPORTANTE: latents_mean/latents_std son BUFFERS del propio VAE
    # (vae.latents_mean / vae.latents_std), NO viven en vae.config.
    # El check anterior (hasattr(vae.config, "latents_mean")) nunca
    # era verdadero, así que la desnormalización real nunca se
    # aplicaba (y el toggle PREVIEW_VAE_INVERSE_SCALE tampoco ayudaba,
    # porque además faltaba dividir por scaling_factor). Se aplica
    # aquí SIEMPRE, igual que hace el pipeline oficial al decodificar:
    #
    #   latents = latents * latents_std / scaling_factor + latents_mean
    #
    # (ver diffusers/pipelines/ltx2/pipeline_ltx2.py: _denormalize_latents).
    latents_mean = vae.latents_mean.to(
        device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1, 1)

    latents_std = vae.latents_std.to(
        device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1, 1)

    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

    latent = latent * latents_std / scaling_factor + latents_mean

    with torch.inference_mode():
        decoded = vae.decode(latent, return_dict=False)[0]

    if decoded.ndim == 5:
        decoded = decoded[:, :, 0]

    decoded = decoded[:, :3].detach().float()

    img = (
        (decoded / 2 + 0.5)
        .clamp(0, 1)[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).astype("uint8")

    return img


def run_preview_ltx(
    model,
    scheduler,
    entries,
    special_texts,
    step,
    audio_channels,
):
    if scheduler is None or not entries:
        return

    was_training = model.training
    model.eval()

    vae = None

    try:
        valid_entries = [e for e in entries if not e["name"].startswith("_")]

        if not valid_entries:
            valid_entries = entries

        if PREVIEW_CAPTION_MODE == "random":
            entry = random.choice(valid_entries)

        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(valid_entries))
            entry = valid_entries[idx]

        else:
            entry = valid_entries[0]

        if PREVIEW_CAPTION_MODE == "custom" and "_custom" in special_texts:
            video_text, audio_text = special_texts["_custom"]
            sample_name = "_custom"
        else:
            video_text = entry["video_text"]
            audio_text = entry["audio_text"]
            sample_name = entry["name"]

        latent_shape = tuple(entry["video"].shape)
        latent_shape = (1,) + latent_shape[1:]

        device = "cuda"

        video_text = video_text.to(device, dtype=torch.bfloat16)
        audio_text = audio_text.to(device, dtype=torch.bfloat16)

        if video_text.ndim == 2:
            video_text = video_text.unsqueeze(0)

        if audio_text.ndim == 2:
            audio_text = audio_text.unsqueeze(0)

        neg_video_text = None
        neg_audio_text = None

        if PREVIEW_CFG > 1.0 and "_neg" in special_texts:
            neg_video_text, neg_audio_text = special_texts["_neg"]

            neg_video_text = neg_video_text.to(device, dtype=torch.bfloat16)
            neg_audio_text = neg_audio_text.to(device, dtype=torch.bfloat16)

            if neg_video_text.ndim == 2:
                neg_video_text = neg_video_text.unsqueeze(0)

            if neg_audio_text.ndim == 2:
                neg_audio_text = neg_audio_text.unsqueeze(0)

        if SEED > 0:
            preview_seed = SEED
        else:
            preview_seed = random.randint(1, 2147483647)

        print()
        print(f"  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {sample_name}")
        print(f"  ↳ Preview Seed used / Semilla utilizada: {preview_seed}")
        print(f"  ↳ Preview Steps / Pasos: {PREVIEW_STEPS}")
        print(f"  ↳ Preview CFG: {PREVIEW_CFG}")

        generator = torch.Generator(device=device).manual_seed(preview_seed)

        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        audio_latent = torch.zeros(
            (1, int(audio_channels), 1),
            device=device,
            dtype=torch.bfloat16,
        )

        patch_size = int(getattr(CURRENT_CONFIG, "patch_size", 1))
        patch_size_t = int(getattr(CURRENT_CONFIG, "patch_size_t", 1))

        video_tokens = patch_video_latent(latents, patch_size, patch_size_t)
        seq_len = video_tokens.shape[1]

        set_scheduler_timesteps_for_preview(scheduler, PREVIEW_STEPS, seq_len, device)

        if len(scheduler.timesteps) == 0:
            return

        with torch.inference_mode():
            for t in scheduler.timesteps:
                video_tokens = patch_video_latent(latents, patch_size, patch_size_t)
                audio_tokens = patch_audio_latent(audio_latent)

                pred_video = preview_forward(
                    model,
                    video_tokens,
                    audio_tokens,
                    video_text,
                    audio_text,
                    t,
                    latent_shape,
                    audio_channels,
                )

                if neg_video_text is not None:
                    pred_neg = preview_forward(
                        model,
                        video_tokens,
                        audio_tokens,
                        neg_video_text,
                        neg_audio_text,
                        t,
                        latent_shape,
                        audio_channels,
                    )

                    pred_video = pred_neg + PREVIEW_CFG * (pred_video - pred_neg)

                step_result = scheduler.step(
                    pred_video.to(torch.float32),
                    t,
                    video_tokens.to(torch.float32),
                    return_dict=False,
                )

                prev_tokens = step_result[0].detach().to(torch.bfloat16)

                latents = unpack_video_latent(
                    prev_tokens,
                    latent_shape,
                    patch_size,
                    patch_size_t,
                )

                latents = latents.detach()

        latents = latents.detach()

        vae = LTXVaeHolder.get().to(device)

        with torch.inference_mode():
            img = decode_preview_latent(vae, latents)

        vae.to("cpu")

        out_path = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        Image.fromarray(img).save(out_path)

        print(f"  ↳ Preview saved to / Preview guardada: {out_path}")

    except Exception as e:
        print(f"  [!] Preview failed / Preview falló: {e}")

    finally:
        if vae is not None:
            try:
                vae.to("cpu")
            except Exception:
                pass

        if was_training:
            model.train()

        free_vram()


# ============================================================================
# GLOBAL CONFIG REFERENCE
# ============================================================================

CURRENT_CONFIG = None
CURRENT_TRANSFORMER = None
CURRENT_AUDIO_CHANNELS = None


# ============================================================================
# TRAIN
# ============================================================================

def train_ltx23():
    global CURRENT_CONFIG
    global CURRENT_TRANSFORMER
    global CURRENT_AUDIO_CHANNELS

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    ensure_ltx23_model_downloaded(MODEL_ID)

    if not os.path.exists(CACHE_DIR):
        raise RuntimeError(f"No existe caché: {CACHE_DIR}.")

    print()
    print("Loading LTX-2.3 Transformer... / Cargando Transformer de LTX-2.3...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=None,
        audio_vae=None,
        text_encoder=None,
        tokenizer=None,
        processor=None,
        vocoder=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    transformer = pipe.transformer
    connectors = getattr(pipe, "connectors", None)
    scheduler = getattr(pipe, "scheduler", None)

    CURRENT_CONFIG = transformer.config
    CURRENT_TRANSFORMER = transformer

    AUDIO_CHANNELS = int(getattr(CURRENT_CONFIG, "audio_in_channels", 128))
    CURRENT_AUDIO_CHANNELS = AUDIO_CHANNELS

    entries = load_cached_entries(CACHE_DIR, AUDIO_CHANNELS, MAX_TEXT_TOKENS)

    prepare_text_conditioning(entries, connectors, MAX_TEXT_TOKENS)

    special_texts = prepare_special_text_conditioning(
        connectors,
        MAX_TEXT_TOKENS,
        PREVIEW_CUSTOM_PROMPT,
    )

    try:
        pipe.connectors = None
    except Exception:
        pass

    del connectors
    del pipe

    free_vram()

    nf4_index = os.path.join(MODEL_ID, "index.json")

    if os.path.exists(nf4_index):
        print()
        print("NF4 CACHE DETECTED! / ¡CACHÉ NF4 DETECTADA!")

        t0 = time.time()

        transformer = load_nf4_cache_(transformer, MODEL_ID)

        CURRENT_TRANSFORMER = transformer

        transformer.requires_grad_(False)

        if CAST_FROZEN_BF16:
            cast_frozen_to_bf16(transformer)

        transformer.to("cuda")

        free_vram()

        print(f"[NF4] Cache loaded in / Caché cargada en {time.time() - t0:.1f}s")
        print(f"Transformer pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    else:
        raise RuntimeError("No existe caché NF4 en MODEL_ID.")

    enable_memory_efficient_attention(transformer)

    if hasattr(transformer, "enable_gradient_checkpointing"):
        try:
            transformer.enable_gradient_checkpointing()
            print("Gradient checkpointing activado.")
        except Exception:
            pass

    target_modules = discover_lora_targets(transformer)

    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)}")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type=None,
        init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)

    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                adapter.to(dtype=torch.bfloat16)

        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                adapter.to(dtype=torch.bfloat16)

    enable_gradient_checkpointing_safe(transformer, model)

    try:
        model.to("cuda")
    except Exception:
        pass

    model.print_trainable_parameters()

    def make_inputs_require_grad(module, inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)

    hooks = []

    for name in ("proj_in", "video_in", "audio_in", "x_embedder"):
        if hasattr(transformer, name):
            try:
                hooks.append(
                    getattr(transformer, name).register_forward_hook(
                        make_inputs_require_grad
                    )
                )
                break
            except Exception:
                pass

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = bnb.optim.PagedAdamW8bit(
        trainable,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------------------
    # Resume checkpoint
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Resume checkpoint (con detección de cambio de rank/targets)
    # ------------------------------------------------------------
    start_step = 0

    adapter_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    adapter_cfg_path = os.path.join(RESUME_DIR, "adapter_config.json")

    # Si el checkpoint es de OTRA configuración de LoRA (rank distinto),
    # el resume es imposible (shapes distintas) y solo genera warnings de
    # "size mismatch". Lo detectamos ANTES de cargar y lo saltamos limpio.
    resume_compatible = True
    if os.path.exists(adapter_cfg_path):
        try:
            with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                acfg = json.load(f)
            saved_r = int(acfg.get("r", -1))
            saved_alpha = int(acfg.get("lora_alpha", -1))
            if saved_r != LORA_RANK:
                resume_compatible = False
                print("=" * 65)
                print(f"[!] Checkpoint INCOMPATIBLE: rank guardado={saved_r}, "
                      f"rank actual={LORA_RANK}.")
                print("    Al cambiar rank hay que entrenar desde 0.")
                print("    Se IGNORA el checkpoint y se arranca limpio.")
                print("    (Borra resume_checkpoint/ y current_step.txt para")
                print("     limpiarlo del disco.)")
                print("=" * 65)
        except Exception:
            pass

    if resume_compatible and os.path.exists(adapter_path) and os.path.exists(STEP_FILE):
        print("=" * 65)
        print("Checkpoint detected! Restoring state... / ¡Checkpoint detectado! Restaurando estado...")
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())

            state = load_file(adapter_path, device="cpu")
            set_peft_model_state_dict(model, state)

            if os.path.exists(OPT_FILE):
                try:
                    optimizer.load_state_dict(torch.load(OPT_FILE, weights_only=False))
                    print("Optimizer restaurado.")
                except Exception:
                    print("[!] No se pudo restaurar optimizer. Se continúa con optimizer nuevo.")

            print(f"Resuming training from step / Reanudando entrenamiento desde el paso {start_step}...")

        except Exception as e:
            print(f"[!] Warning reading checkpoint / Advertencia al leer checkpoint: {e}")
            start_step = 0

        print("=" * 65)

    last_step_executed = start_step

    # ------------------------------------------------------------
    # Checkpoint saver
    # ------------------------------------------------------------

    def save_checkpoint_now(current_s):
        if current_s <= 0:
            return

        print()
        print(f"Saving checkpoint state at step / Guardando estado en paso {current_s}...")

        os.makedirs(RESUME_DIR, exist_ok=True)

        try:
            model.save_pretrained(RESUME_DIR)
        except Exception:
            pass

        try:
            torch.save(optimizer.state_dict(), OPT_FILE)
        except Exception:
            pass

        try:
            with open(STEP_FILE, "w", encoding="utf-8") as f:
                f.write(str(current_s))
        except Exception:
            pass

        try:
            ckpt = os.path.join(OUTPUT_DIR, f"LTX23_LoRA_step_{current_s}.safetensors")
            save_lora(model, ckpt)
            print(f"Checkpoint saved successfully at step / Checkpoint guardado en paso {current_s}: {ckpt}")
        except Exception:
            pass

    # ------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------

    def handle_signal(sig, frame):
        nonlocal last_step_executed

        print()
        print(f"Signal received / Señal de detención recibida ({sig}).")

        save_checkpoint_now(last_step_executed)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)

    except Exception:
        pass

    # ------------------------------------------------------------
    # LR
    # ------------------------------------------------------------

    def lr_at(step):
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)

        progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)

        return LR * (
            MIN_LR_RATIO
            + (1 - MIN_LR_RATIO)
            * 0.5
            * (1 + math.cos(math.pi * progress))
        )

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    if SEED > 0:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    avg_time = 0.0

    print()
    print(f"STARTING TRAINING / ¡ARRANCANDO ENTRENAMIENTO! {len(entries)} images cached / imágenes cacheadas.")
    print(f"LoRA export prefix / Prefijo de exportación: '{LORA_KEY_PREFIX}' (scaling alpha/rank horneado en lora_B).")

    try:
        for step in range(start_step + 1, TOTAL_STEPS + 1):
            last_step_executed = step

            t0 = time.time()

            loss_video = None
            loss_audio = None

            entry = random.choice(entries)

            video_clean = entry["video"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            audio_clean = entry["audio"].to("cuda", dtype=torch.bfloat16, non_blocking=True)

            video_text = entry["video_text"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            audio_text = entry["audio_text"].to("cuda", dtype=torch.bfloat16, non_blocking=True)

            if video_text.ndim == 2:
                video_text = video_text.unsqueeze(0)

            if audio_text.ndim == 2:
                audio_text = audio_text.unsqueeze(0)

            patch_size = int(getattr(CURRENT_CONFIG, "patch_size", 1))
            patch_size_t = int(getattr(CURRENT_CONFIG, "patch_size_t", 1))

            video_tokens = patch_video_latent(video_clean, patch_size, patch_size_t)
            audio_tokens = patch_audio_latent(audio_clean)

            B = video_tokens.shape[0]

            sigma = torch.rand(B, device="cuda", dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)

            noise_video = torch.randn_like(video_tokens)
            noise_audio = torch.randn_like(audio_tokens)

            t_video = sigma.view(B, 1, 1)
            t_audio = sigma.view(B, 1, 1)

            noisy_video = (1.0 - t_video) * video_tokens + t_video * noise_video
            noisy_audio = (1.0 - t_audio) * audio_tokens + t_audio * noise_audio

            target_video = noise_video - video_tokens

            if USE_AUDIO_LOSS:
                target_audio = noise_audio - audio_tokens
            else:
                target_audio = None

            timestep = make_video_timestep(
                sigma,
                video_tokens.shape[1],
                "cuda",
                torch.bfloat16,
            )

            audio_timestep = (
                sigma * float(getattr(CURRENT_CONFIG, "timestep_scale_multiplier", 1000))
            ).to(torch.bfloat16)

            _, _, num_frames, height, width = video_clean.shape

            forward_kwargs = {
                "hidden_states": noisy_video,
                "audio_hidden_states": noisy_audio,
                "encoder_hidden_states": video_text,
                "audio_encoder_hidden_states": audio_text,
                "timestep": timestep,
                "audio_timestep": audio_timestep,
                "sigma": sigma,
                "audio_sigma": sigma,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": FRAME_RATE,
                "audio_num_frames": audio_clean.shape[-1],
                "return_dict": False,
            }

            signature = inspect.signature(transformer.forward)

            forward_kwargs = {
                k: v
                for k, v in forward_kwargs.items()
                if k in signature.parameters
            }

            output = model(**forward_kwargs)

            if isinstance(output, tuple):
                if len(output) == 0:
                    raise RuntimeError("LTX-2.3 forward devolvió tuple vacía.")

                pred_video = output[0]

                if len(output) > 1 and USE_AUDIO_LOSS:
                    pred_audio = output[1]
                else:
                    pred_audio = None

            else:
                pred_video = getattr(output, "video", None)

                if pred_video is None:
                    pred_video = getattr(output, "sample", None)

                pred_audio = None

                if USE_AUDIO_LOSS:
                    pred_audio = getattr(output, "audio", None)

                    if pred_audio is None:
                        pred_audio = getattr(output, "audio_sample", None)

            if pred_video is None:
                raise RuntimeError("No se pudo obtener predicción de vídeo.")

            output = None

            if not USE_AUDIO_LOSS:
                pred_audio = None

            if USE_AUDIO_LOSS and pred_audio is not None:
                if target_audio is None:
                    target_audio = noise_audio - audio_tokens

                if pred_audio.shape != target_audio.shape:
                    raise RuntimeError("La forma de salida de audio no coincide con el target.")

                loss_video = mse_loss_chunked(pred_video, target_video)
                loss_audio = mse_loss_chunked(pred_audio, target_audio)

                loss = (loss_video + loss_audio) * 0.5

            else:
                loss = mse_loss_chunked(pred_video, target_video)

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            running_loss += loss.item() * GRAD_ACCUM_STEPS

            loss = None

            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()

                current_lr = lr_at(step)

                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            else:
                grad_norm = 0.0
                current_lr = lr_at(step)

            elapsed = time.time() - t0

            avg_time = elapsed if avg_time == 0 else (0.1 * elapsed + 0.9 * avg_time)

            eta_s = (TOTAL_STEPS - step) * avg_time
            eta = f"{int(eta_s // 3600):02d}:{int((eta_s % 3600) // 60):02d}:{int(eta_s % 60):02d}"

            pct = step / TOTAL_STEPS
            barra = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))

            avg_loss = running_loss / max(1, step - start_step)

            progress_line = (
                f"Step/Paso {step:4d}/{TOTAL_STEPS} [{barra}] {pct * 100:5.1f}% | "
                f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
                f"lr {current_lr:.2e} | {avg_time:.2f}s/it | ETA {eta}"
            )

            print(f"\r{progress_line}", end="", flush=True)

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                run_preview_ltx(
                    model,
                    scheduler,
                    entries,
                    special_texts,
                    step,
                    AUDIO_CHANNELS,
                )

            free_vram(
                video_clean,
                audio_clean,
                video_text,
                audio_text,
                video_tokens,
                audio_tokens,
                noise_video,
                noise_audio,
                noisy_video,
                noisy_audio,
                target_video,
                target_audio,
                sigma,
                timestep,
                audio_timestep,
                pred_video,
                pred_audio,
                output,
                loss_video,
                loss_audio,
                loss,
            )

    except KeyboardInterrupt:
        save_checkpoint_now(last_step_executed)
        return

    except SystemExit:
        return

    print()
    print()
    print("Training completed! / ¡Entrenamiento finalizado!")

    save_checkpoint_now(TOTAL_STEPS)

    final_path = os.path.join(OUTPUT_DIR, "LTX23_FINAL_LoRA.safetensors")
    save_lora(model, final_path)

    print(f"Final LoRA saved to / Tu LoRA definitivo está en: {final_path}")
    print(f"Formato exportado: prefijo='{LORA_KEY_PREFIX}', sin '.default', scaling horneado -> usa strength=1.0 en ComfyUI.")

    for hook in hooks:
        try:
            hook.remove()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        train_ltx23()
    except Exception:
        print()
        print("=" * 80)
        print("ERROR EN TRAINER LTX-2.3")
        print("=" * 80)
        traceback.print_exc()
        raise