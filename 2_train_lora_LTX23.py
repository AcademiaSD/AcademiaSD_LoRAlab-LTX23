# -*- coding: utf-8 -*-
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

from accelerate import init_empty_weights
from diffusers import DiffusionPipeline, LTX2VideoTransformer3DModel
from diffusers.configuration_utils import FrozenDict
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


# ---------------------------------------------------------------------------
# Activation offload
# ---------------------------------------------------------------------------
try:
    from torch.autograd.graph import save_on_cpu as _save_on_cpu_ctx
    _SAVE_ON_CPU_AVAILABLE = True
except Exception:
    _save_on_cpu_ctx = None
    _SAVE_ON_CPU_AVAILABLE = False

ACTIVATION_OFFLOAD_ACTIVE = False


# ===========================================================================
# CONFIG
# ===========================================================================
DEFAULTS = {
    "model_id": "./LTX23-NF4",
    "cache_dir": "./cached_data_ltx23",
    "output_dir": "./ltx23_lora_output",

    "total_steps": 800,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,

    "lora_rank": 32,
    "lora_alpha": 32,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,

    "save_every": 100,
    "seed": 314159,
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
    "preview_steps": 30,
    "preview_cfg": 3.0,
    "preview_sampler": "euler",
    "preview_lora_scale": 1.0,
    "preview_cfg_max": 7.0,
    "preview_cfg_rescale": 0.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",

    # Preview diagnóstico.
    "preview_mode": "gen",                # gen | recon | onestep
    "preview_recon_sigma": 0.55,
    "preview_frame_index": -1,
    "preview_shift": 1.0,
    "preview_vae_fp32": True,
    "preview_audio_cfg": 1.0,
    "preview_sample_name": "",
    "preview_compare_base": False,

    # VAE.
    # True  -> latent * std / scaling_factor + mean
    # False -> latent * std + mean
    "preview_vae_use_scaling_factor": True,

    # Formato de keys del LoRA exportado.
    "lora_key_prefix": "diffusion_model.",

    # Modo baja VRAM.
    "low_vram_12gb": True,
    "activation_offload": True,
    "loss_chunk_elements": 2000000,
}

CONFIG_PATH = "train_settings.json"

HF_BASE_REPO_ID = "diffusers/LTX-2.3-Diffusers"
HF_NF4_REPO_ID = "AcademiaSD/LTX23_NF4"


if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[OK] Configuración cargada: {}".format(CONFIG_PATH))
else:
    cfg = {}
    print("[!] No existe {}; usando valores por defecto.".format(CONFIG_PATH))


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
PREVIEW_SAMPLER = str(cfg_get("preview_sampler", DEFAULTS["preview_sampler"])).strip().lower()
PREVIEW_LORA_SCALE = float(cfg_get("preview_lora_scale", DEFAULTS["preview_lora_scale"]))
PREVIEW_CFG_MAX = float(cfg_get("preview_cfg_max", DEFAULTS["preview_cfg_max"]))
PREVIEW_CFG_RESCALE = float(cfg_get("preview_cfg_rescale", DEFAULTS["preview_cfg_rescale"]))
PREVIEW_CAPTION_MODE = str(cfg_get("preview_caption_mode", DEFAULTS["preview_caption_mode"])).strip().lower()
PREVIEW_CUSTOM_PROMPT = str(cfg_get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip()

PREVIEW_MODE = str(cfg_get("preview_mode", DEFAULTS["preview_mode"])).strip().lower()
PREVIEW_RECON_SIGMA = float(cfg_get("preview_recon_sigma", DEFAULTS["preview_recon_sigma"]))
PREVIEW_FRAME_INDEX = int(cfg_get("preview_frame_index", DEFAULTS["preview_frame_index"]))
PREVIEW_SHIFT = float(cfg_get("preview_shift", DEFAULTS["preview_shift"]))
PREVIEW_VAE_FP32 = _cfg_bool("preview_vae_fp32", DEFAULTS["preview_vae_fp32"])
PREVIEW_AUDIO_CFG = float(cfg_get("preview_audio_cfg", DEFAULTS["preview_audio_cfg"]))
PREVIEW_SAMPLE_NAME = str(cfg_get("preview_sample_name", DEFAULTS["preview_sample_name"])).strip()
PREVIEW_COMPARE_BASE = _cfg_bool("preview_compare_base", DEFAULTS["preview_compare_base"])

PREVIEW_VAE_USE_SCALING_FACTOR = _cfg_bool(
    "preview_vae_use_scaling_factor",
    DEFAULTS["preview_vae_use_scaling_factor"],
)

LORA_KEY_PREFIX = str(cfg_get("lora_key_prefix", DEFAULTS["lora_key_prefix"]))

LOW_VRAM_12GB = _cfg_bool("low_vram_12gb", DEFAULTS["low_vram_12gb"])
ACTIVATION_OFFLOAD = _cfg_bool("activation_offload", DEFAULTS["activation_offload"])
LOSS_CHUNK_ELEMENTS = int(cfg_get("loss_chunk_elements", DEFAULTS["loss_chunk_elements"]))

if PREVIEW_MODE not in ("gen", "recon", "onestep"):
    PREVIEW_MODE = "gen"

if PROJECT_NAME:
    CACHE_DIR = "./cached_data_ltx23_{}".format(PROJECT_NAME)
    OUTPUT_DIR = "./ltx23_lora_output_{}".format(PROJECT_NAME)
else:
    CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()
    OUTPUT_DIR = str(cfg_get("output_dir", DEFAULTS["output_dir"])).strip()

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE = os.path.join(OUTPUT_DIR, "current_step.txt")


# ===========================================================================
# BANNER
# ===========================================================================
print()
print("=" * 80)
print(" LTX-2.3 LoRA TRAINER ")
print("=" * 80)
print("  Model ID / ID Modelo        : {}".format(MODEL_ID))
print("  Base Repo                   : {}".format(HF_BASE_REPO_ID))
print("  NF4 Repo                    : {}".format(HF_NF4_REPO_ID))
print("  Project / Proyecto          : {}".format(PROJECT_NAME if PROJECT_NAME else "(Default)"))
print("  Trigger Word / Palabra      : {}".format(TRIGGER_WORD))
print("  Cache Dir / Carpeta Cache   : {}".format(CACHE_DIR))
print("  Output Dir / Salida         : {}".format(OUTPUT_DIR))
print("  Total Steps / Pasos         : {}".format(TOTAL_STEPS))
print("  Learning Rate / LR          : {}".format(LR))
print("  LoRA Rank/Alpha             : {}/{}".format(LORA_RANK, LORA_ALPHA))
print("  Batch / Grad Accum          : {}/{}".format(BATCH_SIZE, GRAD_ACCUM_STEPS))
print("  Max Text Tokens             : {}".format(MAX_TEXT_TOKENS))
print("  LoRA Only Attention         : {}".format("ON" if LORA_ONLY_ATTN else "OFF"))
print("  Use Audio Loss              : {}".format("ON" if USE_AUDIO_LOSS else "OFF"))
print("  Preview Mode                : {}".format(PREVIEW_MODE))
print("  Preview Compare Base/LoRA   : {}".format("ON" if PREVIEW_COMPARE_BASE else "OFF"))
print("  Preview Caption Mode        : {}".format(PREVIEW_CAPTION_MODE))
print("  Preview Custom Prompt       : '{}'".format(PREVIEW_CUSTOM_PROMPT))
print("  Preview Sample Name         : '{}'".format(PREVIEW_SAMPLE_NAME))
print("  Preview Every / Steps / CFG : {} / {} / {}".format(PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG))
print("  Preview Audio CFG           : {}".format(PREVIEW_AUDIO_CFG))
print("  Preview Sampler             : {}".format(PREVIEW_SAMPLER))
print("  Preview LoRA Scale          : {:.2f}".format(PREVIEW_LORA_SCALE))
print("  Preview CFG clamp / rescale : max={:.1f} / rescale={:.2f}".format(PREVIEW_CFG_MAX, PREVIEW_CFG_RESCALE))
print("  Preview Recon Sigma         : {:.2f}".format(PREVIEW_RECON_SIGMA))
print("  Preview Frame Index         : {}".format(PREVIEW_FRAME_INDEX))
print("  Preview Shift               : {:.2f}".format(PREVIEW_SHIFT))
print("  Preview VAE FP32            : {}".format("ON" if PREVIEW_VAE_FP32 else "OFF"))
print("  Preview VAE scaling_factor  : {}".format("ON" if PREVIEW_VAE_USE_SCALING_FACTOR else "OFF"))
print("  LoRA Key Prefix / Prefijo   : '{}'".format(LORA_KEY_PREFIX))
print("  Seed Configured / Semilla   : {} ({})".format(SEED, "RANDOM" if SEED <= 0 else "FIXED"))
print("  Low VRAM 12GB mode          : {}".format("ON" if LOW_VRAM_12GB else "OFF"))
print("  Activation Offload          : {} (PyTorch save_on_cpu: {})".format(
    "ON" if ACTIVATION_OFFLOAD else "OFF",
    "OK" if _SAVE_ON_CPU_AVAILABLE else "NO DISPONIBLE"
))
print("  Loss Chunk Elements         : {}".format(LOSS_CHUNK_ELEMENTS))
print("=" * 80)


# ===========================================================================
# DESCARGA DEL MODELO
# ===========================================================================
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
        print("[OK] Modelo local encontrado en: {}".format(local_path))
        return local_path

    print()
    print("=" * 80)
    print("WARNING / ATENCIÓN")
    print("=" * 80)
    print("This will download about 75 GB. This may take several minutes.")
    print("Esto descargará unos 75 GB. Esto puede tardar varios minutos.")
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
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    token = get_hf_token()
    if token:
        print("Using HF Token / Usando token de HF")

    os.makedirs(local_path, exist_ok=True)

    # El Transformer BF16 (~38 GB) no se usa: sale entero del repo NF4.
    print()
    print("Downloading / Descargando:", HF_BASE_REPO_ID)
    snapshot_download(
        repo_id=HF_BASE_REPO_ID,
        local_dir=local_path,
        token=token,
        max_workers=4,
        ignore_patterns=["transformer/*"],
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
    print("[OK] Modelo descargado en: {}".format(local_path))
    return local_path


def ensure_nf4_others_downloaded(local_path):
    if os.path.exists(os.path.join(local_path, "others.safetensors")):
        return

    from huggingface_hub import hf_hub_download

    print("Downloading / Descargando: {}/others.safetensors".format(HF_NF4_REPO_ID))
    hf_hub_download(repo_id=HF_NF4_REPO_ID, filename="others.safetensors", local_dir=local_path, token=get_hf_token())


# ===========================================================================
# UTILIDADES
# ===========================================================================
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


def filter_forward_kwargs(kwargs, forward_fn):
    try:
        signature = inspect.signature(forward_fn)
    except Exception:
        return kwargs

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )

    if has_var_keyword:
        return kwargs

    return {
        k: v
        for k, v in kwargs.items()
        if k in signature.parameters
    }


# ===========================================================================
# NF4 CACHE
# ===========================================================================
def load_nf4_transformer(cache_dir):
    """
    Construye el Transformer vacío desde config.json y lo rellena con la caché NF4
    (weights/ + others.safetensors). No hace falta el Transformer en BF16.
    Builds the empty Transformer from config.json and fills it from the NF4 cache.
    """
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("No existe index.json: {}".format(index_path))

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(LTX2VideoTransformer3DModel.load_config(cache_dir))

    quantized = index.get("quantized", {})
    unquantized = index.get("unquantized", {})
    weights_dir = os.path.join(cache_dir, "weights")

    replaced = 0

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError("No existe peso NF4: {}".format(filepath))

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

        # En meta: nn.Linear.__init__ reservaría y rellenaría una matriz FP32 que se tira en seguida.
        with torch.device("meta"):
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
        if not os.path.exists(filepath):
            raise FileNotFoundError("No existe peso unquantized: {}".format(filepath))

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")
            bias = None
            if info.get("bias", False):
                bias = f.get_tensor("bias")

        with torch.device("meta"):
            layer = torch.nn.Linear(
                int(info["in_features"]),
                int(info["out_features"]),
                bias=info.get("bias", False),
            )

        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        setattr(parent, child_name, layer)

    # Normas, tablas de modulación, etc.: en BF16, como los daba from_pretrained.
    transformer.load_state_dict(load_file(os.path.join(cache_dir, "others.safetensors")), strict=False, assign=True)

    missing = [n for n, t in list(transformer.named_parameters()) + list(transformer.named_buffers()) if t.is_meta]
    if missing:
        raise RuntimeError("Caché NF4 incompleta / NF4 cache incomplete: {}".format(missing[:5]))

    verified = 0
    for _, module in transformer.named_modules():
        if isinstance(module, Linear4bit):
            if (
                getattr(module.weight, "bnb_quantized", False)
                and getattr(module.weight, "quant_state", None) is not None
            ):
                verified += 1

    print("Capas NF4 reconstruidas: {}".format(replaced))
    print("Capas NF4 verificadas: {}".format(verified))

    if verified != replaced:
        raise RuntimeError("La verificación NF4 no coincide.")

    print("[OK] Caché NF4 cargada correctamente.")
    return transformer


# ===========================================================================
# LoRA TARGETS
# ===========================================================================
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


# ===========================================================================
# PROMPT / TEXT CONDITIONING
# ===========================================================================
def load_prompt_structure(cache_dir, prefix):
    path = os.path.join(cache_dir, "{}_structure.json".format(prefix))
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
            result.extend(flatten_tensors(v, "{}.{}".format(prefix, k)))
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            result.extend(flatten_tensors(v, "{}.{}".format(prefix, i)))

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
    tag = "mt{}_reg128_v2".format(int(max_text_tokens or 0))
    video_text_path = os.path.join(cache_dir, "{}video_text{}.pt".format(base, tag))
    audio_text_path = os.path.join(cache_dir, "{}audio_text{}.pt".format(base, tag))
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

        #print()
        #print("Precomputando text conditioning para {} entradas...".format(len(missing)))

        connectors.to("cuda")
        connectors.eval()

        for param in connectors.parameters():
            param.requires_grad_(False)

        for entry in missing:
            base = entry["name"]

            if entry.get("prompt", None) is None:
                entry["prompt"] = load_prompt_structure(CACHE_DIR, "{}_prompt".format(base))

            if entry.get("prompt", None) is None:
                raise RuntimeError("No hay prompt cacheado para {}.".format(base))

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
        #print()
        #print("Precomputando textos especiales para preview:")

        for prefix in missing:
            print("  - {}".format(prefix))

        connectors.to("cuda")
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
        audio_path = os.path.join(cache_dir, "{}_audio_latent.pt".format(base))

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
            prompt_result = load_prompt_structure(cache_dir, "{}_prompt".format(base))
            if prompt_result is None:
                continue

        video_latent = torch.load(video_path, map_location="cpu", weights_only=True)
        if video_latent is None:
            continue

        video_latent = pin_cpu_tensor(video_latent.to(torch.bfloat16))

        audio_latent_raw = torch.load(audio_path, map_location="cpu", weights_only=True)

        if audio_latent_raw is None:
            if video_latent.ndim == 5:
                bsz = video_latent.shape[0]
            else:
                bsz = 1

            audio_latent = torch.zeros(
                (bsz, audio_channels, 1),
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


# ===========================================================================
# PATCHIFY / TIMESTEP / LOSS
# ===========================================================================
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


def align_video_latent_to_patch(latent, patch_size=1, patch_size_t=1):
    if latent.ndim != 5:
        return latent

    B, C, Fm, H, W = latent.shape

    patch_size = max(1, int(patch_size))
    patch_size_t = max(1, int(patch_size_t))

    Fm = (Fm // patch_size_t) * patch_size_t
    H = (H // patch_size) * patch_size
    W = (W // patch_size) * patch_size

    return latent[:, :, :Fm, :H, :W].contiguous()


def make_video_timestep(sigma, seq_len, device, dtype):
    multiplier = float(getattr(CURRENT_CONFIG, "timestep_scale_multiplier", 1000))

    return (
        sigma.view(-1, 1)
        .expand(-1, seq_len)
        * multiplier
    ).to(device=device, dtype=dtype)


def mse_loss_chunked(pred, target, chunk_elements=None):
    if chunk_elements is None:
        chunk_elements = int(globals().get("LOSS_CHUNK_ELEMENTS", 2000000))

    chunk_elements = max(64, int(chunk_elements))

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


# ===========================================================================
# OPTIMIZACIONES
# ===========================================================================
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


# ===========================================================================
# PREVIEW: ESCALADO TEMPORAL DEL LoRA
# ===========================================================================
def _iter_lora_layers(model):
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            yield m


def apply_preview_lora_scale(model, factor):
    factor = float(factor)

    if abs(factor - 1.0) < 1e-9:
        return False

    saved = []

    for m in _iter_lora_layers(model):
        sc = getattr(m, "scaling", None)

        if isinstance(sc, dict):
            saved.append((m, {k: sc[k] for k in sc}))

            for k in sc:
                try:
                    sc[k] = float(sc[k]) * factor
                except Exception:
                    pass

        elif isinstance(sc, (int, float)):
            saved.append((m, sc))

            try:
                m.scaling = float(sc) * factor
            except Exception:
                pass

    if saved:
        model._preview_lora_saved = saved
        return True

    return False


def restore_preview_lora_scale(model):
    saved = getattr(model, "_preview_lora_saved", None)
    if not saved:
        return

    for m, orig in saved:
        try:
            sc = getattr(m, "scaling", None)

            if isinstance(orig, dict) and isinstance(sc, dict):
                for k, v in orig.items():
                    sc[k] = v
            else:
                m.scaling = orig
        except Exception:
            pass

    try:
        del model._preview_lora_saved
    except Exception:
        pass


# ===========================================================================
# EXPORT LoRA
# ===========================================================================
def build_lora_metadata(step):
    """
    Cabecera del .safetensors: no cambia ningún peso. Claves legibles más la convención
    ss_* de kohya-ss, que es lo que leen CivitAI y los gestores de LoRAs para rellenar la
    ficha. CivitAI saca las palabras de activación de ss_tag_frequency ({carpeta: {tag: veces}}).
    """
    meta = {
        "trained_with": "AcademiaSD LoRAlab LTX-2.3",
        "ss_sd_model_name": "LTX-2.3",
        "ss_base_model_version": "LTX-2.3",
        "ss_network_module": "peft.LoraModel",
        "ss_network_dim": LORA_RANK,
        "ss_network_alpha": LORA_ALPHA,
        "ss_learning_rate": LR,
        "ss_lr_scheduler": "cosine_with_warmup",
        "ss_lr_warmup_steps": WARMUP_STEPS,
        "ss_max_train_steps": TOTAL_STEPS,
        "ss_steps": step,
        "ss_batch_size_per_device": BATCH_SIZE,
        "ss_gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "ss_seed": SEED,
        "ss_mixed_precision": "bf16",
    }

    if TRIGGER_WORD:
        meta["trigger_word"] = TRIGGER_WORD
        meta["ss_tag_frequency"] = json.dumps({"dataset": {TRIGGER_WORD: 1}})
    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
    meta["ss_output_name"] = PROJECT_NAME or TRIGGER_WORD or "ltx23_lora"

    # La resolución la fija el pre-caché del proyecto.
    pc_json = os.path.join(CACHE_DIR, "pre_cache_settings_{}.json".format(PROJECT_NAME))
    if os.path.exists(pc_json):
        with open(pc_json, "r", encoding="utf-8") as f:
            area = json.load(f).get("target_area")
        if area:
            side = int(round(float(area) ** 0.5))
            meta["ss_resolution"] = "({},{})".format(side, side)

    # safetensors exige que todos los valores sean str.
    return {k: str(v) for k, v in meta.items()}


def save_lora(model, path, step):
    prefix = LORA_KEY_PREFIX or ""

    scaling = float(LORA_ALPHA) / float(max(1, LORA_RANK))

    state = {}

    for name, tensor in model.state_dict().items():
        if "lora_" not in name:
            continue

        clean = name.replace("base_model.model.", "")
        clean = clean.replace(".default.", ".")

        t = tensor.detach().to(torch.float32).cpu()

        if ".lora_B." in clean:
            t = t * scaling

        t = t.to(torch.bfloat16).contiguous()

        state[prefix + clean] = t

    save_file(
        state,
        path,
        metadata={
            "format": "ltx23_lora",
            "lora_key_prefix": prefix,
            "baked_scaling": "{:.6f}".format(scaling),
            **build_lora_metadata(step),
        },
    )


# ===========================================================================
# PREVIEW
# ===========================================================================
class LTXVaeHolder:
    vae = None

    @classmethod
    def get(cls):
        if cls.vae is None:
            use_fp32 = globals().get("PREVIEW_VAE_FP32", True)
            dtype = torch.float32 if use_fp32 else torch.bfloat16

            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                transformer=None,
                text_encoder=None,
                audio_vae=None,
                tokenizer=None,
                processor=None,
                vocoder=None,
                torch_dtype=dtype,
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
            max_seq = float(getattr(scheduler.config, "max_image_seq_len", 4096))
            base_shift = float(getattr(scheduler.config, "base_shift", 0.5))
            max_shift = float(getattr(scheduler.config, "max_shift", 1.15))

            if max_seq > base_seq:
                m = (max_shift - base_shift) / (max_seq - base_seq)
                b = base_shift - m * base_seq
                mu = float(seq_len) * m + b
                mu = min(max(mu, base_shift), max_shift)
                kwargs["mu"] = mu
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
        forward_kwargs = filter_forward_kwargs(forward_kwargs, CURRENT_TRANSFORMER.forward)
    else:
        forward_kwargs = filter_forward_kwargs(forward_kwargs, model.forward)

    output = model(**forward_kwargs)

    if isinstance(output, tuple):
        if len(output) == 0:
            raise RuntimeError("Preview: forward devolvió tupla vacía.")

        pred_video = output[0]
        pred_audio = output[1] if len(output) > 1 else None
    else:
        pred_video = getattr(output, "video", None)
        if pred_video is None:
            pred_video = getattr(output, "sample", None)

        pred_audio = getattr(output, "audio", None)
        if pred_audio is None:
            pred_audio = getattr(output, "audio_sample", None)

    if pred_video is None:
        raise RuntimeError("Preview: no se pudo obtener predicción de video.")

    if pred_audio is None:
        pred_audio = torch.zeros_like(audio_tokens)

    return pred_video, pred_audio


def _rescale_guidance(guided, cond, rescale):
    if rescale <= 0.0:
        return guided

    dims = list(range(1, cond.ndim))

    std_cond = cond.std(dim=dims, keepdim=True).clamp_min(1e-6)
    std_guided = guided.std(dim=dims, keepdim=True).clamp_min(1e-6)

    rescaled = guided * (std_cond / std_guided)

    return rescale * rescaled + (1.0 - rescale) * guided


def predict_velocity_cfg(
    model,
    video_tokens,
    audio_tokens,
    video_text,
    audio_text,
    neg_video_text,
    neg_audio_text,
    t_val,
    latent_shape,
    audio_channels,
    cfg,
    rescale,
    cfg_audio=None,
):
    if cfg_audio is None:
        cfg_audio = cfg

    cond_v, cond_a = preview_forward(
        model,
        video_tokens,
        audio_tokens,
        video_text,
        audio_text,
        t_val,
        latent_shape,
        audio_channels,
    )

    if cfg <= 1.0 or neg_video_text is None:
        return cond_v, cond_a

    uncond_v, uncond_a = preview_forward(
        model,
        video_tokens,
        audio_tokens,
        neg_video_text,
        neg_audio_text,
        t_val,
        latent_shape,
        audio_channels,
    )

    guided_v = uncond_v + float(cfg) * (cond_v - uncond_v)
    guided_a = uncond_a + float(cfg_audio) * (cond_a - uncond_a)

    guided_v = _rescale_guidance(guided_v, cond_v, rescale)

    if float(cfg_audio) > 1.0 and rescale > 0.0:
        guided_a = _rescale_guidance(guided_a, cond_a, rescale)

    return guided_v, guided_a


def decode_preview_latent(vae, latent, frame_index=-1):
    dtype = next(vae.parameters()).dtype
    latent = latent.detach().to("cuda", dtype=dtype)

    if getattr(vae, "latents_mean", None) is not None and getattr(vae, "latents_std", None) is not None:
        latents_mean = torch.as_tensor(
            vae.latents_mean,
            device=latent.device,
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)

        latents_std = torch.as_tensor(
            vae.latents_std,
            device=latent.device,
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)

        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

        if PREVIEW_VAE_USE_SCALING_FACTOR and abs(scaling_factor) > 1e-12:
            latent = latent * latents_std / scaling_factor + latents_mean
        else:
            latent = latent * latents_std + latents_mean

    with torch.no_grad():
        decoded = vae.decode(latent, return_dict=False)[0]

    if decoded.ndim == 5:
        f = decoded.shape[2]

        if frame_index < 0:
            idx = f // 2
        else:
            idx = min(frame_index, f - 1)

        decoded = decoded[:, :, idx]

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


def make_preview_sigmas(steps, sigma_start=1.0, shift=1.0, device="cuda"):
    steps = max(1, int(steps))
    sigma_start = float(sigma_start)
    shift = float(shift)

    base = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    if abs(shift - 1.0) > 1e-6:
        base = shift * base / (1.0 + (shift - 1.0) * base)

    sigmas = sigma_start * base
    sigmas[0] = sigma_start
    sigmas[-1] = 0.0

    return sigmas


def prepare_preview_tensors(entry, device):
    video = entry["video"].to(device, dtype=torch.bfloat16, non_blocking=True)
    if video.ndim == 4:
        video = video.unsqueeze(0)
    video = video[:1].contiguous()

    audio = entry["audio"].to(device, dtype=torch.bfloat16, non_blocking=True)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    audio = audio[:1].contiguous()

    return video, audio


def run_onestep_x0_preview(
    model,
    video_clean,
    audio_clean,
    video_text,
    audio_text,
    neg_video_text,
    neg_audio_text,
    audio_channels,
    patch_size,
    patch_size_t,
    device,
    sigma_start,
    cfg,
    rescale,
    cfg_audio,
    seed,
):
    latent_shape = tuple(video_clean.shape)

    v0 = patch_video_latent(video_clean, patch_size, patch_size_t)
    a0 = patch_audio_latent(audio_clean)

    g = torch.Generator(device=device).manual_seed(int(seed))

    noise_v = torch.randn(v0.shape, generator=g, device=device, dtype=v0.dtype)
    noise_a = torch.randn(a0.shape, generator=g, device=device, dtype=a0.dtype)

    s = float(sigma_start)

    vx = (1.0 - s) * v0 + s * noise_v
    ax = (1.0 - s) * a0 + s * noise_a

    t_val = s * 1000.0

    with torch.no_grad():
        vel_v, vel_a = predict_velocity_cfg(
            model,
            vx,
            ax,
            video_text,
            audio_text,
            neg_video_text,
            neg_audio_text,
            t_val,
            latent_shape,
            audio_channels,
            cfg,
            rescale,
            cfg_audio,
        )

        x0_tokens = vx - s * vel_v

    latents = unpack_video_latent(
        x0_tokens,
        latent_shape,
        patch_size,
        patch_size_t,
    )

    return latents.detach()


def run_recon_flow_preview(
    model,
    video_clean,
    audio_clean,
    video_text,
    audio_text,
    neg_video_text,
    neg_audio_text,
    audio_channels,
    patch_size,
    patch_size_t,
    device,
    steps,
    cfg,
    rescale,
    cfg_audio,
    sigma_start,
    shift,
    seed,
):
    latent_shape = tuple(video_clean.shape)

    v0 = patch_video_latent(video_clean, patch_size, patch_size_t)
    a0 = patch_audio_latent(audio_clean)

    g = torch.Generator(device=device).manual_seed(int(seed))

    noise_v = torch.randn(v0.shape, generator=g, device=device, dtype=v0.dtype)
    noise_a = torch.randn(a0.shape, generator=g, device=device, dtype=a0.dtype)

    s0 = float(sigma_start)

    vx = (1.0 - s0) * v0 + s0 * noise_v
    ax = (1.0 - s0) * a0 + s0 * noise_a

    sigmas = make_preview_sigmas(
        steps=steps,
        sigma_start=s0,
        shift=shift,
        device=device,
    )

    with torch.no_grad():
        for i in range(sigmas.numel() - 1):
            s_cur = float(sigmas[i].item())
            s_next = float(sigmas[i + 1].item())

            if s_cur <= 0.0:
                break

            t_val = s_cur * 1000.0

            vel_v, vel_a = predict_velocity_cfg(
                model,
                vx,
                ax,
                video_text,
                audio_text,
                neg_video_text,
                neg_audio_text,
                t_val,
                latent_shape,
                audio_channels,
                cfg,
                rescale,
                cfg_audio,
            )

            d_sigma = s_next - s_cur

            vx = vx + d_sigma * vel_v
            ax = ax + d_sigma * vel_a

    latents = unpack_video_latent(
        vx,
        latent_shape,
        patch_size,
        patch_size_t,
    )

    return latents.detach()


def run_fullgen_flow_preview(
    model,
    video_ref,
    audio_ref,
    video_text,
    audio_text,
    neg_video_text,
    neg_audio_text,
    audio_channels,
    patch_size,
    patch_size_t,
    device,
    steps,
    cfg,
    rescale,
    cfg_audio,
    shift,
    seed,
):
    latent_shape = tuple(video_ref.shape)

    v_ref = patch_video_latent(video_ref, patch_size, patch_size_t)
    a_ref = patch_audio_latent(audio_ref)

    g = torch.Generator(device=device).manual_seed(int(seed))

    vx = torch.randn(v_ref.shape, generator=g, device=device, dtype=v_ref.dtype)
    ax = torch.randn(a_ref.shape, generator=g, device=device, dtype=a_ref.dtype)

    sigmas = make_preview_sigmas(
        steps=steps,
        sigma_start=1.0,
        shift=shift,
        device=device,
    )

    with torch.no_grad():
        for i in range(sigmas.numel() - 1):
            s_cur = float(sigmas[i].item())
            s_next = float(sigmas[i + 1].item())

            if s_cur <= 0.0:
                break

            t_val = s_cur * 1000.0

            vel_v, vel_a = predict_velocity_cfg(
                model,
                vx,
                ax,
                video_text,
                audio_text,
                neg_video_text,
                neg_audio_text,
                t_val,
                latent_shape,
                audio_channels,
                cfg,
                rescale,
                cfg_audio,
            )

            d_sigma = s_next - s_cur

            vx = vx + d_sigma * vel_v
            ax = ax + d_sigma * vel_a

    latents = unpack_video_latent(
        vx,
        latent_shape,
        patch_size,
        patch_size_t,
    )

    return latents.detach()


def reload_preview_settings():
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SAMPLER
    global PREVIEW_LORA_SCALE, PREVIEW_CFG_MAX, PREVIEW_CFG_RESCALE
    global PREVIEW_CAPTION_MODE, PREVIEW_CUSTOM_PROMPT
    global PREVIEW_MODE, PREVIEW_RECON_SIGMA, PREVIEW_FRAME_INDEX
    global PREVIEW_SHIFT, PREVIEW_VAE_FP32, PREVIEW_AUDIO_CFG
    global PREVIEW_SAMPLE_NAME, PREVIEW_VAE_USE_SCALING_FACTOR
    global PREVIEW_COMPARE_BASE
    global SEED

    try:
        if not os.path.exists(CONFIG_PATH):
            return

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            pcfg = json.load(f)

        if not isinstance(pcfg, dict):
            return

        def g(key, default):
            if key in pcfg:
                return pcfg[key]
            if (key + " ") in pcfg:
                return pcfg[key + " "]
            return default

        def gb(key, default):
            v = g(key, default)

            if isinstance(v, bool):
                return v

            if isinstance(v, (int, float)):
                return bool(v)

            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "y", "on")

            return bool(v)

        PREVIEW_EVERY = int(g("preview_every", PREVIEW_EVERY))
        PREVIEW_STEPS = int(g("preview_steps", PREVIEW_STEPS))
        PREVIEW_CFG = float(g("preview_cfg", PREVIEW_CFG))
        PREVIEW_SAMPLER = str(g("preview_sampler", PREVIEW_SAMPLER)).strip().lower()

        PREVIEW_LORA_SCALE = float(g("preview_lora_scale", PREVIEW_LORA_SCALE))
        PREVIEW_CFG_MAX = float(g("preview_cfg_max", PREVIEW_CFG_MAX))
        PREVIEW_CFG_RESCALE = float(g("preview_cfg_rescale", PREVIEW_CFG_RESCALE))

        PREVIEW_CAPTION_MODE = str(g("preview_caption_mode", PREVIEW_CAPTION_MODE)).strip().lower()
        PREVIEW_CUSTOM_PROMPT = str(g("preview_custom_prompt", PREVIEW_CUSTOM_PROMPT)).strip()

        PREVIEW_MODE = str(g("preview_mode", PREVIEW_MODE)).strip().lower()
        PREVIEW_RECON_SIGMA = float(g("preview_recon_sigma", PREVIEW_RECON_SIGMA))
        PREVIEW_FRAME_INDEX = int(g("preview_frame_index", PREVIEW_FRAME_INDEX))
        PREVIEW_SHIFT = float(g("preview_shift", PREVIEW_SHIFT))
        PREVIEW_VAE_FP32 = gb("preview_vae_fp32", PREVIEW_VAE_FP32)
        PREVIEW_AUDIO_CFG = float(g("preview_audio_cfg", PREVIEW_AUDIO_CFG))
        PREVIEW_SAMPLE_NAME = str(g("preview_sample_name", PREVIEW_SAMPLE_NAME)).strip()
        PREVIEW_VAE_USE_SCALING_FACTOR = gb("preview_vae_use_scaling_factor", PREVIEW_VAE_USE_SCALING_FACTOR)
        PREVIEW_COMPARE_BASE = gb("preview_compare_base", PREVIEW_COMPARE_BASE)

        if PREVIEW_MODE not in ("gen", "recon", "onestep"):
            PREVIEW_MODE = "gen"

        SEED = int(g("seed", SEED))

    except Exception as e:
        print("  [reload] No se pudo releer {} ({}); se mantiene la receta anterior.".format(
            CONFIG_PATH, e
        ))


_live_mtime = None


def reload_live_settings():
    """
    Ajustes en caliente: si train_settings.json ha cambiado (la GUI lo reescribe con
    Save JSON aunque el entrenamiento esté en marcha), aplica la lista de abajo en el
    paso siguiente. El coste por paso es un getmtime. El resto de ajustes de preview se
    releen en cada preview; rank, alpha, resolución y carpetas necesitan Stop -> Resume.
    """
    global _live_mtime, TOTAL_STEPS, SAVE_EVERY, LR, MAX_GRAD_NORM
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, SEED

    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return []
    if _live_mtime is None or mtime == _live_mtime:
        # La primera llamada solo memoriza la fecha: el guardado previo al lanzamiento no es un cambio.
        _live_mtime = _live_mtime or mtime
        return []

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []  # guardado a medias: se vuelve a intentar en el paso siguiente
    _live_mtime = mtime

    changes = []

    def fresh(key, cast, current):
        if key not in raw:
            return current
        try:
            value = cast(raw[key])
        except (TypeError, ValueError):
            return current
        if value != current:
            changes.append("{}: {} -> {}".format(key, current, value))
        return value

    TOTAL_STEPS   = fresh("total_steps",   int,   TOTAL_STEPS)
    SAVE_EVERY    = fresh("save_every",    int,   SAVE_EVERY)
    LR            = fresh("lr",            float, LR)
    MAX_GRAD_NORM = fresh("max_grad_norm", float, MAX_GRAD_NORM)
    PREVIEW_EVERY = fresh("preview_every", int,   PREVIEW_EVERY)
    PREVIEW_STEPS = fresh("preview_steps", int,   PREVIEW_STEPS)
    PREVIEW_CFG   = fresh("preview_cfg",   float, PREVIEW_CFG)
    SEED          = fresh("seed",          int,   SEED)  # solo previews
    return changes


def _fmt_num(x):
    try:
        f = float(x)
    except Exception:
        return str(x)

    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))

    return ("{:.2f}".format(f)).rstrip("0").rstrip(".").replace(".", "p")


def preview_filename(step):
    parts = [
        "preview_step_{:05d}".format(int(step)),
        str(PREVIEW_MODE or "gen"),
    ]

    if PREVIEW_MODE != "gen":
        parts.append("sig{}".format(_fmt_num(PREVIEW_RECON_SIGMA)))

    parts.extend([
        "{}st".format(int(PREVIEW_STEPS)),
        "cfg{}".format(_fmt_num(PREVIEW_CFG)),
        "acfg{}".format(_fmt_num(PREVIEW_AUDIO_CFG)),
    ])

    if abs(float(PREVIEW_LORA_SCALE) - 1.0) > 1e-6:
        parts.append("ls{}".format(_fmt_num(PREVIEW_LORA_SCALE)))

    return "_".join(parts) + ".png"


def run_preview_diagnostic(
    model,
    scheduler,
    entries,
    special_texts,
    step,
    audio_channels,
):
    if not entries:
        return

    if special_texts is None:
        special_texts = {}

    reload_preview_settings()

    was_training = model.training
    model.eval()

    vae = None

    try:
        valid_entries = [e for e in entries if not e["name"].startswith("_")]
        if not valid_entries:
            valid_entries = entries

        if PREVIEW_SAMPLE_NAME:
            candidates = [e for e in valid_entries if e["name"] == PREVIEW_SAMPLE_NAME]
            entry = candidates[0] if candidates else valid_entries[0]
        elif PREVIEW_CAPTION_MODE == "random":
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

        device = "cuda"

        patch_size = int(getattr(CURRENT_CONFIG, "patch_size", 1))
        patch_size_t = int(getattr(CURRENT_CONFIG, "patch_size_t", 1))

        video_clean, audio_clean = prepare_preview_tensors(entry, device)
        video_clean = align_video_latent_to_patch(video_clean, patch_size, patch_size_t)

        video_text = video_text.to(device, dtype=torch.bfloat16)
        audio_text = audio_text.to(device, dtype=torch.bfloat16)

        if video_text.ndim == 2:
            video_text = video_text.unsqueeze(0)

        if audio_text.ndim == 2:
            audio_text = audio_text.unsqueeze(0)

        eff_cfg = min(float(PREVIEW_CFG), float(PREVIEW_CFG_MAX))
        eff_audio_cfg = float(PREVIEW_AUDIO_CFG)

        neg_video_text = None
        neg_audio_text = None

        if eff_cfg > 1.0 and "_neg" in special_texts:
            neg_video_text, neg_audio_text = special_texts["_neg"]

            neg_video_text = neg_video_text.to(device, dtype=torch.bfloat16)
            neg_audio_text = neg_audio_text.to(device, dtype=torch.bfloat16)

            if neg_video_text.ndim == 2:
                neg_video_text = neg_video_text.unsqueeze(0)

            if neg_audio_text.ndim == 2:
                neg_audio_text = neg_audio_text.unsqueeze(0)
        else:
            eff_cfg = 1.0
            eff_audio_cfg = 1.0

        if SEED > 0:
            preview_seed = SEED
        else:
            preview_seed = random.randint(1, 2147483647)

        sigma_start = min(max(float(PREVIEW_RECON_SIGMA), 0.05), 1.0)
        shift = max(0.1, float(PREVIEW_SHIFT))

        print()
        print("  [Preview-Diag] Mode: {} | Sample: {}".format(PREVIEW_MODE, sample_name))
        print("  -> Seed: {}".format(preview_seed))
        print("  -> Steps: {}".format(PREVIEW_STEPS))
        print("  -> CFG video/audio: {:.2f} / {:.2f}".format(eff_cfg, eff_audio_cfg))
        print("  -> CFG rescale: {:.2f}".format(PREVIEW_CFG_RESCALE))
        print("  -> LoRA scale: {:.2f}".format(PREVIEW_LORA_SCALE))
        print("  -> Shift: {:.2f}".format(shift))
        print("  -> Negative active: {}".format(neg_video_text is not None))
        print("  -> Compare BASE vs LoRA: {}".format("ON" if PREVIEW_COMPARE_BASE else "OFF"))

        if PREVIEW_MODE != "gen":
            print("  -> Recon sigma: {:.2f}".format(sigma_start))

        def _run_mode(scale_factor):
            applied_local = False

            try:
                if abs(float(scale_factor) - 1.0) > 1e-6:
                    applied_local = apply_preview_lora_scale(model, float(scale_factor))

                    if applied_local:
                        print("  -> Preview LoRA scale temporal: {:.2f}".format(float(scale_factor)))

                if PREVIEW_MODE == "onestep":
                    return run_onestep_x0_preview(
                        model=model,
                        video_clean=video_clean,
                        audio_clean=audio_clean,
                        video_text=video_text,
                        audio_text=audio_text,
                        neg_video_text=neg_video_text,
                        neg_audio_text=neg_audio_text,
                        audio_channels=audio_channels,
                        patch_size=patch_size,
                        patch_size_t=patch_size_t,
                        device=device,
                        sigma_start=sigma_start,
                        cfg=eff_cfg,
                        rescale=PREVIEW_CFG_RESCALE,
                        cfg_audio=eff_audio_cfg,
                        seed=preview_seed,
                    )

                elif PREVIEW_MODE == "recon":
                    return run_recon_flow_preview(
                        model=model,
                        video_clean=video_clean,
                        audio_clean=audio_clean,
                        video_text=video_text,
                        audio_text=audio_text,
                        neg_video_text=neg_video_text,
                        neg_audio_text=neg_audio_text,
                        audio_channels=audio_channels,
                        patch_size=patch_size,
                        patch_size_t=patch_size_t,
                        device=device,
                        steps=PREVIEW_STEPS,
                        cfg=eff_cfg,
                        rescale=PREVIEW_CFG_RESCALE,
                        cfg_audio=eff_audio_cfg,
                        sigma_start=sigma_start,
                        shift=shift,
                        seed=preview_seed,
                    )

                else:
                    return run_fullgen_flow_preview(
                        model=model,
                        video_ref=video_clean,
                        audio_ref=audio_clean,
                        video_text=video_text,
                        audio_text=audio_text,
                        neg_video_text=neg_video_text,
                        neg_audio_text=neg_audio_text,
                        audio_channels=audio_channels,
                        patch_size=patch_size,
                        patch_size_t=patch_size_t,
                        device=device,
                        steps=PREVIEW_STEPS,
                        cfg=eff_cfg,
                        rescale=PREVIEW_CFG_RESCALE,
                        cfg_audio=eff_audio_cfg,
                        shift=shift,
                        seed=preview_seed,
                    )

            finally:
                if applied_local:
                    restore_preview_lora_scale(model)

        vae = LTXVaeHolder.get().to("cuda")

        if PREVIEW_COMPARE_BASE:
            print("  -> Generating BASE preview (LoRA scale 0.0)")
            latents_base = _run_mode(0.0)

            print("  -> Generating LoRA preview (LoRA scale {:.2f})".format(PREVIEW_LORA_SCALE))
            latents_lora = _run_mode(PREVIEW_LORA_SCALE)

            if latents_base is None or latents_lora is None:
                return

            with torch.no_grad():
                img_base = decode_preview_latent(
                    vae,
                    latents_base,
                    frame_index=PREVIEW_FRAME_INDEX,
                )

                img_lora = decode_preview_latent(
                    vae,
                    latents_lora,
                    frame_index=PREVIEW_FRAME_INDEX,
                )

            img = np.concatenate([img_base, img_lora], axis=1)

            vae.to("cpu")

            out_name = preview_filename(step).replace(".png", "_compare_BASE_LoRA.png")
            out_path = os.path.join(OUTPUT_DIR, out_name)

            Image.fromarray(img).save(out_path)

            print("  -> Compare preview saved: {}".format(out_path))

            free_vram(latents_base, latents_lora, img_base, img_lora, img)

        else:
            latents = _run_mode(PREVIEW_LORA_SCALE)

            if latents is None:
                return

            with torch.no_grad():
                img = decode_preview_latent(
                    vae,
                    latents,
                    frame_index=PREVIEW_FRAME_INDEX,
                )

            vae.to("cpu")

            out_path = os.path.join(OUTPUT_DIR, preview_filename(step))
            Image.fromarray(img).save(out_path)

            print("  -> Preview saved: {}".format(out_path))

            free_vram(latents, img)

    except Exception as e:
        print("  [!] Preview diagnostic failed: {}".format(e))
        traceback.print_exc()

    finally:
        if vae is not None:
            try:
                vae.to("cpu")
            except Exception:
                pass

        if was_training:
            model.train()

        free_vram()


# ===========================================================================
# GLOBAL CONFIG REFERENCE
# ===========================================================================
CURRENT_CONFIG = None
CURRENT_TRANSFORMER = None
CURRENT_AUDIO_CHANNELS = None


# ===========================================================================
# TRAIN
# ===========================================================================
def train_ltx23():
    global CURRENT_CONFIG
    global CURRENT_TRANSFORMER
    global CURRENT_AUDIO_CHANNELS
    global ACTIVATION_OFFLOAD_ACTIVE
    global LORA_RANK, LORA_ALPHA  # al reanudar se toman del checkpoint

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if LOW_VRAM_12GB:
        try:
            torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
        except Exception:
            pass

        try:
            if platform.system() != "Windows":
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception:
            pass

    if ACTIVATION_OFFLOAD and not _SAVE_ON_CPU_AVAILABLE:
        ACTIVATION_OFFLOAD_ACTIVE = False
        print("[VRAM] activation_offload=True pero torch.autograd.graph.save_on_cpu")
        print("       no está disponible (requiere PyTorch >= 2.1). Se desactiva.")
    else:
        ACTIVATION_OFFLOAD_ACTIVE = bool(ACTIVATION_OFFLOAD and _SAVE_ON_CPU_AVAILABLE)

    if ACTIVATION_OFFLOAD_ACTIVE:
        print("[VRAM] activation_offload ACTIVO: saved tensors -> CPU pinned.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_video_latent.pt") for f in os.listdir(CACHE_DIR)):
        print("\n[!] ERROR: Cache directory '{}' is empty or does not exist.".format(CACHE_DIR))
        print("[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        sys.exit(2)  # la GUI muestra este código como "falta la pre-caché"

    ensure_ltx23_model_downloaded(MODEL_ID)

    # Instalaciones anteriores traían el Transformer en BF16 en vez de others.safetensors.
    ensure_nf4_others_downloaded(MODEL_ID)

    print()
    print("Loading LTX-2.3 connectors... / Cargando connectors de LTX-2.3...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        transformer=None,
        vae=None,
        audio_vae=None,
        text_encoder=None,
        tokenizer=None,
        processor=None,
        vocoder=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    connectors = getattr(pipe, "connectors", None)
    scheduler = getattr(pipe, "scheduler", None)

    CURRENT_CONFIG = FrozenDict(LTX2VideoTransformer3DModel.load_config(MODEL_ID))

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

    print()
    print("Loading LTX-2.3 Transformer (NF4)... / Cargando Transformer de LTX-2.3 (NF4)...")

    t0 = time.time()

    transformer = load_nf4_transformer(MODEL_ID)
    CURRENT_CONFIG = transformer.config
    CURRENT_TRANSFORMER = transformer

    transformer.requires_grad_(False)

    if CAST_FROZEN_BF16:
        cast_frozen_to_bf16(transformer)

    transformer.to("cuda")
    free_vram()

    print("[NF4] Cache loaded in / Cache cargada en {:.1f}s".format(time.time() - t0))
    print("Transformer pinned in VRAM. Usage / Uso: {:.1f} GB".format(torch.cuda.memory_allocated() / 1e9))

    enable_memory_efficient_attention(transformer)

    if hasattr(transformer, "enable_gradient_checkpointing"):
        try:
            transformer.enable_gradient_checkpointing()
            print("Gradient checkpointing activado.")
        except Exception:
            pass

    target_modules = discover_lora_targets(transformer)
    print("Target LoRA Layers / Capas LoRA objetivo: {}".format(len(target_modules)))

    # Reanudar es continuar EL MISMO LoRA: capas, rank y alpha salen del checkpoint, no de la GUI.
    # Si se construyera con otros, el optimizador no encajaría y quedaría una mezcla de pesos.
    # Las capas se leen de los PESOS guardados: adapter_config.json las guarda abreviadas.
    resume_weights = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    if os.path.exists(STEP_FILE) and os.path.exists(resume_weights):
        with safe_open(resume_weights, framework="pt", device="cpu") as f:
            saved_targets = sorted({k.split(".lora_")[0].replace("base_model.model.", "", 1) for k in f.keys()})
        with open(os.path.join(RESUME_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        saved = (len(saved_targets), saved_cfg["r"], saved_cfg["lora_alpha"])
        if saved != (len(target_modules), LORA_RANK, LORA_ALPHA):
            print("\n[!] Resuming with the checkpoint's LoRA: {} layers, rank {}, alpha {} "
                  "(settings ask for {} layers, rank {}, alpha {}; they apply to new trainings).".format(
                      *saved, len(target_modules), LORA_RANK, LORA_ALPHA))
            print("[!] Se reanuda con el LoRA del checkpoint: {} capas, rank {}, alpha {} "
                  "(los ajustes piden {} capas, rank {}, alpha {}; se aplican a entrenamientos nuevos).".format(
                      *saved, len(target_modules), LORA_RANK, LORA_ALPHA))
        target_modules, LORA_RANK, LORA_ALPHA = saved_targets, saved[1], saved[2]

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
        if not torch.is_grad_enabled():
            return

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
    start_step = 0

    adapter_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")

    if os.path.exists(adapter_path) and os.path.exists(STEP_FILE):
        print("=" * 65)
        print("Checkpoint detected! Restoring state... / Checkpoint detectado! Restaurando estado...")

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

            print("Resuming training from step / Reanudando entrenamiento desde el paso {}...".format(start_step))

        except Exception as e:
            print("[!] Warning reading checkpoint / Advertencia al leer checkpoint: {}".format(e))
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
        print("Saving checkpoint state at step / Guardando estado en paso {}...".format(current_s))

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
            ckpt = os.path.join(OUTPUT_DIR, "LTX23_LoRA_step_{}.safetensors".format(current_s))
            save_lora(model, ckpt, current_s)
            print("Checkpoint saved successfully at step / Checkpoint guardado en paso {}: {}".format(current_s, ckpt))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------
    def handle_signal(sig, frame):
        nonlocal last_step_executed

        print()
        print("Signal received / Señal de detención recibida ({}).".format(sig))

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
    print("STARTING TRAINING / ARRANCANDO ENTRENAMIENTO! {} entradas cacheadas.".format(len(entries)))
    #print("LoRA export prefix / Prefijo de exportación: '{}'".format(LORA_KEY_PREFIX))

    reload_live_settings()
    step = start_step
    try:
        # while y no range(): TOTAL_STEPS puede cambiar en caliente, en ambos sentidos.
        while step < TOTAL_STEPS:
            step += 1

            changes = reload_live_settings()
            if changes:
                print("\n[LIVE] Settings reloaded without stopping / Ajustes recargados sin parar:")
                for c in changes:
                    print("[LIVE]   {}".format(c))
                if step > TOTAL_STEPS:
                    break

            last_step_executed = step
            t0 = time.time()

            loss_video = None
            loss_audio = None

            entry = random.choice(entries)

            video_clean = entry["video"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            audio_clean = entry["audio"].to("cuda", dtype=torch.bfloat16, non_blocking=True)

            if video_clean.ndim == 4:
                video_clean = video_clean.unsqueeze(0)

            if audio_clean.ndim == 2:
                audio_clean = audio_clean.unsqueeze(0)

            video_text = entry["video_text"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            audio_text = entry["audio_text"].to("cuda", dtype=torch.bfloat16, non_blocking=True)

            if video_text.ndim == 2:
                video_text = video_text.unsqueeze(0)

            if audio_text.ndim == 2:
                audio_text = audio_text.unsqueeze(0)

            patch_size = int(getattr(CURRENT_CONFIG, "patch_size", 1))
            patch_size_t = int(getattr(CURRENT_CONFIG, "patch_size_t", 1))

            video_clean = align_video_latent_to_patch(video_clean, patch_size, patch_size_t)

            video_tokens = patch_video_latent(video_clean, patch_size, patch_size_t)
            audio_tokens = patch_audio_latent(audio_clean)

            B = video_tokens.shape[0]
            video_seq_len = video_tokens.shape[1]

            num_frames = video_clean.shape[2]
            height = video_clean.shape[3]
            width = video_clean.shape[4]

            audio_num_frames = audio_clean.shape[-1]

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
                video_seq_len,
                "cuda",
                torch.bfloat16,
            )

            audio_timestep = (
                sigma * float(getattr(CURRENT_CONFIG, "timestep_scale_multiplier", 1000))
            ).to(torch.bfloat16)

            del video_clean, audio_clean, video_tokens, audio_tokens
            del noise_video, noise_audio, t_video, t_audio

            video_clean = None
            audio_clean = None
            video_tokens = None
            audio_tokens = None
            noise_video = None
            noise_audio = None

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
                "audio_num_frames": audio_num_frames,
                "return_dict": False,
            }

            forward_kwargs = filter_forward_kwargs(forward_kwargs, transformer.forward)

            if ACTIVATION_OFFLOAD_ACTIVE:
                try:
                    with _save_on_cpu_ctx(pin_memory=True):
                        output = model(**forward_kwargs)
                except Exception as e_offload:
                    print()
                    print("[VRAM] save_on_cpu falló en runtime ({}).".format(e_offload))
                    print("       Se desactiva activation_offload y se reintenta el step sin offload.")

                    ACTIVATION_OFFLOAD_ACTIVE = False
                    output = model(**forward_kwargs)
            else:
                output = model(**forward_kwargs)

            pred_video = None
            pred_audio = None

            if isinstance(output, tuple):
                if len(output) == 0:
                    raise RuntimeError("LTX-2.3 forward devolvió tupla vacía.")

                pred_video = output[0]

                if len(output) > 1 and USE_AUDIO_LOSS:
                    pred_audio = output[1]
            else:
                pred_video = getattr(output, "video", None)

                if pred_video is None:
                    pred_video = getattr(output, "sample", None)

                if USE_AUDIO_LOSS:
                    pred_audio = getattr(output, "audio", None)

                    if pred_audio is None:
                        pred_audio = getattr(output, "audio_sample", None)

            if pred_video is None:
                raise RuntimeError("No se pudo obtener predicción de video.")

            output = None

            if not USE_AUDIO_LOSS:
                pred_audio = None

            if USE_AUDIO_LOSS and pred_audio is not None and target_audio is not None:
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
            eta = "{:02d}:{:02d}:{:02d}".format(
                int(eta_s // 3600),
                int((eta_s % 3600) // 60),
                int(eta_s % 60),
            )

            pct = step / TOTAL_STEPS
            barra = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))

            avg_loss = running_loss / max(1, step - start_step)

            progress_line = (
                "Step/Paso {:4d}/{} [{}] {:5.1f}% | "
                "Loss {:.4f} | gnorm {:.3f} | "
                "lr {:.2e} | {:.2f}s/it | ETA {}".format(
                    step,
                    TOTAL_STEPS,
                    barra,
                    pct * 100,
                    avg_loss,
                    grad_norm,
                    current_lr,
                    avg_time,
                    eta,
                )
            )

            print("\r{}".format(progress_line), end="", flush=True)

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                save_checkpoint_now(step)

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

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                run_preview_diagnostic(
                    model,
                    scheduler,
                    entries,
                    special_texts,
                    step,
                    AUDIO_CHANNELS,
                )

    except KeyboardInterrupt:
        save_checkpoint_now(last_step_executed)
        return

    except SystemExit:
        return

    print()
    print()
    print("Training completed! / Entrenamiento finalizado!")

    # Pasos realmente entrenados: si se bajan los pasos en caliente por debajo del actual, se para aquí.
    save_checkpoint_now(last_step_executed)

    final_path = os.path.join(OUTPUT_DIR, "LTX23_FINAL_LoRA.safetensors")
    save_lora(model, final_path, last_step_executed)

    print("Final LoRA saved to / Tu LoRA definitivo está en: {}".format(final_path))
    #print("Formato exportado: prefijo='{}', scaling horneado -> usa strength=1.0 en ComfyUI.".format(LORA_KEY_PREFIX))

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