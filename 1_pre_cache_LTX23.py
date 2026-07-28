# -*- coding: utf-8 -*-
"""
1_pre_cache_LTX23.py

Pre-cache de LTX-2.3 para entrenamiento LoRA (dataset de imágenes).

OPTIMIZACIÓN DE VRAM:
  El text encoder de LTX-2.3 es enorme y NO cabe en GPUs de 16 GB.
  Antes se hacía pipe.to("cuda") y se saturaba la VRAM (swap -> lentísimo).
  Ahora el pipeline se carga en CPU y se aplica offload según
  `precache_offload`:
    - "none"      : todo en VRAM (solo si tienes VRAM de sobra).
    - "model"     : 1 componente en VRAM cada vez (solo si cada uno cabe).
    - "sequential": capa por capa en VRAM (RECOMENDADO en 16 GB).
    - "cpu"       : text encoder en CPU, VAE en VRAM (lento pero 0 VRAM texto).
  Y con `text_encoder_4bit: true` se intenta cuantizar el text encoder a
  4-bit para que quepa entero en VRAM (experimental, con fallback).
"""

import os
import gc
import json
import math
import sys
import traceback

import torch
import torchvision.transforms.functional as F_vision

from PIL import Image
from diffusers import DiffusionPipeline


# ============================================================================
# CONFIG
# ============================================================================

DEFAULTS = {
    "model_id": "./LTX23-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_ltx23",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 32,
    "max_seq_len": 1024,
    "frame_rate": 24.0,
    "num_frames": 1,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",

    # --- NUEVAS: gestión de VRAM del pre-cache ---
    "precache_offload": "sequential",   # none | model | sequential | cpu
    "text_encoder_4bit": True,         # experimental: cuantiza text encoder a 4-bit
}

CONFIG_PATH = "pre_cache_settings.json"


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


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


MODEL_ID = str(cfg_get("model_id", DEFAULTS["model_id"])).strip()
DATASET_PATH = str(cfg_get("dataset_path", DEFAULTS["dataset_path"])).strip()
TARGET_AREA = int(cfg_get("target_area", DEFAULTS["target_area"]))
MAX_SIDE = int(cfg_get("max_side", DEFAULTS["max_side"]))
MULTIPLE = int(cfg_get("multiple", DEFAULTS["multiple"]))
MAX_SEQ_LEN = int(cfg_get("max_seq_len", DEFAULTS["max_seq_len"]))
FRAME_RATE = float(cfg_get("frame_rate", DEFAULTS["frame_rate"]))
NUM_FRAMES = int(cfg_get("num_frames", DEFAULTS["num_frames"]))
TRIGGER_WORD = str(cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
PROJECT_NAME = str(cfg_get("project_name", DEFAULTS["project_name"])).strip()
PREVIEW_CUSTOM_PROMPT = str(cfg_get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip()

PRECACHE_OFFLOAD = str(cfg_get("precache_offload", DEFAULTS["precache_offload"])).strip().lower()
TEXT_ENCODER_4BIT = bool(cfg_get("text_encoder_4bit", DEFAULTS["text_encoder_4bit"]))

if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_ltx23_{PROJECT_NAME}"
else:
    CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()

# LTX-2.3 exige dimensiones divisibles por 32.
MULTIPLE = max(32, MULTIPLE)


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


def vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def vram_peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def read_audio_channels(model_id, default=128):
    """Lee audio_in_channels del config.json del transformer en disco (0 VRAM)."""
    for rel in ("transformer/config.json", os.path.join("transformer", "config.json")):
        p = os.path.join(model_id, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = json.load(f)
                return int(c.get("audio_in_channels", default))
            except Exception:
                pass
    return default


def try_quantize_text_encoder_4bit(pipe):
    """
    Intenta recargar el text encoder cuantizado a 4-bit (nf4) para que quepa
    entero en VRAM. Si falla por cualquier motivo, devuelve False y el pipe
    se queda con su text encoder original en CPU.
    """
    te = getattr(pipe, "text_encoder", None)
    if te is None:
        print("[4bit] El pipeline no tiene text_encoder; se omite.")
        return False

    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:
        print("[4bit] transformers.BitsAndBytesConfig no disponible:", e)
        return False

    te_cls = type(te)
    subfolder = "text_encoder"

    # Probar varios nombres de subcarpeta por si el layout cambia.
    candidates = [subfolder, "text_encoder_2", ""]

    for sub in candidates:
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            kwargs = dict(
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if sub:
                kwargs["subfolder"] = sub

            print(f"[4bit] Intentando cuantizar text_encoder ({te_cls.__name__}, subfolder={sub or '<raíz>'})...")

            new_te = te_cls.from_pretrained(MODEL_ID, **kwargs)

            pipe.text_encoder = new_te
            print("[4bit] OK: text encoder cuantizado a 4-bit y residente en VRAM.")
            return True

        except Exception as e:
            print(f"[4bit] Fallo con subfolder={sub or '<raíz>'}: {e}")
            continue

    print("[4bit] No se pudo cuantizar el text encoder; se usará el modo de offload configurado.")
    return False


def _patch_module_to_noop_device(module):
    """
    Neutraliza module.to(...) para cambios de DEVICE (no de dtype).
    Evita que encode_prompt haga text_encoder.to("cuda") y dispare OOM
    cuando usamos offload secuencial (cuyos hooks ya mueven las hojas).
    """
    if module is None:
        return
    if getattr(module, "_ltx_to_patched", False):
        return

    orig_to = module.to

    def _to(*args, **kwargs):
        dtype = kwargs.get("dtype", None)
        for a in args:
            if isinstance(a, torch.dtype):
                dtype = a

        # Si piden dtype, lo aplicamos SIN tocar el device.
        if dtype is not None:
            try:
                return orig_to(dtype=dtype)
            except Exception:
                return module

        # Cambio de device -> ignorado (los hooks de offload lo gestionan).
        return module

    module.to = _to
    module._ltx_to_patched = True


def setup_offload(pipe, mode):
    """
    Configura la ubicación de los componentes y devuelve el device de texto
    que debe usarse en encode_prompt ("cuda" o "cpu").
    """
    mode = (mode or "sequential").lower()

    vae = getattr(pipe, "vae", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    connectors = getattr(pipe, "connectors", None)

    if mode == "none":
        # Comportamiento original: todo en VRAM (solo con VRAM de sobra).
        pipe.to("cuda")
        print("[OFFLOAD] none -> pipeline completo en VRAM.")
        print(f"[VRAM] tras pipe.to(cuda): {vram_gb():.2f} GB")
        return "cuda"

    if mode == "cpu":
        # Text encoder en CPU (0 VRAM de texto), VAE en VRAM (cabe).
        if vae is not None:
            vae.to("cuda")
        print("[OFFLOAD] cpu -> text encoder en CPU, VAE en VRAM.")
        print(f"[VRAM] tras mover VAE: {vram_gb():.2f} GB")
        return "cpu"

    if mode == "model":
        try:
            pipe.enable_model_cpu_offload(device="cuda")
            print("[OFFLOAD] model -> 1 componente en VRAM cada vez.")
            print("[OFFLOAD] (solo válido si cada componente cabe solo en VRAM)")
            return "cuda"
        except Exception as e:
            print("[OFFLOAD] model falló, fallback a sequential:", e)
            mode = "sequential"

    # mode == "sequential"
    try:
        pipe.enable_sequential_cpu_offload(device="cuda")
        # Blindar text_encoder y connectors contra .to("cuda") internos.
        _patch_module_to_noop_device(text_encoder)
        _patch_module_to_noop_device(connectors)
        print("[OFFLOAD] sequential -> text encoder capa por capa en VRAM.")
        print(f"[VRAM] pico tras setup: {vram_peak_gb():.2f} GB")
        return "cuda"
    except Exception as e:
        print("[OFFLOAD] sequential falló, fallback a cpu:", e)
        if vae is not None:
            vae.to("cuda")
        return "cpu"


def json_safe(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if torch.is_tensor(value):
        return {"tensor": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bucket_size(width, height):
    ar = width / height
    bh = math.sqrt(TARGET_AREA / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        scale = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * scale) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * scale) // MULTIPLE * MULTIPLE)
    return int(bw), int(bh)


def read_prompt(base_name):
    path = os.path.join(DATASET_PATH, base_name + ".txt")
    prompt = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def save_prompt_result(result, prefix):
    def recurse(obj, path):
        if torch.is_tensor(obj):
            filename = f"{prefix}_" + path.replace(".", "_") + ".pt"
            torch.save(obj.detach().cpu(), os.path.join(CACHE_DIR, filename))
            return {"type": "tensor", "file": filename, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, dict):
            return {"type": "dict", "items": {str(k): recurse(v, f"{path}_{k}") for k, v in obj.items()}}
        if isinstance(obj, (tuple, list)):
            return {"type": "tuple" if isinstance(obj, tuple) else "list", "items": [recurse(v, f"{path}_{i}") for i, v in enumerate(obj)]}
        return {"type": "value", "value": json_safe(obj)}

    structure = recurse(result, "root")
    atomic_json(structure, os.path.join(CACHE_DIR, f"{prefix}_structure.json"))
    return structure


def extract_prompt_tensors(result):
    found = []

    def recurse(obj, path="root"):
        if torch.is_tensor(obj):
            found.append((path, obj.detach().cpu()))
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, f"{path}.{k}")
            return
        if isinstance(obj, (tuple, list)):
            for i, v in enumerate(obj):
                recurse(v, f"{path}.{i}")

    recurse(result)
    return found


def encode_prompt(pipe, prompt, text_device):
    result = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        max_sequence_length=MAX_SEQ_LEN,
        device=torch.device(text_device),
        dtype=torch.bfloat16,
    )
    return result


def encode_video_latent(vae, image):
    image_tensor = F_vision.pil_to_tensor(image).float() / 127.5 - 1.0
    image_tensor = (
        image_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, NUM_FRAMES, 1, 1)
    ).to("cuda", dtype=torch.bfloat16)

    encoded = vae.encode(image_tensor)

    if hasattr(encoded, "latent_dist"):
        latent = encoded.latent_dist.sample()
    elif torch.is_tensor(encoded):
        latent = encoded
    elif isinstance(encoded, tuple):
        latent = encoded[0]
    else:
        raise RuntimeError("Salida desconocida de VAE.encode(): " + str(type(encoded)))

    latent = latent.detach()

    # IMPORTANTE: el VAE de LTX-2.3 (AutoencoderKLLTX2) no produce un
    # espacio latente de varianza ~1. El pipeline oficial SIEMPRE
    # normaliza el latente crudo antes de pasarlo al transformer:
    #
    #   latents = (latents - latents_mean) * scaling_factor / latents_std
    #
    # (ver diffusers/pipelines/ltx2/pipeline_ltx2.py: _normalize_latents).
    # Sin este paso, el transformer (preentrenado sobre latentes YA
    # normalizados) recibe un "clean" con escala/varianza equivocada:
    # el objetivo de flow-matching (noise - clean) queda descalibrado
    # y el modelo aprende a denoisear un dominio distinto del real.
    latents_mean = vae.latents_mean.to(
        device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1, 1)

    latents_std = vae.latents_std.to(
        device=latent.device, dtype=latent.dtype
    ).view(1, -1, 1, 1, 1)

    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

    latent = (latent - latents_mean) * scaling_factor / latents_std

    return latent.detach().to(torch.bfloat16).cpu().contiguous()


def make_audio_latent(video_latent, audio_channels):
    return torch.zeros((video_latent.shape[0], audio_channels, 1), dtype=torch.bfloat16)


# ============================================================================
# MAIN
# ============================================================================

def preprocess_ltx23():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    if not os.path.isdir(MODEL_ID):
        raise FileNotFoundError(f"No existe el modelo: {MODEL_ID}")

    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    images = sorted(
        f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )

    if not images:
        raise RuntimeError(f"No hay imágenes en {DATASET_PATH}")

    print()
    print("=" * 80)
    print(" LTX-2.3 PRE-CACHE")
    print("=" * 80)
    print("Model        :", os.path.abspath(MODEL_ID))
    print("Dataset      :", os.path.abspath(DATASET_PATH))
    print("Cache        :", os.path.abspath(CACHE_DIR))
    print("Target area  :", TARGET_AREA)
    print("Multiple     :", MULTIPLE)
    print("Frames       :", NUM_FRAMES)
    print("FPS          :", FRAME_RATE)
    print("Max seq len  :", MAX_SEQ_LEN)
    print("Offload mode :", PRECACHE_OFFLOAD)
    print("TextEnc 4bit :", TEXT_ENCODER_4BIT)
    print("=" * 80)

    # Canales de audio desde disco (sin cargar el transformer).
    audio_channels = read_audio_channels(MODEL_ID, 128)
    print("Audio latent channels:", audio_channels)

    # ------------------------------------------------------------------
    # Cargar pipeline en CPU (SIN .to("cuda")).
    # ------------------------------------------------------------------
    print()
    print("Cargando LTX-2.3 en CPU (sin transformer)...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        transformer=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Pipeline:", type(pipe).__name__)
    print("VAE:", type(getattr(pipe, "vae", None)).__name__)
    print("Text encoder:", type(getattr(pipe, "text_encoder", None)).__name__)
    print("Audio VAE:", type(getattr(pipe, "audio_vae", None)).__name__)

    # ------------------------------------------------------------------
    # Opcional: cuantizar text encoder a 4-bit (experimental).
    # ------------------------------------------------------------------
    used_4bit = False
    if TEXT_ENCODER_4BIT:
        used_4bit = try_quantize_text_encoder_4bit(pipe)

    # ------------------------------------------------------------------
    # Offload. Si 4bit tuvo éxito, el text encoder ya está en VRAM y
    # basta con poner el VAE también (modo "none"-like pero sin tocar
    # el text encoder). Si no, aplicamos el modo configurado.
    # ------------------------------------------------------------------
    if used_4bit:
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            vae.to("cuda")
        text_device = "cuda"
        print("[OFFLOAD] 4bit activo -> text encoder 4-bit + VAE en VRAM.")
        print(f"[VRAM] tras 4bit + VAE: {vram_gb():.2f} GB")
    else:
        text_device = setup_offload(pipe, PRECACHE_OFFLOAD)

    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # NEGATIVE PROMPT
    # ------------------------------------------------------------------
    print("\nEncoding negative/empty prompt...")
    with torch.inference_mode():
        neg_result = encode_prompt(pipe, "", text_device)

    save_prompt_result(neg_result, "_neg")

    print("Negative prompt tensors:")
    for path, tensor in extract_prompt_tensors(neg_result):
        print(" ", path, tuple(tensor.shape), tensor.dtype)

    print(f"[VRAM] pico tras neg prompt: {vram_peak_gb():.2f} GB")

    free_vram(neg_result)

    # ------------------------------------------------------------------
    # CUSTOM PROMPT
    # ------------------------------------------------------------------
    if PREVIEW_CUSTOM_PROMPT:
        custom_prompt = PREVIEW_CUSTOM_PROMPT
        if TRIGGER_WORD and TRIGGER_WORD.lower() not in custom_prompt.lower():
            custom_prompt = f"{TRIGGER_WORD}, {custom_prompt}".strip(", ")

        print("\nEncoding custom prompt:", custom_prompt)

        with torch.inference_mode():
            custom_result = encode_prompt(pipe, custom_prompt, text_device)

        save_prompt_result(custom_result, "_custom")
        free_vram(custom_result)

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------
    for idx, filename in enumerate(images, start=1):
        base = os.path.splitext(filename)[0]

        video_path = os.path.join(CACHE_DIR, f"{base}_video_latent.pt")
        audio_path = os.path.join(CACHE_DIR, f"{base}_audio_latent.pt")
        prompt_structure_path = os.path.join(CACHE_DIR, f"{base}_prompt_structure.json")

        if (
            os.path.exists(video_path)
            and os.path.exists(audio_path)
            and os.path.exists(prompt_structure_path)
        ):
            print(f"[{idx}/{len(images)}] SKIP {filename}")
            continue

        print()
        print("=" * 80)
        print(f"[{idx}/{len(images)}] {filename}")

        image = Image.open(os.path.join(DATASET_PATH, filename)).convert("RGB")

        bw, bh = bucket_size(image.width, image.height)
        scale = max(bw / image.width, bh / image.height)

        image = image.resize(
            (math.ceil(image.width * scale), math.ceil(image.height * scale)),
            Image.LANCZOS,
        )

        left = (image.width - bw) // 2
        top = (image.height - bh) // 2
        image = image.crop((left, top, left + bw, top + bh))

        print("Bucket:", f"{bw}x{bh}")

        # VIDEO VAE
        print("Encoding video latent...")
        with torch.inference_mode():
            video_latent = encode_video_latent(pipe.vae, image)

        torch.save(video_latent, video_path)
        print("Video latent:", tuple(video_latent.shape))

        # AUDIO LATENT
        audio_latent = make_audio_latent(video_latent, audio_channels)
        torch.save(audio_latent, audio_path)
        print("Audio latent:", tuple(audio_latent.shape))

        # TEXT
        prompt = read_prompt(base)
        print("Prompt:", prompt)

        with torch.inference_mode():
            prompt_result = encode_prompt(pipe, prompt, text_device)

        structure = save_prompt_result(prompt_result, f"{base}_prompt")

        print("Prompt result:")
        for path, tensor in extract_prompt_tensors(prompt_result):
            print(" ", path, tuple(tensor.shape), tensor.dtype)

        print(f"[VRAM] pico acumulado: {vram_peak_gb():.2f} GB")

        atomic_json(
            {
                "filename": filename,
                "width": bw,
                "height": bh,
                "num_frames": NUM_FRAMES,
                "frame_rate": FRAME_RATE,
                "prompt": prompt,
                "video_latent": os.path.basename(video_path),
                "audio_latent": os.path.basename(audio_path),
                "prompt_structure": structure,
            },
            os.path.join(CACHE_DIR, f"{base}_info.json"),
        )

        free_vram(prompt_result, audio_latent)

    # ------------------------------------------------------------------
    # CACHE INFO
    # ------------------------------------------------------------------
    cache_info = {
        "format": "LTX23-LoRA-Precache",
        "version": 1,
        "model_id": MODEL_ID,
        "dataset_path": DATASET_PATH,
        "cache_dir": CACHE_DIR,
        "target_area": TARGET_AREA,
        "multiple": MULTIPLE,
        "frame_rate": FRAME_RATE,
        "num_frames": NUM_FRAMES,
        "max_sequence_length": MAX_SEQ_LEN,
        "trigger_word": TRIGGER_WORD,
        "audio_latent_channels": audio_channels,
        "precache_offload": PRECACHE_OFFLOAD,
        "text_encoder_4bit": bool(used_4bit),
        "prompt_encoding": "LTX2Pipeline.encode_prompt",
        "note": "Image dataset cache. Audio latent is zero-filled minimal conditioning.",
    }

    atomic_json(cache_info, os.path.join(CACHE_DIR, "cache_info.json"))

    free_vram(pipe)

    print()
    print("=" * 80)
    print("LTX-2.3 PRE-CACHE COMPLETADO")
    print("=" * 80)
    print("Cache:", os.path.abspath(CACHE_DIR))
    print("Imágenes:", len(images))
    print(f"VRAM pico total: {vram_peak_gb():.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    try:
        preprocess_ltx23()
    except Exception:
        print()
        print("=" * 80)
        print("ERROR EN PRE-CACHE LTX-2.3")
        print("=" * 80)
        traceback.print_exc()
        raise