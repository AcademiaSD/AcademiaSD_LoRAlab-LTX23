# -*- coding: utf-8 -*-
"""
1_pre_cache_LTX23.py
Pre-cache de LTX-2.3 para entrenamiento LoRA (dataset de imágenes).

OPTIMIZACIÓN DE VRAM:
El text encoder de LTX-2.3 es enorme y NO cabe en GPUs de 16 GB.
Antes se hacía pipe.to("cuda") y se saturaba la VRAM (swap -> lentísimo).
Ahora el pipeline se carga en CPU y se aplica offload según
`precache_offload`:

"none"      : todo en VRAM (solo si tienes VRAM de sobra).
"model"     : 1 componente en VRAM cada vez (solo si cada uno cabe).
"sequential": capa por capa en VRAM (RECOMENDADO en 16 GB).
"cpu"       : text encoder en CPU, VAE en VRAM (lento pero 0 VRAM texto).

Y con `text_encoder_4bit: true` se intenta cuantizar el text encoder a
4-bit para que quepa entero en VRAM (experimental, con fallback).

CAMBIOS PARA BAJA RAM:
- Detección de RAM física.
- Modo LOW_RAM automático si la RAM es menor o igual a low_ram_threshold_gb.
- En LOW_RAM se fuerza sequential offload y text encoder 4-bit.
- Se limita la CPU que puede usar Accelerate.
- Se usa offload_folder para permitir apoyo en disco.
- Se evita el fallback BF16 completo en CPU como primera opción.
- Si el 4-bit falla, se intenta BF16 con device_map auto + disk offload.
"""

import os
import gc
import json
import math
import sys
import traceback
import importlib
import shutil
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DiffusionPipeline


# ============================================================================
# DETECCIÓN DE RAM FÍSICA
# ============================================================================
def _detect_system_ram_gb():
    """
    Intenta detectar la RAM física total.
    Usa psutil si está instalado; si no, usa /proc/meminfo o Windows API.
    """
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        except Exception:
            pass
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            pass

    return None


SYSTEM_RAM_GB = _detect_system_ram_gb()


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

    # --- Gestión de VRAM del pre-cache ---
    "precache_offload": "sequential",   # none | model | sequential | cpu
    "text_encoder_4bit": True,          # experimental: cuantiza text encoder a 4-bit

    # --- Baja RAM ---
    # Si la RAM física detectada es <= este valor, activa modo conservador.
    # 48 GB es recomendable si el pico actual está cerca de 64 GB.
    # Si solo quieres activarlo en sistemas de 32 GB o menos, pon 32.
    "low_ram_threshold_gb": 48.0,

    # Último recurso en LOW_RAM si falla disk offload:
    # cargar BF16 completo en CPU. Puede usar pagefile/swap, pero puede dar OOM.
    "low_ram_allow_cpu_fallback": False,
}

CONFIG_PATH = "pre_cache_settings.json"

HF_BASE_REPO_ID = "diffusers/LTX-2.3-Diffusers"
HF_NF4_REPO_ID = "AcademiaSD/LTX23_NF4"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

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
    if not isinstance(cfg, dict):
        return default

    candidates = [
        key,
        key + " ",
        " " + key,
        " " + key + " ",
    ]

    for candidate in candidates:
        if candidate in cfg:
            return cfg[candidate]

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

PRECACHE_OFFLOAD = str(
    cfg_get("precache_offload", DEFAULTS["precache_offload"])
).strip().lower()

TEXT_ENCODER_4BIT = _cfg_bool(
    "text_encoder_4bit",
    DEFAULTS["text_encoder_4bit"]
)

LOW_RAM_THRESHOLD_GB = float(
    cfg_get("low_ram_threshold_gb", DEFAULTS["low_ram_threshold_gb"])
)

LOW_RAM_ALLOW_CPU_FALLBACK = _cfg_bool(
    "low_ram_allow_cpu_fallback",
    DEFAULTS["low_ram_allow_cpu_fallback"]
)

LOW_RAM_MODE = bool(
    LOW_RAM_THRESHOLD_GB > 0
    and SYSTEM_RAM_GB is not None
    and SYSTEM_RAM_GB <= LOW_RAM_THRESHOLD_GB
)

if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_ltx23_{PROJECT_NAME}"
else:
    CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()

# LTX-2.3 exige dimensiones divisibles por 32.
MULTIPLE = max(32, MULTIPLE)

if LOW_RAM_MODE:
    PRECACHE_OFFLOAD = "sequential"
    TEXT_ENCODER_4BIT = True
    MAX_SEQ_LEN = min(MAX_SEQ_LEN, 512)

    # Limitar threads puede reducir picos secundarios en CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    print()
    print("=" * 80)
    print("[LOW-RAM] Modo conservador activado")
    print(f"[LOW-RAM] RAM detectada: {SYSTEM_RAM_GB:.1f} GB")
    print(f"[LOW-RAM] Umbral: {LOW_RAM_THRESHOLD_GB:.1f} GB")
    print("[LOW-RAM] Se forzará sequential offload + límite de CPU + disk offload.")
    print("=" * 80)


# ============================================================================
# DESCARGA DESDE HUGGING FACE
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
    local_path = str(local_path or "./LTX23-NF4").strip()

    if not local_path:
        local_path = "./LTX23-NF4"

    has_base = os.path.exists(os.path.join(local_path, "model_index.json"))
    has_nf4 = os.path.exists(os.path.join(local_path, "index.json"))

    if has_base and has_nf4:
        print(f"[OK] Modelo local encontrado en / Local model found at: {local_path}")
        if not os.path.exists(os.path.join(local_path, "text_encoder_NF4", "config.json")):
            # Instalaciones anteriores traían el text encoder en FP32 en vez del NF4 ya cuantizado.
            from huggingface_hub import snapshot_download
            print("Downloading / Descargando: {}/text_encoder_NF4 (~8 GB)".format(HF_NF4_REPO_ID))
            snapshot_download(repo_id=HF_NF4_REPO_ID, local_dir=local_path, token=get_hf_token(),
                              max_workers=4, allow_patterns=["text_encoder_NF4/*"])
        return local_path

    print()
    print("=" * 80)
    print("WARNING / ATENCIÓN")
    print("=" * 80)
    print("This will download about 32 GB. This may take several minutes.")
    print("Esto descargará unos 32 GB. Esto puede tardar varios minutos.")
    print("=" * 80)

    auto = os.environ.get("LTX_AUTO_CONFIRM_DOWNLOAD", "0").strip().lower()
    if auto not in ("1", "true", "yes", "y", "on"):
        try:
            input("")
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
    # El Transformer BF16 (~38 GB) y el text encoder FP32 (~49 GB) no se usan: salen ya cuantizados del repo NF4.
    print("Downloading / Descargando:", HF_BASE_REPO_ID)
    snapshot_download(
        repo_id=HF_BASE_REPO_ID,
        local_dir=local_path,
        token=token,
        max_workers=4,
        ignore_patterns=["transformer/*", "text_encoder/*"],
    )

    print()
    print("Downloading / Descargando:", HF_NF4_REPO_ID)
    snapshot_download(
        repo_id=HF_NF4_REPO_ID,
        local_dir=local_path,
        token=token,
        max_workers=4,
    )

    if not os.path.exists(os.path.join(local_path, "model_index.json")):
        raise RuntimeError(
            f"Descarga incompleta: falta model_index.json en {local_path}"
        )

    if not os.path.exists(os.path.join(local_path, "index.json")):
        raise RuntimeError(
            f"Descarga incompleta: falta index.json en {local_path}"
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


def vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def vram_peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def read_audio_channels(model_id, default=128):
    """
    Lee audio_in_channels del config.json del transformer en disco (0 VRAM).
    El del repo NF4 (raíz) es el mismo que el de transformer/, que ya no se descarga.
    """
    for rel in ("config.json", os.path.join("transformer", "config.json")):
        p = os.path.join(model_id, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = json.load(f)
                return int(c.get("audio_in_channels", default))
            except Exception:
                pass

    return default


def read_model_index_components(model_id):
    """
    Lee model_index.json y devuelve los componentes del pipeline.
    """
    path = os.path.join(model_id, "model_index.json")

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out = {}

        if isinstance(data, dict):
            for key, value in data.items():
                if key.startswith("_"):
                    continue

                if isinstance(value, (list, tuple)) and len(value) == 2:
                    out[key] = value

        return out

    except Exception as exc:
        print("[LIGHT] No se pudo leer model_index.json: {}".format(exc))
        return {}


def resolve_component_class(model_id, component_name, current_object=None):
    """
    Resuelve la clase de un componente del pipeline a partir de model_index.json.
    """
    info = read_model_index_components(model_id).get(component_name)

    if info is not None:
        module_name, class_name = info

        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            print(
                "[LIGHT] No se pudo importar {}.{}: {}".format(
                    module_name,
                    class_name,
                    exc
                )
            )

    if current_object is not None:
        return type(current_object)

    return None


def list_text_encoder_components(pipe):
    """
    Detecta todos los text encoders presentes en el pipeline.
    """
    names = []

    # Desde model_index.json
    for key in read_model_index_components(MODEL_ID).keys():
        if key.lower().startswith("text_encoder") and key not in names:
            names.append(key)

    # Por si existe alguna instancia directa en el pipeline
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if hasattr(pipe, attr) and attr not in names:
            names.append(attr)

    return names


def unload_pipeline_component(pipe, name):
    """
    Libera de RAM un componente del pipeline antes de recargarlo.
    """
    obj = getattr(pipe, name, None)

    if obj is not None:
        try:
            setattr(pipe, name, None)
        except Exception:
            pass

        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cpu_ram_limit_gb():
    """
    Límite de RAM que dejamos usar a Accelerate para offload.
    En modo baja RAM hay que ser mucho más conservador.
    """
    total = SYSTEM_RAM_GB or 32.0

    if LOW_RAM_MODE:
        # Para 32 GB o menos: dejar ~35% como máximo, con tope 12 GB.
        # Si tienes 16 GB, esto dejaría ~5-6 GB.
        return max(2.0, min(12.0, total * 0.35))

    # Para máquinas con más RAM.
    return max(8.0, min(24.0, total * 0.45))


def auto_max_memory_for_cuda():
    """
    Limita la memoria máxima usada por accelerate en modo auto.
    En baja RAM también limita agresivamente la CPU para favorecer
    que accelerate/transformers puedan usar offload a disco.
    """
    try:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)

        # Dejamos margen para CUDA, VAE, activaciones y el propio Python.
        gpu_limit_gb = max(1.0, free_gb - 1.5)
    except Exception:
        gpu_limit_gb = 6.0

    cpu_limit_gb = _cpu_ram_limit_gb()

    return {
        0: f"{gpu_limit_gb:.1f}GiB",
        "cpu": f"{cpu_limit_gb:.1f}GiB",
    }


def load_precache_pipeline_light(model_id, skip_text_encoders=True):
    """
    Carga el pipeline en CPU evitando cargar de golpe los text encoders.
    Esto reduce muchísimo el primer pico de RAM.
    """
    overrides = {
        "transformer": None,
    }

    components = read_model_index_components(model_id)

    # No cargar text encoders originales si vamos a cuantizarlos después
    if skip_text_encoders:
        for name in components.keys():
            if name.lower().startswith("text_encoder"):
                overrides[name] = None

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **overrides
        )
        return pipe

    except TypeError as exc:
        print("[LIGHT] Overrides no aceptados por el pipeline ({}).".format(exc))
        print("[LIGHT] Fallback: cargando solo sin transformer.")
        return DiffusionPipeline.from_pretrained(
            model_id,
            transformer=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )


def quantize_one_text_encoder_4bit(pipe, component_name):
    """
    Cuantiza UN text encoder concreto:
    - libera antes el original de RAM
    - intenta cargarlo 4-bit directo a VRAM si no estamos en LOW_RAM
    - si falla, intenta auto con offload
    - en LOW_RAM evita el fallback BF16 completo en CPU como primera opción
    """
    current_object = getattr(pipe, component_name, None)

    component_class = resolve_component_class(
        MODEL_ID,
        component_name,
        current_object
    )

    if component_class is None:
        print("[4bit] No se pudo resolver la clase de {}.".format(component_name))
        return False, None

    # CLAVE: liberar el original antes de cargar el cuantizado
    print("[4bit] Liberando {} de RAM antes de cuantizar...".format(component_name))
    unload_pipeline_component(pipe, component_name)

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        print("[4bit] BitsAndBytesConfig no disponible: {}".format(exc))
        return False, None

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    offload_dir = os.path.join(".", "_offload_tmp_{}".format(component_name))
    os.makedirs(offload_dir, exist_ok=True)

    strategies = []

    # En baja RAM evitamos intentar meter todo directamente en CUDA de golpe.
    # Es preferible device_map auto con límites de CPU/GPU.
    if not LOW_RAM_MODE:
        strategies.append(
            (
                "cuda:0 directo",
                {
                    "device_map": "cuda:0",
                },
            )
        )

    strategies.append(
        (
            "auto 4bit con offload",
            {
                "device_map": "auto",
                "max_memory": auto_max_memory_for_cuda(),
                "offload_folder": offload_dir,
            },
        )
    )

    # text_encoder_NF4/ es este mismo text encoder ya cuantizado con bnb_cfg (salidas idénticas):
    # se carga tal cual, sin leer ni descargar el FP32 de ~49 GB.
    prequantized = component_name + "_NF4"
    if os.path.exists(os.path.join(MODEL_ID, prequantized, "config.json")):
        base_kwargs = {"subfolder": prequantized}
    else:
        base_kwargs = {"subfolder": component_name, "quantization_config": bnb_cfg}

    for label, extra_kwargs in strategies:
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            **base_kwargs,
        }
        kwargs.update(extra_kwargs)

        try:
            print("[4bit] {} -> intentando {}...".format(component_name, label))

            new_object = component_class.from_pretrained(
                MODEL_ID,
                **kwargs
            )

            setattr(pipe, component_name, new_object)

            print("[4bit] {} cuantizado OK con {}.".format(component_name, label))
            print("[4bit] VRAM actual: {:.2f} GB".format(vram_gb()))

            # No borramos la carpeta si la estrategia usó offload,
            # porque podría contener datos necesarios durante la ejecución.
            if os.path.exists(offload_dir) and "offload" not in label.lower():
                try:
                    shutil.rmtree(offload_dir, ignore_errors=True)
                except Exception:
                    pass

            return True, label

        except Exception as exc:
            print("[4bit] {} fallo con {}: {}".format(component_name, label, exc))
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------------
    # En baja RAM NO cargamos directamente el text encoder BF16 completo en CPU,
    # porque eso puede pedir decenas de GB y causar OOM.
    #
    # En su lugar intentamos device_map="auto" con límite de CPU y offload a disco.
    # Será más lento, pero evita el pico de RAM.
    # ------------------------------------------------------------------------
    if LOW_RAM_MODE:
        try:
            print(
                "[4bit] {} -> LOW_RAM fallback: bf16 con device_map auto + offload a disco...".format(
                    component_name
                )
            )

            fallback_object = component_class.from_pretrained(
                MODEL_ID,
                subfolder=component_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=auto_max_memory_for_cuda(),
                offload_folder=offload_dir,
            )

            setattr(pipe, component_name, fallback_object)

            print(
                "[4bit] {} cargado en modo bf16 auto/disk offload (lento pero bajo en RAM).".format(
                    component_name
                )
            )

            return True, "bf16 auto/disk offload"

        except Exception as exc:
            print(
                "[4bit] LOW_RAM fallback falló para {}: {}".format(
                    component_name,
                    exc
                )
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Último recurso opcional. Puede volver a causar pico alto de RAM,
            # pero si el sistema tiene pagefile/swap suficiente podría funcionar.
            if LOW_RAM_ALLOW_CPU_FALLBACK:
                try:
                    print(
                        "[4bit] {} -> último recurso LOW_RAM: bf16 completo en CPU...".format(
                            component_name
                        )
                    )

                    fallback_object = component_class.from_pretrained(
                        MODEL_ID,
                        subfolder=component_name,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )

                    setattr(pipe, component_name, fallback_object)

                    print(
                        "[4bit] {} cargado en CPU como último recurso.".format(
                            component_name
                        )
                    )

                    return True, "bf16 CPU fallback"

                except Exception as exc2:
                    print(
                        "[4bit] CRITICO: último recurso CPU falló para {}: {}".format(
                            component_name,
                            exc2
                        )
                    )

            return False, None

    # En máquinas con suficiente RAM sí podemos mantener el fallback original.
    try:
        print(
            "[4bit] {} -> recargando original bf16 en CPU como fallback...".format(
                component_name
            )
        )

        fallback_object = component_class.from_pretrained(
            MODEL_ID,
            subfolder=component_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        setattr(pipe, component_name, fallback_object)

    except Exception as exc:
        print(
            "[4bit] CRITICO: no se pudo restaurar {}: {}".format(
                component_name,
                exc
            )
        )

    return False, None


def try_quantize_text_encoders_4bit(pipe):
    """
    Cuantiza TODOS los text encoders detectados, uno detrás de otro.
    Devuelve:
    - True/False si todos se cuantizaron correctamente
    - text_device recomendado
    """
    names = list_text_encoder_components(pipe)

    if not names:
        print("[4bit] No se detectaron text encoders; se omite.")
        return False, "cpu"

    print("[4bit] Text encoders detectados: {}".format(", ".join(names)))

    all_ok = True

    for name in names:
        ok, label = quantize_one_text_encoder_4bit(pipe, name)

        if not ok:
            all_ok = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_ok:
        print("[4bit] Algun text encoder no se pudo cuantizar; se usara fallback/offload.")
        return False, "cpu"

    return True, "cuda"


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

        # Blindar todos los text encoders y connectors contra .to("cuda") internos.
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors"):
            _patch_module_to_noop_device(getattr(pipe, attr, None))

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
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype)
        }

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
            safe_path = path.replace(".", "_").replace("/", "_").replace("\\", "_")
            filename = f"{prefix}_{safe_path}.pt"

            torch.save(obj.detach().cpu(), os.path.join(CACHE_DIR, filename))

            return {
                "type": "tensor",
                "file": filename,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }

        if isinstance(obj, dict):
            return {
                "type": "dict",
                "items": {
                    str(k): recurse(v, f"{path}_{k}")
                    for k, v in obj.items()
                }
            }

        if isinstance(obj, (tuple, list)):
            return {
                "type": "tuple" if isinstance(obj, tuple) else "list",
                "items": [
                    recurse(v, f"{path}_{i}")
                    for i, v in enumerate(obj)
                ]
            }

        return {
            "type": "value",
            "value": json_safe(obj)
        }

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
            return

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
    # Sin este paso, el transformer (preentrenado sobre latentes YA
    # normalizados) recibe un "clean" con escala/varianza equivocada.
    latents_mean = getattr(vae, "latents_mean", None)
    latents_std = getattr(vae, "latents_std", None)

    if latents_mean is not None and latents_std is not None:
        latents_mean = latents_mean.to(
            device=latent.device,
            dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)

        latents_std = latents_std.to(
            device=latent.device,
            dtype=latent.dtype
        ).view(1, -1, 1, 1, 1)

        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

        latent = (latent - latents_mean) * scaling_factor / latents_std

    return latent.detach().to(torch.bfloat16).cpu().contiguous()


def make_audio_latent(video_latent, audio_channels):
    return torch.zeros(
        (video_latent.shape[0], audio_channels, 1),
        dtype=torch.bfloat16
    )


# ============================================================================
# MAIN
# ============================================================================
def preprocess_ltx23():
    global MODEL_ID

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    # ------------------------------------------------------------------
    # Descargar modelo antes de buscar la carpeta local
    # ------------------------------------------------------------------
    MODEL_ID = ensure_ltx23_model_downloaded(MODEL_ID)

    if not os.path.isdir(MODEL_ID):
        raise FileNotFoundError(f"No existe el modelo: {MODEL_ID}")

    if not os.path.exists(os.path.join(MODEL_ID, "model_index.json")):
        raise FileNotFoundError(
            f"Falta model_index.json en: {MODEL_ID}. "
            "La descarga del modelo base puede haber quedado incompleta."
        )

    if not os.path.exists(os.path.join(MODEL_ID, "index.json")):
        raise FileNotFoundError(
            f"Falta index.json en: {MODEL_ID}. "
            "La descarga del modelo NF4 puede haber quedado incompleta."
        )

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
    print("Low RAM mode :", LOW_RAM_MODE)

    if SYSTEM_RAM_GB is not None:
        print("System RAM   : {:.1f} GB".format(SYSTEM_RAM_GB))
    else:
        print("System RAM   : desconocida")

    print("=" * 80)

    # Canales de audio desde disco (sin cargar el transformer).
    audio_channels = read_audio_channels(MODEL_ID, 128)
    print("Audio latent channels:", audio_channels)

    # ------------------------------------------------------------------
    # Cargar pipeline en CPU (SIN .to("cuda")).
    # ------------------------------------------------------------------
    print()

    if TEXT_ENCODER_4BIT:
        print("Cargando LTX-2.3 en CPU (sin transformer y SIN text encoders)...")
    else:
        print("Cargando LTX-2.3 en CPU (sin transformer)...")

    pipe = load_precache_pipeline_light(
        MODEL_ID,
        skip_text_encoders=TEXT_ENCODER_4BIT,
    )

    print("Pipeline:", type(pipe).__name__)
    print("VAE:", type(getattr(pipe, "vae", None)).__name__)

    for te_name in list_text_encoder_components(pipe):
        te_obj = getattr(pipe, te_name, None)

        if te_obj is None:
            print("{}: None (se cargara cuantizado)".format(te_name))
        else:
            print("{}: {}".format(te_name, type(te_obj).__name__))

    print("Audio VAE:", type(getattr(pipe, "audio_vae", None)).__name__)

    # ------------------------------------------------------------------
    # Opcional: cuantizar TODOS los text encoders a 4-bit.
    # ------------------------------------------------------------------
    used_4bit = False
    text_device = "cpu"

    if TEXT_ENCODER_4BIT:
        used_4bit, text_device = try_quantize_text_encoders_4bit(pipe)

    # ------------------------------------------------------------------
    # Offload.
    # ------------------------------------------------------------------
    if used_4bit:
        vae = getattr(pipe, "vae", None)

        if vae is not None:
            vae.to("cuda")

        # Evitar que el pipeline intente moverlos de golpe a CUDA más adelante
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors"):
            _patch_module_to_noop_device(getattr(pipe, attr, None))

        text_device = "cuda"

        print("[OFFLOAD] 4bit activo -> text encoder(s) 4-bit + VAE en VRAM.")
        print("[VRAM] tras 4bit + VAE: {:.2f} GB".format(vram_gb()))

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
        "low_ram_mode": LOW_RAM_MODE,
        "low_ram_threshold_gb": LOW_RAM_THRESHOLD_GB,
        "system_ram_gb": SYSTEM_RAM_GB,
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