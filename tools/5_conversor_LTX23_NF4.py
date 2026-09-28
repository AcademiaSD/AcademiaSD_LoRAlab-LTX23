# -*- coding: utf-8 -*-
"""
5_conversor_LTX23_NF4_v3.py

Conversor incremental de los pesos Linear del Transformer de LTX-2.3 a NF4.

IMPORTANTE:
La versión v2 intentaba convertir TODAS las Linear del Transformer a
Linear4bit antes de empezar a guardar. En LTX-2.3 esto puede consumir una
cantidad enorme de RAM/VRAM y el proceso puede terminar antes de llegar a
la fase de guardado. Por eso aparecía:

    NF4 -> transformer_blocks.25...

pero la carpeta weights quedaba vacía y no existía conversion_log.json.

Esta v3 NO hace esa conversión global.

En su lugar:

1. Carga LTX-2.3.
2. Localiza cada torch.nn.Linear.
3. Para cada capa:
   - obtiene sus pesos originales;
   - crea UNA Linear4bit temporal;
   - mueve SOLO esa capa a CUDA;
   - fuerza la materialización NF4;
   - extrae weight + QuantState;
   - guarda inmediatamente un .safetensors;
   - verifica el archivo;
   - libera la capa temporal y la VRAM.
4. Actualiza index.json y conversion_log.json después de CADA capa.

Así, aunque el proceso se interrumpa en la capa 300, las 299 anteriores
permanecen guardadas.

Salida:

    ./LTX23-NF4_weights/
        config.json
        scheduler_config.json
        pipeline_info.json
        module_inventory.json
        metadata.json
        index.json
        conversion_log.json
        weights/
            transformer_blocks__0__attn1__to_q.safetensors
            ...

REQUISITOS:

    torch
    diffusers
    bitsandbytes
    safetensors
"""

import os
import gc
import json
import time
import traceback

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import save_file


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL_ID = "./LTX23-Raw"
OUTPUT_DIR = "./LTX23-NF4_weights"

COMPUTE_DTYPE = torch.bfloat16
QUANT_TYPE = "nf4"

# Para LTX-2.3 se convierten todas las Linear encontradas.
# Si posteriormente descubrimos capas que deban permanecer BF16,
# se pueden añadir aquí por nombre.
SKIP_QUANT_PATTERNS = ()

# Continuar desde archivos ya existentes.
RESUME = True

# Comprobar cada archivo creado.
VERIFY_FILES = True


# ============================================================================
# UTILIDADES
# ============================================================================

def free_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def safe_json(value):

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, dict):
        return {
            str(k): safe_json(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            safe_json(v)
            for v in value
        ]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def atomic_json_save(data, path):

    tmp = path + ".tmp"

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            safe_json(data),
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )


def load_json(path, default):

    if not os.path.exists(path):
        return default

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return default


def safe_filename(name):

    return name.replace(
        ".",
        "__",
    )


def should_skip(name):

    return any(
        pattern in name
        for pattern in SKIP_QUANT_PATTERNS
        if pattern
    )


def is_valid_file(path):

    if not os.path.exists(path):
        return False

    if os.path.getsize(path) <= 0:
        return False

    if not VERIFY_FILES:
        return True

    try:

        from safetensors import safe_open

        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as f:

            keys = list(f.keys())

        return "weight" in keys

    except Exception:

        return False


# ============================================================================
# GUARDAR METADATOS DEL MODELO
# ============================================================================

def save_model_metadata(
    pipe,
    transformer,
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # config Transformer
    # ------------------------------------------------------------

    if hasattr(
        transformer,
        "config",
    ):

        try:

            atomic_json_save(

                dict(
                    transformer.config
                ),

                os.path.join(
                    output_dir,
                    "config.json",
                ),

            )

        except Exception:

            pass

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    if hasattr(
        pipe,
        "scheduler",
    ):

        try:

            atomic_json_save(

                dict(
                    pipe.scheduler.config
                ),

                os.path.join(
                    output_dir,
                    "scheduler_config.json",
                ),

            )

        except Exception:

            pass

    # ------------------------------------------------------------
    # Información del pipeline
    # ------------------------------------------------------------

    atomic_json_save(

        {

            "pipeline_class":
                pipe.__class__.__name__,

            "pipeline_module":
                pipe.__class__.__module__,

            "transformer_class":
                transformer.__class__.__name__,

            "transformer_module":
                transformer.__class__.__module__,

            "model_id":
                MODEL_ID,

            "quant_type":
                QUANT_TYPE,

            "compute_dtype":
                "bfloat16",

        },

        os.path.join(
            output_dir,
            "pipeline_info.json",
        ),

    )


# ============================================================================
# INVENTARIO DE LINEAR
# ============================================================================

def get_linear_layers(transformer):

    layers = []

    for name, module in transformer.named_modules():

        if isinstance(
            module,
            torch.nn.Linear,
        ):

            layers.append(
                (
                    name,
                    module,
                )
            )

    return layers


def save_module_inventory(
    transformer,
    output_dir,
):

    inventory = []

    for name, module in transformer.named_modules():

        item = {

            "name":
                name,

            "class":
                module.__class__.__name__,

            "module":
                module.__class__.__module__,

        }

        if isinstance(
            module,
            torch.nn.Linear,
        ):

            item.update({

                "type":
                    "Linear",

                "in_features":
                    module.in_features,

                "out_features":
                    module.out_features,

                "bias":
                    module.bias is not None,

                "skip":
                    should_skip(name),

            })

        inventory.append(
            item
        )

    atomic_json_save(

        inventory,

        os.path.join(
            output_dir,
            "module_inventory.json",
        ),

    )


# ============================================================================
# CUANTIZACIÓN DE UNA SOLA LINEAR
# ============================================================================

def quantize_single_linear(
    name,
    source_layer,
    weights_dir,
):

    from bitsandbytes.nn import (
        Linear4bit,
        Params4bit,
    )

    # ------------------------------------------------------------
    # Copiar los pesos FUERA de la capa original.
    # ------------------------------------------------------------

    weight = (

        source_layer.weight.detach()

        .float()

        .cpu()

        .contiguous()

    )

    if source_layer.bias is not None:

        bias = (

            source_layer.bias.detach()

            .to(
                COMPUTE_DTYPE
            )

            .cpu()

            .contiguous()

        )

    else:

        bias = None

    in_features = source_layer.in_features
    out_features = source_layer.out_features

    # ------------------------------------------------------------
    # Crear UNA única Linear4bit temporal.
    # ------------------------------------------------------------

    layer = Linear4bit(

        in_features,

        out_features,

        bias=(
            bias is not None
        ),

        compute_dtype=COMPUTE_DTYPE,

        compress_statistics=True,

        quant_type=QUANT_TYPE,

    )

    # ------------------------------------------------------------
    # Asignar el peso FP32.
    # ------------------------------------------------------------

    layer.weight = Params4bit(

        weight,

        requires_grad=False,

        quant_type=QUANT_TYPE,

        compress_statistics=True,

    )

    if bias is not None:

        layer.bias = torch.nn.Parameter(

            bias,

            requires_grad=False,

        )

    del weight

    # ------------------------------------------------------------
    # AQUÍ se materializa NF4.
    #
    # MUY IMPORTANTE:
    # Solo esta capa viaja a CUDA.
    # ------------------------------------------------------------

    layer = layer.to(
        "cuda"
    )

    # Forzar acceso al peso cuantizado.
    # bitsandbytes materializa quant_state durante .to("cuda").
    qweight = layer.weight

    if not getattr(
        qweight,
        "bnb_quantized",
        False,
    ):

        raise RuntimeError(

            f"{name}: "
            "bnb_quantized=False después de .to(cuda)."

        )

    quant_state = getattr(

        qweight,

        "quant_state",

        None,

    )

    if quant_state is None:

        raise RuntimeError(

            f"{name}: "
            "quant_state=None después de .to(cuda)."

        )

    # ------------------------------------------------------------
    # Serializar QuantState.
    # ------------------------------------------------------------

    qs_dict = quant_state.as_dict(
        packed=True
    )

    tensors = {

        "weight":

            qweight.detach()

            .cpu()

            .contiguous()

    }

    if layer.bias is not None:

        tensors["bias"] = (

            layer.bias.detach()

            .cpu()

            .contiguous()

        )

    metadata = {

        "format":
            "LTX23-NF4",

        "version":
            "3",

        "layer_name":
            name,

        "type":
            "Linear4bit",

        "quant_type":
            "nf4",

        "compute_dtype":
            "bfloat16",

        "bnb_quantized":
            "true",

        "in_features":
            str(
                in_features
            ),

        "out_features":
            str(
                out_features
            ),

        "bias":
            str(
                bias is not None
            ),

    }

    for key, value in qs_dict.items():

        if torch.is_tensor(
            value
        ):

            tensors[

                "quant_state."

                + key

            ] = (

                value.detach()

                .cpu()

                .contiguous()

            )

        else:

            metadata[

                "quant_state_"

                + key

            ] = json.dumps(

                safe_json(
                    value
                )

            )

    filename = (

        safe_filename(name)

        + ".safetensors"

    )

    filepath = os.path.join(

        weights_dir,

        filename,

    )

    # ------------------------------------------------------------
    # GUARDADO INMEDIATO.
    # ------------------------------------------------------------

    save_file(

        tensors,

        filepath,

        metadata=metadata,

    )

    # ------------------------------------------------------------
    # Verificación inmediata.
    # ------------------------------------------------------------

    if not is_valid_file(
        filepath
    ):

        raise RuntimeError(

            f"{name}: "
            f"archivo inválido: {filepath}"

        )

    file_size = os.path.getsize(
        filepath
    )

    # ------------------------------------------------------------
    # LIBERAR ESTA ÚNICA CAPA.
    # ------------------------------------------------------------

    del tensors
    del metadata
    del qs_dict
    del quant_state
    del qweight
    del layer

    free_cuda()

    return {

        "file":
            filename,

        "file_size":
            file_size,

        "in_features":
            in_features,

        "out_features":
            out_features,

        "bias":
            bias is not None,

        "quant_type":
            "nf4",

        "compute_dtype":
            "bfloat16",

        "status":
            "saved",

    }


# ============================================================================
# GUARDAR BF16
# ============================================================================

def save_single_bf16(
    name,
    source_layer,
    weights_dir,
):

    filename = (

        safe_filename(name)

        + ".safetensors"

    )

    filepath = os.path.join(

        weights_dir,

        filename,

    )

    tensors = {

        "weight":

            source_layer.weight.detach()

            .to(
                torch.bfloat16
            )

            .cpu()

            .contiguous()

    }

    if source_layer.bias is not None:

        tensors["bias"] = (

            source_layer.bias.detach()

            .to(
                torch.bfloat16
            )

            .cpu()

            .contiguous()

        )

    save_file(

        tensors,

        filepath,

        metadata={

            "format":
                "LTX23-NF4",

            "version":
                "3",

            "layer_name":
                name,

            "type":
                "Linear",

            "quantized":
                "false",

            "dtype":
                "bfloat16",

        },

    )

    if not is_valid_file(
        filepath
    ):

        raise RuntimeError(

            f"{name}: "
            "archivo BF16 inválido."

        )

    size = os.path.getsize(
        filepath
    )

    del tensors

    gc.collect()

    return {

        "file":
            filename,

        "file_size":
            size,

        "in_features":
            source_layer.in_features,

        "out_features":
            source_layer.out_features,

        "bias":
            source_layer.bias is not None,

        "dtype":
            "bfloat16",

        "status":
            "saved_bf16",

    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    start = time.time()

    print()
    print("=" * 80)
    print(" LTX-2.3 NF4 CONVERTER v3")
    print(" CONVERSIÓN Y GUARDADO INCREMENTAL POR CAPA")
    print("=" * 80)

    print()
    print(
        "Modelo:",
        os.path.abspath(
            MODEL_ID
        ),
    )

    print(
        "Salida:",
        os.path.abspath(
            OUTPUT_DIR
        ),
    )

    # ------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------

    if not os.path.isdir(
        MODEL_ID
    ):

        raise FileNotFoundError(

            f"No existe MODEL_ID: "
            f"{os.path.abspath(MODEL_ID)}"

        )

    if not torch.cuda.is_available():

        raise RuntimeError(

            "CUDA no está disponible."

        )

    print()
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    # ------------------------------------------------------------
    # Crear carpetas ANTES de cargar el modelo.
    # ------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    weights_dir = os.path.join(
        OUTPUT_DIR,
        "weights",
    )

    os.makedirs(
        weights_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Crear conversion_log INMEDIATAMENTE.
    #
    # Incluso si el proceso muere al cargar el modelo, existirá.
    # ------------------------------------------------------------

    log_path = os.path.join(
        OUTPUT_DIR,
        "conversion_log.json",
    )

    index_path = os.path.join(
        OUTPUT_DIR,
        "index.json",
    )

    log = load_json(

        log_path,

        {

            "status":
                "starting",

            "model_id":
                MODEL_ID,

            "layers":
                {},

        },

    )

    index = load_json(

        index_path,

        {

            "format":
                "LTX23-NF4",

            "version":
                3,

            "quantized":
                {},

            "unquantized":
                {},

        },

    )

    atomic_json_save(
        log,
        log_path,
    )

    atomic_json_save(
        index,
        index_path,
    )

    print()
    print(
        "conversion_log.json creado."
    )

    print(
        "index.json creado."
    )

    # ------------------------------------------------------------
    # Cargar modelo
    # ------------------------------------------------------------

    print()
    print(
        "Cargando LTX-2.3..."
    )

    pipe = DiffusionPipeline.from_pretrained(

        MODEL_ID,

        torch_dtype=COMPUTE_DTYPE,

    )

    if not hasattr(
        pipe,
        "transformer",
    ):

        raise RuntimeError(

            "El pipeline no contiene "
            "pipe.transformer."

        )

    transformer = pipe.transformer

    print()
    print(
        "Transformer:",
        transformer.__class__.__name__,
    )

    # ------------------------------------------------------------
    # Guardar metadatos
    # ------------------------------------------------------------

    save_model_metadata(

        pipe,

        transformer,

        OUTPUT_DIR,

    )

    save_module_inventory(

        transformer,

        OUTPUT_DIR,

    )

    # ------------------------------------------------------------
    # Obtener referencias a todas las Linear.
    #
    # No modificamos el Transformer.
    # ------------------------------------------------------------

    layers = get_linear_layers(
        transformer
    )

    print()
    print(
        "Linear encontradas:",
        len(layers),
    )

    # ------------------------------------------------------------
    # Actualizar log
    # ------------------------------------------------------------

    log[
        "status"
    ] = "converting"

    log[
        "total_layers"
    ] = len(
        layers
    )

    atomic_json_save(
        log,
        log_path,
    )

    # ------------------------------------------------------------
    # CONVERSIÓN INCREMENTAL
    # ------------------------------------------------------------

    for position, (
        name,
        source_layer,
    ) in enumerate(
        layers,
        start=1,
    ):

        filename = (

            safe_filename(name)

            + ".safetensors"

        )

        filepath = os.path.join(

            weights_dir,

            filename,

        )

        print()
        print(
            "=" * 80
        )

        print(

            f"[{position}/{len(layers)}] "
            f"{name}"

        )

        # --------------------------------------------------------
        # Resume
        # --------------------------------------------------------

        if (

            RESUME

            and name
            in index[
                "quantized"
            ]

            and is_valid_file(
                filepath
            )

        ):

            print(
                "  YA GUARDADA ->",
                filepath,
            )

            continue

        # --------------------------------------------------------
        # Skip BF16
        # --------------------------------------------------------

        if should_skip(
            name
        ):

            print(
                "  BF16 ->",
                name,
            )

            try:

                entry = save_single_bf16(

                    name,

                    source_layer,

                    weights_dir,

                )

                index[
                    "unquantized"
                ][name] = entry

                log[
                    "layers"
                ][name] = entry

                atomic_json_save(

                    index,

                    index_path,

                )

                atomic_json_save(

                    log,

                    log_path,

                )

                continue

            except Exception as e:

                log[
                    "layers"
                ][name] = {

                    "status":
                        "failed",

                    "error":
                        repr(e),

                    "traceback":
                        traceback.format_exc(),

                }

                atomic_json_save(
                    log,
                    log_path,
                )

                raise

        # --------------------------------------------------------
        # NF4
        # --------------------------------------------------------

        print(
            "  NF4 ->",
            name,
        )

        try:

            entry = quantize_single_linear(

                name,

                source_layer,

                weights_dir,

            )

            # ----------------------------------------------------
            # GUARDAR ÍNDICE INMEDIATAMENTE.
            # ----------------------------------------------------

            index[
                "quantized"
            ][name] = entry

            log[
                "layers"
            ][name] = entry

            log[
                "last_completed"
            ] = name

            atomic_json_save(

                index,

                index_path,

            )

            atomic_json_save(

                log,

                log_path,

            )

            print(
                "  GUARDADO OK:",
                entry[
                    "file"
                ],
            )

            print(
                "  Tamaño:",
                f"{entry['file_size'] / 1024 / 1024:.2f} MB",
            )

            print(
                "  VRAM:",
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            )

        except Exception as e:

            print()
            print(
                "  !!! ERROR !!!"
            )

            print(
                "  Capa:",
                name,
            )

            print(
                "  Error:",
                repr(e),
            )

            log[
                "layers"
            ][name] = {

                "status":
                    "failed",

                "error":
                    repr(e),

                "traceback":
                    traceback.format_exc(),

            }

            log[
                "failed_layer"
            ] = name

            atomic_json_save(

                log,

                log_path,

            )

            print()
            print(
                "El error se ha guardado en:"
            )

            print(
                os.path.abspath(
                    log_path
                )
            )

            raise

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    elapsed = time.time() - start

    log[
        "status"
    ] = "completed"

    log[
        "completed_layers"
    ] = len(
        index[
            "quantized"
        ]
    )

    log[
        "elapsed_seconds"
    ] = elapsed

    atomic_json_save(
        log,
        log_path,
    )

    atomic_json_save(

        {

            "format":
                "LTX23-NF4",

            "version":
                3,

            "model_id":
                MODEL_ID,

            "quant_type":
                "nf4",

            "compute_dtype":
                "bfloat16",

            "total_linear":
                len(layers),

            "quantized":
                len(
                    index[
                        "quantized"
                    ]
                ),

            "unquantized":
                len(
                    index[
                        "unquantized"
                    ]
                ),

            "incremental":
                True,

            "one_layer_at_a_time":
                True,

            "loader_note":
                "QuantState serialized with "
                "quant_state.as_dict(packed=True)",

        },

        os.path.join(

            OUTPUT_DIR,

            "metadata.json",

        ),

    )

    # Resto de parámetros (normas, tablas de modulación...): el trainer ya no carga
    # el Transformer en BF16, así que tienen que ir junto a weights/.
    covered = set(index["quantized"]) | set(index["unquantized"])
    others = {
        k: v.detach().cpu().contiguous()
        for k, v in transformer.state_dict().items()
        if not any(k.startswith(m + ".") for m in covered)
    }
    save_file(others, os.path.join(OUTPUT_DIR, "others.safetensors"))
    print()
    print("others.safetensors:", len(others), "tensores")

    print()
    print("=" * 80)
    print("CONVERSIÓN COMPLETADA")
    print("=" * 80)

    print(
        "Capas NF4:",
        len(
            index[
                "quantized"
            ]
        ),
    )

    print(
        "Capas BF16:",
        len(
            index[
                "unquantized"
            ]
        ),
    )

    print(
        "Tiempo:",
        f"{elapsed / 60:.2f} minutos",
    )

    print()
    print(
        "Pesos guardados en:"
    )

    print(
        os.path.abspath(
            weights_dir
        )
    )

    print("=" * 80)


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Conversión interrumpida."
        )

        print(
            "Los archivos ya guardados permanecen."
        )

        print(
            "Puedes volver a ejecutar el script."
        )

    except Exception:

        print()
        print("=" * 80)
        print(
            "CONVERSIÓN DETENIDA POR ERROR"
        )
        print("=" * 80)

        traceback.print_exc()

        raise
