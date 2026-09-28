from huggingface_hub import snapshot_download
import os

# ==========================================
# CONFIGURACIÓN DE DESTINO
# Pon la ruta de tu disco secundario aquí:
# ==========================================
DIRECTORIO_DESTINO = "./LTX23-Raw"  # la misma carpeta que lee 5_conversor_LTX23_NF4.py

os.makedirs(DIRECTORIO_DESTINO, exist_ok=True)

print("====================================================")
print(f" Iniciando descarga de LTX-2.3")
print(f" Destino: {DIRECTORIO_DESTINO}")
print("====================================================\n")
print("Advertencia: Este modelo es gigante (22 Billones de parámetros).")
print("Descargará decenas de Gigabytes. Ve a tomar un café...\n")

# Hacemos la descarga directa a tu carpeta elegida
snapshot_download(
    repo_id="diffusers/LTX-2.3-Diffusers",
    local_dir=DIRECTORIO_DESTINO,
    local_dir_use_symlinks=False, # ¡VITAL EN WINDOWS! Para que no use C: en secreto
    resume_download=True,         # Si se corta el internet, continuará por donde iba
    ignore_patterns=["*.msgpack", "*.h5", "*.bin"] # Descarga solo los Safetensors modernos para ahorrar espacio
)

print("\n====================================================")
print(" ¡Descarga completada con éxito!")
print("====================================================")