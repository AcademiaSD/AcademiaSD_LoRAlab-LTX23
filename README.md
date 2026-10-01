> [!IMPORTANT]
> ## ➡️ This project continues in [AcademiaSD LoRAlab Trainer Studio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio)
> All the AcademiaSD LoRAlab trainers — **Qwen-Image 2.1, Krea 2, LTX-2.3 and MiniMax-H3** — in **one installation** with one launcher, smaller model downloads and new trainers as they come out. **This repository is no longer updated.**
> Moving over? Copy this trainer's model folder into Trainer Studio and it will not be downloaded again.
>
> ## ➡️ Este proyecto continúa en [AcademiaSD LoRAlab Trainer Studio](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio)
> Todos los entrenadores LoRAlab de AcademiaSD — **Qwen-Image 2.1, Krea 2, LTX-2.3 y MiniMax-H3** — en **una sola instalación** con un lanzador común, descargas de modelos más pequeñas y los nuevos entrenadores según vayan saliendo. **Este repositorio ya no se actualiza.**
> ¿Te cambias? Copia la carpeta del modelo de este entrenador dentro de Trainer Studio y no tendrás que volver a descargarlo.

# AcademiaSD LoRAlab-LTX-2.3 Beta v0.93

![AcademiaSD_LoRAlab-LTX23](assets/portada.jpg)

<p align="center">
  <b>An ultra-fast, low-resource Web GUI & pipeline for training LTX 2.3 LoRAs.</b>
</p>

> **LTX-2.5:** this trainer is also compatible with LTX-2.5. / Este entrenador también es compatible con LTX-2.5.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/UI-Flask%20%2B%20HTML5-purple.svg" alt="Web UI">
</p>

---
## YouTube video instructions.
https://www.youtube.com/watch?v=52Kyr7GZ6Y0
---

## 🔬 Technical Deep-Dive: Why is it so fast, light, & high quality?

Training a 22-Billion parameter Diffusion Transformer (DiT) like **LTX-2.3** typically requires enterprise-grade hardware (30GB+ VRAM, 64GB+ System RAM) and hours of compute. **AcademiaSD LoRAlab LTX 2.3** breaks this barrier, allowing training on consumer GPUs (12GB-16GB VRAM) in record time while preserving 100% of the model's generation quality. 

Here is the exact architectural breakdown of how this is achieved:

---

### 1. 📉 VRAM Usage is Reduced to ~12 GB

* **4-Bit NormalFloat (NF4) Quantization (`bitsandbytes`)**: The 22-Billion parameter DiT backbone is quantized into 4-bit NF4 weights (`Linear4bit`). Base weights are frozen and pinned to CUDA memory, compressing the 22B model footprint from ~35 GB down to ~9 GB.
* **Zero VRAM Wasted on Encoders (Offline Pre-Caching)**: During training (`2_train_lora_ltx23.py`), **neither the Text Encoder nor the VAE are loaded into VRAM**. All text embeddings and image latents are pre-computed once during the pre-cache stage and stored on disk.
* **8-Bit AdamW Optimizer (`bitsandbytes.optim.AdamW8bit`)**: Optimizer states are stored in 8-bit precision instead of 32-bit float, cutting optimizer VRAM overhead by 75%.
* **Gradient Checkpointing**: Intermediate activation tensors are recomputed during backward passes rather than stored in memory, keeping activation VRAM flat regardless of resolution.

---

### 2. ⚡ Why Training Speed is So Fast

* **No Per-Step Encoding Overhead**: In standard pipelines, every training step spends compute cycles passing images through the VAE and prompts through the LLM/Text Encoder. By eliminating this in the pre-cache phase, **100% of GPU compute during training is dedicated purely to the DiT backward pass**.
* **2x2 Latent Patch Packing**: Image latents `[B, C, H, W]` are packed into 2x2 spatial patches (`(H//2)*(W//2)` sequence length). Patching reduces the DiT self-attention sequence length by **4x**, accelerating attention matrix calculations exponentially.
* **Pinned RAM & Non-Blocking CUDA Transfers**: Pre-cached `.pt` latents and embeddings are cached into pinned host memory (`pin_memory()`) and transferred asynchronously to GPU memory (`non_blocking=True`), completely bypassing CPU-to-GPU data bottlenecks.

---

### 3. 🎨 Why Generation Quality is 100% Preserved

* **Exact Channel-Wise VAE Normalization**: VAE uses specific channel-wise mean and standard deviation tensors (`latents_mean`, `latents_std`). Our pre-caching applies exact channel normalization `(z - mean) / std`, ensuring latent distributions match LTX-2.3s pre-trained space down to the float.
* **Full-Layer Target Coverage**: LoRA adapters target **all** `Linear` and `Linear4bit` modules across the DiT architecture, enabling deep feature learning (concepts, styles, faces, lighting) rather than surface-level overfitting.
* **Native LTX 2.3 Noise Shift Schedule**: Implements LTX 2.3's exact mathematical noise shift function (`calculate_shift`) and logit-normal/shifted timestep sampling (`sample_sigma`), preserving the true velocity-matching diffusion trajectories.

---

## ✨ Features

- **🌐 Modern Web GUI**: Control pre-caching, dataset editing, training, checkpointing, and model export from a sleek single-page web app powered by Flask.
- **🚀 1-Click Auto Launch**: Double-click `Run_LoRAlab-LTX23.bat` to automatically launch the server and open `http://127.0.0.1:5000` in your default browser.
- **📊 Real-Time Hardware Telemetry**:
  - System **RAM** usage.
  - Physical **GPU VRAM** usage (via `nvidia-smi` / `torch`).
  - **GPU Temperature (°C)** with dynamic color coding (Green <70°C, Orange 70–79°C, Red >80°C).
- **🔑 Hugging Face Token Support**: Optional HF token management (`HF_token.json`) for faster model downloads with live progress bars (MBs, transfer speed, ETA).
- **✍️ Auto-Captions**: **Create Captions** writes one `.txt` per image with Qwen3-VL-8B (NF4, ~5.5 GB VRAM), the same captioner as the Qwen-Image 2.1 LoRAlab, with the trigger word first. The first time it downloads ~5 GB to `Captioner-Qwen3-VL-8B/`. The prompt is editable and **Overwrite** off only fills the missing captions.
- **🖼️ Dataset Manager & Inline Caption Editor**:
  - Visual grid with status badges (🟢 **Green** = Caption present, 🔴 **Red** = Missing caption), resizable by dragging.
  - Modal lightbox to view high-res images and **edit `.txt` captions directly on disk** (warns before closing with unsaved changes).
  - Batch tool: put the **Trigger Word** first and **Append / Replace / Remove** a common text in every caption. **Clear Captions** and a per-image delete button.
- **🔄 Live Settings**: while training, **Save JSON** applies steps, save/preview every, preview settings and LR on the next step.
- **🗑️ Delete Pre-Cache / Delete Training** buttons to empty the current project's folders.
- **⏱️ Exact Step Resume Checkpoints**: Interrupt or stop training at any step (e.g. Step 333); the exact state (`current_step.txt`, `optimizer.pt`, `adapter_model.safetensors`) is saved automatically. Click **Start/Resume** to continue from that exact step. Resuming always keeps the checkpoint's rank and alpha.
- **🏷️ LoRA Metadata**: exported LoRAs carry kohya-style metadata (trigger word, rank, steps, resolution) that CivitAI and LoRA managers read.
- **📂 Automatic Project Folder Management**: Dynamically routes cache and outputs to `./cached_data_LTX23_<project>` and `./LTX23_lora_output_<project>` based on your project name.
- **🚀 One-Click WebUI Export ("Send to Models")**: Export the best `.safetensors` LoRA directly to your preferred WebUI folder (ComfyUI, ForgeNeo, etc.).
- **🌐 Fully Bilingual (English / Español)**: All buttons, console logs, dialogs, and progress bars display labels in both English and Spanish.

---

## 🖥️ System Requirements

| Requirement | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11 |
| **GPU** | NVIDIA GPU with **12 GB VRAM** | NVIDIA GPU with **12 GB–24 GB VRAM** |
| **Python** | Python 3.10+ (inside `venv`) | Python 3.10 / 3.11 |
| **CUDA Toolkit** | CUDA 11.8 or 12.1+ | CUDA 12.1+ |

---

## 📦 Installation

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/AcademiaSD/AcademiaSD-LoRAlab-LTX23.git
   cd AcademiaSD-LoRAlab-LTX23
   ```

2. **Install virtual environment & dependencies**:
   Double-click `Install_LoRAlab-venv.bat` to automatically set up the Python virtual environment (`venv`) and install all required core libraries.

3. **(Optional) Install Triton & SageAttention 2.2**:
   Double-click `Install_Triton&SageAtten220.bat` to install Triton and SageAttention 2.2 for enhanced attention speedup and memory optimization.

4. **Update the application**:
   You can check for and apply updates at any time by running `Update_LoRAlab-LTX23.bat`.

> **Model download (~32 GB):** the transformer and the Gemma 3 text encoder come already quantized from the NF4 repo `AcademiaSD/LTX23_NF4`; the BF16 `transformer/` (~38 GB) and the FP32 `text_encoder/` (~49 GB) of `diffusers/LTX-2.3-Diffusers` are no longer downloaded. The embeddings are identical to the ones before. **Updating:** the first Pre-Cache downloads `LTX23-NF4/text_encoder_NF4` (~8 GB) and the first training `LTX23-NF4/others.safetensors` (13 MB); then you can delete the old folders `LTX23-NF4/transformer` and `LTX23-NF4/text_encoder`.

---

## ⚡ Usage Guide

### 1. Launch the Application
Simply double-click the launcher:
```cmd
Run_LoRAlab-LTX23.bat
```
The server will start, and your web browser will automatically open `http://127.0.0.1:5000`.

Double-click the Updater LoRAlab-LTX2.3.
```cmd
Update_LoRAlab-LTX23.bat
```

### 2. Captions (optional)
1. Enter a **Project Name** (e.g., `cherry2`) and a **Trigger Word**.
2. Select your image folder using the native Windows file requester (**Browse / Explorar**).
3. In the Dataset Manager, click **Create Captions / Crear Captions** and review them before pre-caching.

### 3. Pre-Cache Dataset
1. Set your target resolution (e.g., `768x768`) and **Multiple** (`8`, `16`, `32`, or `64`).
2. Click **Start Pre-Cache / Iniciar Pre-Caché**.

### 4. Train LoRA
1. Configure **Total Steps** (e.g., `1200`), **Learning Rate** (e.g., `0.0001`), **LoRA Rank/Alpha**, and **Save Every**.
2. Click **Start / Resume**.
3. You can stop training at any time by clicking **Stop Training**; exact step state will be saved automatically for seamless resuming.

### 5. Export to ComfyUI / WebUI
1. Enter your preferred **Final LoRA Filename** (e.g., `my_character.safetensors`).
2. Select your ComfyUI / Forge / A1111 `models/loras` directory using **Browse / Explorar**.
3. Click **🚀 Send to Models**.

---

## 📁 Project Structure

```text
AcademiaSD_LoRAlab-LTX23/
├── assets/
│   ├── banner.png             # Web GUI top header banner
│   └── logo.png               # Logo & browser favicon
├── 0_caption_LTX23.py          # Dataset auto-captioning (Qwen3-VL-8B)
├── 1_pre_cache_LTX23.py        # Latent VAE & Text Embedding pre-caching script
├── 2_train_lora_LTX23.py       # DiT 22B NF4 LoRA training script
├── server.py                   # Flask backend web server
├── trainer_ui.html             # HTML5 / CSS3 / JS Web GUI
├── Run_LoRAlab-LTX23.bat       # Windows 1-click launcher
├── Update_LoRAlab-LTX23.bat    # Updater
├── pre_cache_settings.json     # Active pre-cache configuration
├── train_settings.json         # Active training configuration
├── caption_settings.json       # Auto-caption prompt and options
└── HF_token.json               # Optional Hugging Face access token
```

---

## 💬 Community & Support

Join the **AcademiaSD** community to learn more about local image and video AI.!

- ▶ **YouTube**: [youtube.com/@Academia_SD](https://www.youtube.com/@Academia_SD)
- 𝕏 **X (Twitter)**: [twitter.com/Academia_S_D](https://twitter.com/Academia_S_D)
- 💬 **Discord**: [discord.gg/Syuaduy678](https://discord.gg/Syuaduy678)
- ☕ **Ko-Fi**: [ko-fi.com/academiasd](https://ko-fi.com/academiasd)

---

## 📜 Credits & License

Developed with ❤️ by **AcademiaSD**. Built upon PyTorch, Diffusers, PEFT, Bitsandbytes, and Hugging Face Hub.




