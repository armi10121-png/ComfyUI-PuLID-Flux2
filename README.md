# ComfyUI-PuLID-Flux2

[![GitHub stars](https://img.shields.io/github/stars/iFayens/ComfyUI-PuLID-Flux2?style=social)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Flux.2 Klein](https://img.shields.io/badge/Flux.2-Klein%204B%2F9B-green)](https://huggingface.co/black-forest-labs)

> **First PuLID implementation natively adapted for FLUX.2 Klein (4B & 9B)**  
> Consistent face identity injection without model pollution — March 2026

---

## 🎯 What is this?

This custom node brings **PuLID (Pure Identity)** face consistency to **FLUX.2 Klein**, the latest generation model from Black Forest Labs.

Previous PuLID implementations only support Flux.1 Dev. This project is the **first** to adapt PuLID's architecture specifically for Flux.2 Klein's unique transformer structure.

### Key differences vs existing PuLID nodes

| | PuLID Flux.1 (lldacing) | **ComfyUI-PuLID-Flux2** |
|---|---|---|
| Model | Flux.1 Dev | **Flux.2 Klein 4B / 9B** |
| Double blocks | 19 | 5 (4B) / 8 (9B) |
| Single blocks | 38 | 20 (4B) / 24 (9B) |
| Hidden dim | 4096 | **3072** (4B) / 4096 (9B) |
| Modulation | Per block | **Shared** (Klein-specific) |
| Text encoder | T5 | **Qwen3** |

---

## ✨ Results

| Reference | Generated |
|---|---|
| Input face photo | Consistent identity preserved in new scene |

> Results shown use `pulid_flux_v0.9.1.safetensors` (Flux.1 weights, partial compatibility).  
> Native Klein-trained weights coming soon via the training script included in this repo.

---

## 📦 Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git
cd ComfyUI-PuLID-Flux2
pip install -r requirements.txt
```

### 2. Install EVA-CLIP (downloads automatically on first run)

```bash
python -c "import open_clip; open_clip.create_model_and_transforms('EVA02-L-14-336', pretrained='merged2b_s6b_b61k')"
```

### 3. Download InsightFace AntelopeV2

Download from: https://huggingface.co/MonsterMMORPG/InsightFace_AntelopeV2  
Place in: `ComfyUI/models/insightface/models/antelopev2/`

### 4. Download PuLID weights (Flux.1 compatible, Klein 9B)

```
https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors
```
Place in: `ComfyUI/models/pulid/`

---

## 🔌 Nodes

| Node | Description |
|---|---|
| `Load InsightFace (PuLID Klein)` | Loads AntelopeV2 face detector |
| `Load EVA-CLIP (PuLID Klein)` | Loads EVA02-CLIP-L-14-336 visual encoder |
| `Load PuLID Flux.2 Model` | Loads PuLID weights (.safetensors) |
| **`Apply PuLID ✦ Flux.2`** | Main node — patches Flux.2 Klein model |
| `PuLID Klein — Face Debug Preview` | Visualizes detected faces (debug) |

---

## ⚙️ Recommended Parameters

| Parameter | Value | Notes |
|---|---|---|
| `weight` | `0.7` | Start here, adjust to taste |
| `start_at` | `0.0` | Let PuLID guide from the start |
| `end_at` | `1.0` | Full generation coverage |
| `face_index` | `0` | Use largest detected face |

> ⚠️ Keep `weight` below 0.8 to avoid image quality degradation with non-native weights.

---

## 🏗️ Architecture

```
Reference image
    │
    ├──► InsightFace AntelopeV2
    │         └─► 512-dim face embedding
    │
    └──► EVA02-CLIP-L-14-336
              └─► 768-dim visual features
                       │
                       ▼
              IDFormer (MLP + PerceiverCA)
                       │
                       ▼
              id_tokens [B, 4, dim]
                       │
              ┌────────┴──────────────────────┐
              │  Injection into Flux.2 Klein   │
              │                                │
              │  double_blocks[0,2,4,...]:      │
              │    img += w * PerceiverCA(...)  │
              └────────────────────────────────┘
                       │
                       ▼
              Generated image with consistent identity
```

---

## 🚀 Training Native Klein Weights

This repo includes the **first training script for PuLID on Flux.2 Klein**.

```bash
# Step 1: Prepare dataset
python prepare_dataset.py --output ./dataset --source celeba --max_images 2000

# Step 2: Train
python train_pulid_klein.py \
  --dataset ./dataset/filtered \
  --output ./output \
  --comfyui_path C:/AI/ComfyUI \
  --dim 4096 \
  --epochs 20 \
  --batch_size 4
```

See [README_TRAINING.md](README_TRAINING.md) for full details.

---

## 🛠️ Requirements

- ComfyUI (latest)
- Python 3.10+
- CUDA GPU (12 GB+ VRAM recommended)
- PyTorch 2.4.0+cu121

```
insightface>=0.7.3
onnxruntime-gpu>=1.16.0
open_clip_torch>=3.2.0
safetensors>=0.4.0
opencv-python
numpy
```

---

## 📋 Roadmap

- [x] Custom node for Flux.2 Klein 4B / 9B
- [x] EVA-CLIP integration via open_clip
- [x] InsightFace CUDA support
- [x] Training dataset preparation script
- [x] Training script (Phase 1 — embedding only)
- [ ] Native Klein-trained weights (community training)
- [ ] Training script Phase 2 (full pipeline with Flux)
- [ ] HuggingFace model release
- [ ] Single blocks re-activation after training

---

## 🙏 Credits

- **PuLID original**: [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) (Apache 2.0)
- **PuLID Flux.1**: [lldacing/ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll)
- **Flux.2 Klein**: [Black Forest Labs](https://blackforestlabs.ai)
- **EVA-CLIP**: [BAAI](https://github.com/baaivision/EVA)
- **Adaptation for Flux.2 Klein**: [@iFayens](https://github.com/iFayens) — March 2026

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
