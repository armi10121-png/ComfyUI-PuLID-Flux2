# ComfyUI-PuLID-Flux2

[![GitHub stars](https://img.shields.io/github/stars/iFayens/ComfyUI-PuLID-Flux2?style=social)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Flux.2](https://img.shields.io/badge/Flux.2-Klein%20%26%20Dev-green)](https://huggingface.co/black-forest-labs)
[![Version](https://img.shields.io/badge/version-0.2.0-orange)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)

> **First PuLID implementation for FLUX.2 — supports Klein (4B/9B) and Dev (32B)**  
> Auto model detection — just plug and play  
> March 2026

---

## ⚠️ Current State & Honest Assessment

| | Status | Notes |
|---|---|---|
| Node loads and runs | ✅ | All 5 nodes functional |
| InsightFace face detection | ✅ | AntelopeV2 + buffalo_l fallback |
| EVA-CLIP encoding | ✅ | via open_clip |
| Auto model detection | ✅ | Klein 4B / 9B / Dev auto-detected |
| Flux.2 Klein support | ✅ | Best results |
| Flux.2 Dev support | ✅ | Works, slightly lower quality than Klein |
| Green image artifact | ✅ Fixed | Auto-cleanup between runs |
| Native Klein/Dev weights | ❌ Not yet | Training script included |

**Best results:** Use PuLID at low weight (0.2-0.3) combined with Klein's native Reference Conditioning.

---

## 🎯 What is this?

This custom node brings **PuLID (Pure Identity)** face consistency to **FLUX.2**, supporting both Klein and Dev variants with automatic model detection.

Previous PuLID nodes only support Flux.1 Dev. This project is the **first** to adapt PuLID for the entire FLUX.2 family.

### Supported models

| Model | Double blocks | Single blocks | Hidden dim | Status |
|---|---|---|---|---|
| Flux.1 Dev (lldacing) | 19 | 38 | 4096 | ❌ Not this node |
| **Flux.2 Klein 4B** | 5 | 20 | 3072 | ✅ Best results |
| **Flux.2 Klein 9B** | 8 | 24 | 4096 | ✅ Best results |
| **Flux.2 Dev 32B** | 8 | 48 | 6144 | ✅ Works |

> The node **automatically detects** which model you are using — just select `auto (recommended)`.

---

## 📦 Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git
cd ComfyUI-PuLID-Flux2
```

### Update

```bash
cd ComfyUI/custom_nodes/ComfyUI-PuLID-Flux2
git pull
```

> ⚠️ **If you already have a working ComfyUI**, install only the required packages:
> ```bash
> pip install insightface onnxruntime-gpu open-clip-torch safetensors ml_dtypes==0.3.2
> ```
> Only run `pip install -r requirements.txt` on a **fresh install**.

### 2. EVA-CLIP (automatic)

EVA02-CLIP-L-14-336 downloads automatically on first run via `open_clip` (~800MB).

> ⚠️ **Do NOT** install `eva_clip` from GitHub — that package is broken.

### 3. Download InsightFace AntelopeV2

Download from: https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2

```
ComfyUI/models/insightface/models/antelopev2/
├── 1k3d68.onnx
├── 2d106det.onnx
├── genderage.onnx
├── glintr100.onnx
└── scrfd_10g_bnkps.onnx
```
> ⚠️ Folder must be named exactly `antelopev2` (lowercase).  
> Auto-fallback to `buffalo_l` if not found.

### 4. Download PuLID weights

Download `pulid_flux_v0.9.1.safetensors` from https://huggingface.co/guozinan/PuLID  
Place in: `ComfyUI/models/pulid/`

### 5. Example workflow

Ready-to-use workflow available in the `workflows/` folder — drag & drop into ComfyUI.

---

## 🔌 Available Nodes

| Node | Description |
|---|---|
| `Load InsightFace (PuLID Klein)` | Loads AntelopeV2 (falls back to buffalo_l) |
| `Load EVA-CLIP (PuLID Klein)` | Loads EVA02-CLIP-L-14-336 |
| `Load PuLID Flux.2 Model` | Loads PuLID weights |
| **`Apply PuLID ✦ Flux.2`** | Main node — patches Flux.2 model |
| `PuLID Klein — Face Debug Preview` | Visualizes detected faces |

---

## ⚙️ Recommended Parameters

| Parameter | Klein | Dev |
|---|---|---|
| `model_variant` | `auto (recommended)` | `auto (recommended)` |
| `weight` | `0.5-0.7` | `0.3-0.5` |
| `start_at` | `0.0` | `0.0` |
| `end_at` | `1.0` | `1.0` |

---

## 🏗️ How it works

```
Reference image
    │
    ├──► InsightFace AntelopeV2 → 512-dim face embedding
    │
    └──► EVA02-CLIP-L-14-336 → 768-dim visual features
                   │
                   ▼
          IDFormer (MLP + PerceiverCA)
                   │
                   ▼
          id_tokens [B, 4, dim]
          auto-projected to model dim (3072/4096/6144)
                   │
          Injection into double_blocks
```

---

## 🚀 Training Native Weights

```bash
python training/prepare_dataset.py --output ./dataset --source celeba --max_images 2000

python training/train_pulid_klein.py \
  --dataset ./dataset/filtered \
  --output ./output \
  --comfyui_path C:/AI/ComfyUI \
  --dim 4096 \
  --epochs 20
```

See [training/README_TRAINING.md](training/README_TRAINING.md) for details.

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| `AssertionError: 'detection' not in models` | Install AntelopeV2 (section 3) |
| `EVA-CLIP not available` | `pip install open-clip-torch` |
| `ml_dtypes has no attribute 'float4_e2m1fn'` | `pip install ml_dtypes==0.3.2` |
| Torch downgraded after install | `pip install insightface --no-deps` |
| Green image after changing weight | Update with `git pull` + restart ComfyUI |
| No difference with PuLID | Generate 4-5 images with different seeds and compare |

---

## 📋 Roadmap

- [x] Flux.2 Klein 4B / 9B support
- [x] Flux.2 Dev 32B support
- [x] Auto model detection
- [x] Auto dim projection (3072/4096/6144)
- [x] EVA-CLIP via open_clip
- [x] InsightFace CUDA + buffalo_l fallback
- [x] Green image artifact fix
- [x] Training scripts
- [x] Example workflow
- [ ] **Native Klein/Dev trained weights** ← main priority
- [ ] Edit mode (img2img) support
- [ ] Body consistency (LoRA + PuLID)
- [ ] HuggingFace model release

---

## 🙏 Credits

- **PuLID original**: [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) (Apache 2.0)
- **PuLID Flux.1**: [lldacing/ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll)
- **Flux.2**: [Black Forest Labs](https://blackforestlabs.ai)
- **EVA-CLIP**: [BAAI](https://github.com/baaivision/EVA)
- **Adaptation for Flux.2**: [@iFayens](https://github.com/iFayens) — March 2026

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
