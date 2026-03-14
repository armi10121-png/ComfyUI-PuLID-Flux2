# PuLID Flux.2 Klein — Script d'entraînement

> **Premier script d'entraînement PuLID dédié à Flux.2 Klein.**  
> Développé en Mars 2026.

---

## Vue d'ensemble

Ce script entraîne le module PuLID (IDFormer + PerceiverCA) spécifiquement
pour l'architecture **Flux.2 Klein (4B et 9B)**.

Le modèle Flux.2 Klein est **complètement gelé** — seul PuLID apprend.

### Ce qui est entraîné (~50M paramètres)
```
IDFormer
  ├── proj MLP : InsightFace(512) + EVA-CLIP(768) → tokens [B, 4, dim]
  └── PerceiverCA layers (x4)

PerceiverCA dans double_blocks (x3 pour Klein 9B)
PerceiverCA dans single_blocks (x6 pour Klein 9B)
```

---

## Prérequis

- **GPU** : RTX 3090 24 Go (ou équivalent)
- **Python** : 3.10+
- **ComfyUI** installé avec le custom node `ComfyUI-PuLID-Flux2Klein`

### Installation des dépendances

```bash
cd C:\AI\ComfyUI
.\venv\Scripts\activate
pip install -r requirements_train.txt
```

---

## Étape 1 — Préparer le dataset

### Option A : Télécharger CelebA (recommandé pour débuter)
```bash
python prepare_dataset.py --output ./dataset --source celeba --max_images 2000
```

### Option B : Utiliser vos propres images
```bash
# Mettre vos images dans un dossier, puis :
python prepare_dataset.py --output ./dataset --input_dir C:\mes_photos
```

### Option C : Dataset mixte (meilleurs résultats)
Combinez CelebA + vos propres photos pour un dataset de ~2000 images variées.

Le script va automatiquement :
- ✅ Garder uniquement les images avec 1 visage bien visible
- ✅ Filtrer les visages trop petits
- ✅ Resize à 512x512
- ✅ Sauvegarder les embeddings InsightFace

---

## Étape 2 — Entraîner PuLID

### Configuration recommandée pour RTX 3090 (24 Go)

```bash
python train_pulid_klein.py \
  --dataset ./dataset/filtered \
  --output ./output \
  --comfyui_path C:/AI/ComfyUI \
  --dim 4096 \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4
```

### Paramètres

| Paramètre | Valeur recommandée | Description |
|---|---|---|
| `--dim` | `4096` | Klein 9B. Utiliser `3072` pour Klein 4B |
| `--epochs` | `20` | 10 epochs = résultats corrects, 30+ = meilleurs |
| `--batch_size` | `4` | Pour 24 Go VRAM. Réduire à 2 si OOM |
| `--lr` | `1e-4` | Learning rate de départ |
| `--save_every` | `500` | Checkpoint tous les 500 steps |

---

## Étape 3 — Utiliser le modèle entraîné

```bash
# Copier le meilleur modèle vers ComfyUI
copy output\pulid_klein_best.safetensors C:\AI\ComfyUI\models\pulid\pulid_klein_9b_v1.safetensors
```

Puis dans ComfyUI, dans le node `Load PuLID Flux.2 Klein Model` :
- Sélectionner `pulid_klein_9b_v1.safetensors`
- Variant : `klein_9B (dim=4096)`

---

## Phases d'entraînement

### Phase 1 (actuelle) — Embedding Only
Le script entraîne l'IDFormer à encoder correctement l'identité faciale.
**Flux.2 Klein n'est pas requis pour cette phase.**

```bash
python train_pulid_klein.py --dataset ./dataset/filtered --output ./output
```

### Phase 2 (avancée) — Full Pipeline
Entraînement avec le modèle Flux.2 Klein complet pour affiner l'injection.
**Requiert ~20 Go VRAM supplémentaires ou gradient checkpointing.**

```bash
python train_pulid_klein.py \
  --dataset ./dataset/filtered \
  --output ./output_phase2 \
  --flux_model_path C:/AI/ComfyUI/models/diffusion_models/flux-2-klein-9b-fp8.safetensors \
  --epochs 10 \
  --lr 5e-5
```

---

## Monitoring de l'entraînement

Les métriques à surveiller :

| Métrique | Début | Bon résultat |
|---|---|---|
| `loss` | ~0.8-1.0 | < 0.3 |
| `loss_identity` | ~0.8 | < 0.25 |
| `loss_reg` | ~0.01 | ~0.005 |

Si la loss ne descend pas après 5 epochs → réduire `--lr` à `5e-5`.

---

## Structure des fichiers

```
PuLID-Klein-Training/
├── prepare_dataset.py      ← Prépare le dataset
├── train_pulid_klein.py    ← Script d'entraînement principal
├── requirements_train.txt  ← Dépendances
└── README_TRAINING.md      ← Ce fichier

dataset/
├── raw/                    ← Images téléchargées brutes
└── filtered/
    ├── images/             ← Images filtrées et normalisées
    ├── rejected/           ← Images rejetées (debug)
    └── metadata.json       ← Embeddings InsightFace pré-calculés

output/
├── checkpoints/            ← Checkpoints intermédiaires
├── pulid_klein_best.safetensors   ← Meilleur modèle
└── pulid_klein_latest.safetensors ← Dernier checkpoint
```

---

## Partager avec la communauté

Si tu obtiens de bons résultats, tu peux partager le modèle sur HuggingFace :

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='output/pulid_klein_best.safetensors',
    path_in_repo='pulid_klein_9b_v1.safetensors',
    repo_id='TON_USERNAME/PuLID-Flux2Klein',
    repo_type='model',
)
"
```

---

## Crédits

- PuLID original : [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID)
- Flux.2 Klein : [Black Forest Labs](https://blackforestlabs.ai)
- Adaptation Klein + script entraînement : **toi** 🎉 (Mars 2026)
