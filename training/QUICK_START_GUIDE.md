# 🚀 Guide Démarrage Rapide - Entraînement PuLID

## 📋 Prérequis

### 1. Installation Environnement

```bash
# PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dépendances principales
pip install accelerate diffusers transformers
pip install open_clip_torch insightface safetensors
pip install opencv-python pillow tqdm

# Vérification
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Structure des Dossiers

```
PuLID-Training/
├── train_pulid_COMPLETE.py       # ← Le script complet
├── dataset/
│   └── filtered/
│       ├── images/                 # Vos photos
│       │   ├── person_001_01.jpg
│       │   ├── person_001_02.jpg
│       │   └── ...
│       └── metadata.json           # Généré par prepare_dataset.py
├── output/                         # Checkpoints Phase 1
└── output_phase2/                  # Checkpoints Phase 2
```

### 3. Dataset Requis

**Phase 1 et 2** :
- ✅ Minimum : 50 identités, 5 photos chacune = 250 images
- ✅ Recommandé : 200 identités, 10 photos chacune = 2000 images
- ✅ Idéal : 500+ identités, 10-20 photos = 5000-10000 images

**Critères qualité** :
- Photos HD (>512px)
- Visage clairement visible
- Diversité : angles, expressions, éclairages
- Pas de masques/lunettes de soleil/occlusions

---

## 🎯 PHASE 1 : Entraînement IDFormer

### Objectif
Apprendre à créer des **tokens d'identité riches** qui capturent l'essence du visage.

### Commande Simple

```bash
python train_pulid_COMPLETE.py \
    --dataset ./dataset/filtered \
    --output ./output \
    --phase 1
```

**Paramètres auto-configurés** :
- Epochs : 25
- Batch size : 4
- Learning rate : 1e-4
- Image size : 512

### Commande Personnalisée

```bash
python train_pulid_COMPLETE.py \
    --dataset ./dataset/filtered \
    --output ./output \
    --phase 1 \
    --comfyui_path /path/to/ComfyUI \
    --dim 4096 \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_every 500
```

### Monitoring Phase 1

**Regarder ces métriques** :

```
Epoch 1:
  loss_identity  : 0.280  ← Début
  loss_reg       : 0.008
  loss_total     : 0.288

Epoch 10:
  loss_identity  : 0.085  ← En progrès
  loss_reg       : 0.005
  loss_total     : 0.090

Epoch 20:
  loss_identity  : 0.042  ← ✅ BON
  loss_reg       : 0.004
  loss_total     : 0.046

Epoch 25:
  loss_identity  : 0.028  ← ✅ EXCELLENT
  loss_reg       : 0.003
  loss_total     : 0.031  ← Meilleur
```

**Critères de succès** :
- ✅ `loss_identity < 0.05` : Bon, peut passer à Phase 2
- ✅ `loss_identity < 0.03` : Excellent, résultats optimaux
- ⚠️ `loss_identity > 0.1` : Pas encore convergé, continuer
- ❌ `loss_identity stagne > 0.15` : Problème dataset ou config

### Durée Estimée

| GPU | Batch Size | Durée (25 epochs) |
|-----|------------|-------------------|
| RTX 3090 24GB | 4 | 2-3 heures |
| RTX 4090 24GB | 8 | 1-2 heures |
| A100 40GB | 12 | 1 heure |
| RTX 3060 12GB | 2 | 4-6 heures |

### Fichiers Générés

```
output/
├── checkpoints/
│   ├── pulid_klein_phase1_epoch001_step000500.safetensors
│   ├── pulid_klein_phase1_epoch010_step005000.safetensors
│   └── ...
├── pulid_klein_phase1_latest.safetensors     # Dernier checkpoint
└── pulid_klein_phase1_best.safetensors       # ⭐ MEILLEUR (à utiliser)
```

### Troubleshooting Phase 1

**Problème : CUDA Out of Memory**
```bash
# Solution : Réduire batch_size
python train_pulid_COMPLETE.py --phase 1 --batch_size 2
```

**Problème : Loss ne descend pas**
```bash
# Vérifier le dataset
python -c "
from train_pulid_COMPLETE import FaceDataset
ds = FaceDataset('./dataset/filtered')
print(f'{len(ds)} images chargées')
"

# Si <200 images : dataset insuffisant
# Solution : Collecter plus de photos
```

**Problème : Loss descend puis remonte (overfitting)**
```bash
# Solution : Utiliser le meilleur checkpoint (pas le dernier)
# Le script sauvegarde automatiquement le meilleur
```

---

## 🎯 PHASE 2 : Entraînement PerceiverCA

### Objectif
Apprendre à **injecter l'identité** dans les image tokens de Flux.2.

### Prérequis

1. **Checkpoint Phase 1** : `output/pulid_klein_phase1_best.safetensors`
2. **Flux.2 Klein** : Télécharger depuis HuggingFace
   ```bash
   # Klein 9B (recommandé)
   huggingface-cli download black-forest-labs/FLUX.2-Klein-9B \
       --local-dir ./models/flux2-klein-9b
   
   # Ou utiliser un .safetensors
   # flux-2-klein-9b-fp8.safetensors
   ```

### Commande Simple

```bash
python train_pulid_COMPLETE.py \
    --dataset ./dataset/filtered \
    --output ./output_phase2 \
    --phase 2 \
    --resume ./output/pulid_klein_phase1_best.safetensors \
    --flux_model_path ./models/flux2-klein-9b
```

**Paramètres auto-configurés** :
- Epochs : 15
- Batch size : 2
- Learning rate : 5e-5

### Commande Personnalisée

```bash
python train_pulid_COMPLETE.py \
    --dataset ./dataset/filtered \
    --output ./output_phase2 \
    --phase 2 \
    --resume ./output/pulid_klein_phase1_best.safetensors \
    --flux_model_path ./models/flux-2-klein-9b-fp8.safetensors \
    --comfyui_path /path/to/ComfyUI \
    --dim 4096 \
    --epochs 20 \
    --batch_size 1 \
    --lr 3e-5
```

### Monitoring Phase 2

**Regarder ces métriques** :

```
Epoch 1:
  loss_identity  : 0.045  ← Maintenu depuis Phase 1
  loss_perceiver : 0.018  ← Début apprentissage PerceiverCA
  loss_reg       : 0.005
  loss_total     : 0.113

Epoch 5:
  loss_identity  : 0.038  ← Légère amélioration
  loss_perceiver : 0.008  ← Descend bien
  loss_reg       : 0.004
  loss_total     : 0.088

Epoch 10:
  loss_identity  : 0.032  ← ✅ BON
  loss_perceiver : 0.005  ← ✅ Convergé
  loss_reg       : 0.003
  loss_total     : 0.072

Epoch 15:
  loss_identity  : 0.028  ← ✅ EXCELLENT
  loss_perceiver : 0.004  ← ✅ Très bon
  loss_reg       : 0.003
  loss_total     : 0.063  ← Meilleur
```

**Critères de succès** :
- ✅ `loss_identity < 0.03` : Excellent
- ✅ `loss_perceiver < 0.01` : PerceiverCA bien entraîné
- ⚠️ `loss_identity augmente` : Vérifier dataset ou learning rate
- ⚠️ `loss_perceiver > 0.02` : Pas encore convergé

### Durée Estimée

| GPU | Batch Size | Durée (15 epochs) |
|-----|------------|-------------------|
| RTX 3090 24GB | 2 | 8-12 heures |
| RTX 4090 24GB | 3 | 6-8 heures |
| A100 40GB | 4 | 4-6 heures |
| RTX 3060 12GB | 1 | 16-20 heures |

⚠️ **Phase 2 est BEAUCOUP plus lente que Phase 1** (Flux.2 chargé en mémoire)

### Fichiers Générés

```
output_phase2/
├── checkpoints/
│   ├── pulid_klein_phase2_epoch001_step000500.safetensors
│   ├── pulid_klein_phase2_epoch010_step005000.safetensors
│   └── ...
├── pulid_klein_phase2_latest.safetensors
└── pulid_klein_phase2_best.safetensors  # ⭐ MEILLEUR (à copier dans ComfyUI)
```

### Troubleshooting Phase 2

**Problème : CUDA Out of Memory**
```bash
# Solution 1 : Réduire batch_size
--batch_size 1

# Solution 2 : Utiliser Klein 4B au lieu de 9B
--dim 3072
--flux_model_path ./models/flux-2-klein-4b-fp8.safetensors
```

**Problème : loss_identity augmente**
```bash
# Vérifier que Phase 1 était bon
python -c "
from safetensors.torch import load_file
state = load_file('./output/pulid_klein_phase1_best.safetensors')
print(f'Checkpoint Phase 1 chargé : {len(state)} clés')
"

# Si Phase 1 mauvaise : recommencer Phase 1
```

**Problème : loss_perceiver ne descend pas**
```bash
# Solution : Réduire learning rate
--lr 2e-5

# Ou augmenter epochs
--epochs 25
```

---

## 🧪 Test dans ComfyUI

### 1. Copier le Checkpoint

```bash
# Copier le meilleur checkpoint Phase 2
cp output_phase2/pulid_klein_phase2_best.safetensors \
   ComfyUI/models/pulid/pulid_klein_custom.safetensors
```

### 2. Workflow ComfyUI

```
┌──────────────────┐
│ Load Checkpoint  │  Flux.2 Klein 9B
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Load VAE         │
└────────┬─────────┘
         │
┌────────▼─────────┐     ┌─────────────────┐
│ Load InsightFace │────►│ Load EVA-CLIP   │
└────────┬─────────┘     └────────┬────────┘
         │                        │
┌────────▼────────────────────────▼─────┐
│ Load PuLID Model                      │
│ File: pulid_klein_custom.safetensors  │
└────────┬──────────────────────────────┘
         │
┌────────▼─────────┐
│ Input Image Ref  │  Photo de la personne
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Apply PuLID      │
│ - Weight: 1.0    │
│ - Start: 0.0     │
│ - End: 1.0       │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Text Prompt      │
│ Positive: "professional portrait, sunset by river"
│ Negative: "blurry, low quality, deformed face"
└────────┬─────────┘
         │
┌────────▼─────────┐
│ KSampler         │
│ - Steps: 28      │
│ - CFG: 3.5       │
│ - Sampler: euler │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ VAE Decode       │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Save Image       │
└──────────────────┘
```

### 3. Tests Progressifs

**Test 1 : Conservation Identité Pure**
```
Image Ref : Votre selfie
Prompt    : "professional headshot, neutral background, studio lighting"
Weight    : 1.0
CFG       : 3.5

Résultat attendu : Visage IDENTIQUE à la ref
```

**Test 2 : Changement Léger**
```
Image Ref : Votre selfie
Prompt    : "portrait at sunset by the river, golden hour"
Weight    : 0.9
CFG       : 4.0

Résultat attendu : MÊME visage, nouveau contexte
```

**Test 3 : Changement Radical**
```
Image Ref : Votre selfie
Prompt    : "cyberpunk character, neon city, futuristic"
Weight    : 1.2
CFG       : 4.5

Résultat attendu : Visage reconnaissable, style différent
```

### 4. Ajustement des Paramètres

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| **weight** | 0.6-0.8 | Identité subtile, plus de créativité |
| | 0.8-1.0 | ✅ Équilibré (recommandé) |
| | 1.0-1.5 | Identité forte, moins de variation |
| | >1.5 | ⚠️ Risque d'artefacts |
| **start_at** | 0.0 | Injection dès le début (recommandé) |
| | 0.2 | Injection après structure initiale |
| **end_at** | 1.0 | Injection jusqu'à la fin (recommandé) |
| | 0.8 | Arrête avant détails finaux |
| **CFG scale** | 2.0-3.0 | Créatif, identité peut dériver |
| | 3.5-4.5 | ✅ Équilibré (recommandé) |
| | 5.0+ | Moins créatif, identité forte |

---

## 📊 Évaluation des Résultats

### Grille d'Évaluation

| Aspect | Score /10 | Notes |
|--------|-----------|-------|
| Visage identique | __/10 | Traits faciaux préservés |
| Cheveux identiques | __/10 | Coupe, couleur, style |
| Expression cohérente | __/10 | Structure du visage |
| Détails (barbe, grain peau) | __/10 | Texture préservée |
| Genre respecté | __/10 | Masculin/féminin |
| **TOTAL** | **__/50** | |

**Interprétation** :
- **45-50/50** : ✅ Parfait ! Clonage d'identité réussi
- **40-44/50** : ✅ Très bon, légers ajustements possibles
- **35-39/50** : ⚠️ Bon mais identité partiellement perdue
- **30-34/50** : ⚠️ Moyen, vérifier les paramètres
- **<30/50** : ❌ Échec, réentraîner ou améliorer dataset

### Que Faire Si...

**Identité complètement perdue** :
1. Vérifier loss Phase 2 < 0.05
2. Augmenter weight à 1.3-1.5
3. Tester avec CFG 4.0-4.5
4. Si toujours mauvais : réentraîner Phase 2

**Identité OK mais artefacts** :
1. Réduire weight à 0.7-0.9
2. Réduire CFG à 3.0-3.5
3. Augmenter steps à 35-40

**Identité dérive sur certains prompts** :
1. Normal pour prompts très éloignés
2. Augmenter weight pour ces cas
3. Simplifier le prompt

**Mélange d'identités (plusieurs personnes)** :
1. Bug dataset : même identité avec plusieurs IDs
2. Vérifier metadata.json
3. Re-préparer le dataset

---

## 📈 Optimisations Avancées

### Augmenter la Qualité

**Plus de données** :
- Passer de 50 → 200 identités : +30% qualité
- Passer de 5 → 10 photos/identité : +20% robustesse

**Plus d'epochs** :
```bash
# Phase 1 : jusqu'à loss < 0.02
--epochs 40

# Phase 2 : jusqu'à convergence complète
--epochs 25
```

**Learning rate adaptatif** :
```bash
# Si loss stagne, réduire LR
--lr 3e-5  # au lieu de 5e-5
```

### Réduire la VRAM

**Phase 1** :
```bash
--batch_size 2
```

**Phase 2** :
```bash
# Utiliser Klein 4B
--dim 3072
--flux_model_path flux-2-klein-4b.safetensors
--batch_size 1
```

### Accélérer l'Entraînement

**Multi-GPU** (si disponible) :
```bash
# À implémenter : DDP (Distributed Data Parallel)
# Pour l'instant : utiliser 1 GPU le plus puissant
```

**Mixed Precision** (déjà activé) :
```python
# Déjà dans le code :
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    ...
```

---

## 🔍 FAQ

**Q : Combien de temps total ?**
R : Phase 1 (3h) + Phase 2 (10h) = ~13h sur RTX 3090

**Q : Puis-je skip Phase 1 ?**
R : Non, Phase 1 est essentielle. Phase 2 sans Phase 1 = échec garanti.

**Q : Quel GPU minimum ?**
R : 12GB VRAM (RTX 3060) avec batch_size=1. Recommandé : 24GB (RTX 3090).

**Q : Klein 4B vs Klein 9B ?**
R : Klein 9B = meilleure qualité. Klein 4B = plus rapide, moins de VRAM.

**Q : Combien de photos par personne ?**
R : Minimum 5, idéal 10-20, avec diversité (angles, expressions, éclairages).

**Q : Peut-on utiliser des photos basse qualité ?**
R : Oui mais résultats dégradés. Privilégier >512px, bien éclairées.

**Q : Early stopping à 5 epochs, c'est assez ?**
R : Oui, le meilleur checkpoint est sauvegardé avant. Patience=5 évite l'overfitting.

---

## ✅ Checklist Complète

### Avant Phase 1
- [ ] PyTorch + CUDA installés
- [ ] Dataset préparé (>250 images, metadata.json)
- [ ] ComfyUI installé avec custom node PuLID
- [ ] GPU avec >12GB VRAM disponible

### Pendant Phase 1
- [ ] loss_identity descend régulièrement
- [ ] Pas de CUDA OOM
- [ ] Checkpoints sauvegardés

### Après Phase 1
- [ ] loss_identity < 0.05 (idéalement < 0.03)
- [ ] Checkpoint best.safetensors existe
- [ ] Prêt pour Phase 2

### Avant Phase 2
- [ ] Flux.2 Klein téléchargé
- [ ] Checkpoint Phase 1 disponible
- [ ] GPU stable (Phase 2 est longue !)

### Pendant Phase 2
- [ ] loss_identity maintenue < 0.05
- [ ] loss_perceiver descend < 0.01
- [ ] Pas de divergence

### Après Phase 2
- [ ] loss_identity < 0.03
- [ ] loss_perceiver < 0.008
- [ ] Checkpoint copié dans ComfyUI

### Tests ComfyUI
- [ ] Test 1 (identité pure) : ✅
- [ ] Test 2 (changement style) : ✅
- [ ] Test 3 (changement radical) : ⚠️
- [ ] Score global >40/50

---

## 🎉 Succès !

Si tous les tests passent, félicitations ! Vous avez :
- ✅ Un modèle PuLID entraîné custom
- ✅ Clonage d'identité fonctionnel
- ✅ Génération d'images avec n'importe quelle identité

**Prochaines étapes** :
- Partager vos résultats sur GitHub
- Expérimenter avec différents styles
- Affiner avec plus de données

---

**Version** : 1.0  
**Date** : 19 mars 2026  
**Auteur** : Claude (Anthropic)
