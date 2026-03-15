# ComfyUI-PuLID-Flux2Klein

Custom node ComfyUI implémentant PuLID (Pure Identity) pour **FLUX.2 Klein** (4B et 9B).

---

## ⚙️ Architecture — Pourquoi ce node existe

PuLID original ne supporte que Flux.1 Dev. FLUX.2 Klein a une architecture différente :

| | Flux.1 Dev | Flux.2 Klein 4B | Flux.2 Klein 9B |
|---|---|---|---|
| Double blocks | 19 | 5 | 8 |
| Single blocks | 38 | 20 | 24 |
| Hidden dim | 4096 | 3072 | 4096 |
| Modulation | Par bloc | Partagée | Partagée |
| Text encoder | T5 | Qwen3-4B | Qwen3-8B |
| Conditioning | [B,77,4096] | [B,512,12288] | [B,512,12288] |

Ce node adapte l'injection d'embeddings faciaux à ces spécificités.

---

## 📦 Installation

### 1. Copier dans custom_nodes
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git ComfyUI-PuLID-Flux2Klein
cd ComfyUI-PuLID-Flux2Klein
pip install -r requirements.txt
```

### 2. EVA-CLIP (automatique)
EVA02-CLIP-L-14-336 se télécharge automatiquement au premier lancement via `open_clip` (~800 Mo).

> ⚠️ **Ne pas** installer `eva_clip` depuis GitHub — ce package est cassé et non requis.
> `open-clip-torch` est déjà inclus dans `requirements.txt`.

### 3. Télécharger les modèles

#### InsightFace AntelopeV2
Télécharger depuis : https://huggingface.co/MonsterMMORPG/InsightFace_AntelopeV2
Placer dans : `ComfyUI/models/insightface/models/antelopev2/`

#### EVA02-CLIP-L-14-336
Téléchargement automatique au premier lancement.
Ou manuellement depuis HuggingFace → placer dans `ComfyUI/models/clip/`

#### Poids PuLID pour Flux.2 Klein
> ⚠️ Il n'existe pas encore de poids pré-entraînés officiels pour Flux.2 Klein.
> 
> **Option A** : Utiliser les poids PuLID-Flux (Flux.1) comme point de départ
>   - Télécharger depuis : https://huggingface.co/guozinan/PuLID
>   - Placer dans : `ComfyUI/models/pulid/`
>   - Note: compatibilité partielle (dim=4096 → Klein 9B uniquement sans ajustement)
>
> **Option B** : Lancer sans poids (le node crée un IDFormer aléatoire)
>   - Résultats aléatoires mais le node fonctionnera sans erreur
>   - Utile pour tester l'intégration avant d'avoir des poids fine-tunés

---

## 🔌 Workflow ComfyUI

```
[Image de référence]
       │
[PuLID Klein — Load InsightFace]─────────────┐
[PuLID Klein — Load EVA-CLIP]────────────────┤
[PuLID Klein — Load Model]───────────────────┤
                                             ▼
[Ton modèle Flux.2 Klein] ──────► [Apply PuLID ✦ Flux.2 Klein] ──► [KSampler / modèle patché]
                                             │
                                    weight, start_at, end_at
```

### Nodes disponibles

| Node | Description |
|---|---|
| `Load InsightFace (PuLID Klein)` | Charge le détecteur de visage AntelopeV2 |
| `Load EVA-CLIP (PuLID Klein)` | Charge l'encodeur visuel EVA02-CLIP |
| `Load PuLID Flux.2 Klein Model` | Charge les poids PuLID (.safetensors/.pt) |
| `Apply PuLID ✦ Flux.2 Klein` | **Node principal** — patch le modèle Flux.2 Klein |
| `PuLID Klein — Face Debug Preview` | Visualise les visages détectés |

### Paramètres de Apply

| Paramètre | Défaut | Description |
|---|---|---|
| `weight` | 0.8 | Force de l'injection (0=off, 1.5=max) |
| `start_at` | 0.0 | Début de l'injection (0.0 = dès le début) |
| `end_at` | 1.0 | Fin de l'injection |
| `face_index` | 0 | Quel visage utiliser si plusieurs détectés |

---

## 🏗️ Comment ça marche

### Pipeline complet

```
Image référence
    │
    ├──► InsightFace AntelopeV2
    │         └─► embedding 512-dim (identité faciale)
    │
    └──► EVA02-CLIP-L-14-336
              └─► features 768-dim (features visuelles haute-niveau)
                       │
                       ▼
              IDFormer (MLP + Perceiver)
                       │
                       ▼
              id_tokens [B, 4, dim]
                       │
              ┌────────┴────────────────────────────────┐
              │  Injection dans Flux.2 Klein             │
              │                                          │
              │  double_blocks[0,2,4,...]:               │
              │    img_tokens += w * PerceiverCA(img, id)│
              │                                          │
              │  single_blocks[0,4,8,...]:               │
              │    out[:n_img] += w * PerceiverCA(x, id) │
              └──────────────────────────────────────────┘
```

### Adaptations spécifiques à Klein

1. **Shared modulation** : Flux.2 Klein partage les paramètres AdaLayerNorm entre tous les blocs.
   → L'injection PuLID agit *après* la modulation, donc indépendamment.

2. **single_transformer_blocks** : Dans Klein, ces blocs fusionnent `attn.to_qkv_mlp_proj`.
   → On injecte sur les tokens de sortie (pas sur Q/K/V), ce qui est plus stable.

3. **Conditioning shape** : Qwen3 produit [B, 512, 12288], projeté en [B, 512, dim] par `txt_in`.
   → PuLID n'interagit pas avec le text stream, uniquement avec img_stream.

4. **In-context conditioning** : Klein passe l'image de référence comme tokens additionnels.
   → Le n_img estimé = total_tokens - 512 (txt_tokens).

---

## ⚡ Entraîner des poids dédiés (avancé)

Pour de meilleurs résultats, il faudra entraîner l'IDFormer sur Flux.2 Klein :

```bash
# Dataset : ~1000 images portrait avec captions
# Utiliser Consistent Character Creator 3.5 pour générer le dataset

python train_pulid_klein.py \
  --model black-forest-labs/FLUX.2-klein-4B \
  --dataset ./face_dataset \
  --output ./pulid_klein_4b.safetensors \
  --epochs 10 \
  --lr 1e-4
```

> Un script d'entraînement `train_pulid_klein.py` sera ajouté prochainement.

---

## 🐛 Dépannage

**"Aucun visage détecté"**
→ Vérifier que AntelopeV2 est bien dans `models/insightface/models/antelopev2/`
→ Utiliser le node `PuLID Klein — Face Debug Preview` pour visualiser

**"EVA-CLIP non disponible"**
→ `pip install git+https://github.com/baaivision/EVA.git#subdirectory=EVA-CLIP`

**"Impossible de trouver double_blocks"**
→ Votre version de ComfyUI/diffusers peut nommer les blocs différemment.
→ Ouvrir une issue avec le nom exact des attributs de votre modèle.

**Résultats non cohérents**
→ Aucun poids pré-entraîné sur Klein n'existe encore → entraîner ou attendre la communauté.
→ Essayer weight=0.9, start_at=0.1, end_at=0.9

---

## 📄 Licence

MIT — basé sur PuLID original (Apache 2.0) par ToTheBeginning/PuLID
et ComfyUI-PuLID-Flux par balazik/lldacing.
