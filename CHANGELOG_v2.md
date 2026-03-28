# PuLID-Flux2 v2.0 - CHANGELOG & INSTALLATION

## 🎯 Améliorations principales

### 1. **Auto-détection dtype selon GPU** ✅
**Problème résolu:** RTX 3090 (Ampere) n'a pas de support bf16 natif → émulation via fp32 = 10x+ plus lent

**Solution:**
- Détection automatique de l'architecture GPU via `torch.cuda.get_device_capability()`
- **Ampere (RTX 30xx, A100):** dtype = `fp16` (optimal)
- **Ada (RTX 40xx):** dtype = `bf16` (natif)
- **Hopper (H100):** dtype = `bf16` (natif + Tensor Cores)

**Avant:**
```python
dtype = torch.bfloat16  # Hardcodé → lent sur RTX 3090
```

**Après:**
```python
dtype = get_optimal_dtype(device)  # Auto fp16 sur RTX 3090
```

---

### 2. **Cleanup VRAM agressif** 🗑️
**Problème résolu:** Memory leak après chaque génération → run 2+ devient lent

**Solution:**
- Garbage collection automatique après chaque application
- Option `cleanup_vram` (défaut: True) dans le node Apply PuLID
- Fonction `clear_model_cache()` pour vider le cache global

**Impact:**
- Run 1: 2s/it
- Run 2+: 2s/it (stable, pas de ralentissement)

---

### 3. **Cache configurable** 🎛️
**Problème résolu:** Cache global peut causer des conflits avec certains workflows

**Solution:**
- Option `use_cache` dans les loaders EVA-CLIP et InsightFace
- Nouveau node **PuLIDCacheCleaner** pour vider manuellement le cache
- Cache peut être désactivé si nécessaire

**Usage:**
```
PuLIDEVACLIPLoader → use_cache=False (reload à chaque fois)
PuLIDCacheCleaner → clear=True (vide le cache)
```

---

### 4. **Détection de conflits de patches** ⚠️
**Problème résolu:** Collisions silencieuses avec Consistency Models, LoRAs, etc.

**Solution:**
- Scan automatique des markers de patches (`_lora_patched`, `_consistency_patched`, etc.)
- Warning si conflit détecté
- Logs détaillés en debug mode

**Exemple output:**
```
[PuLID] ⚠️  3 patch conflict(s) detected - may cause instability
  - double_block_2 has _consistency_patched
  - single_block_6 has _lora_patched
```

---

### 5. **Paramètres sigma range configurables** 📊
**Nouveau paramètre:** `sigma_start` et `sigma_end` dans Apply PuLID

**Usage:**
- Défaut: `sigma_start=0.0, sigma_end=1.0` (tout le denoising)
- Custom: `sigma_start=0.3, sigma_end=0.8` (injecte PuLID seulement milieu du denoising)

**Cas d'usage:**
- Préservation légère: `sigma_start=0.0, sigma_end=0.5`
- Préservation forte: `sigma_start=0.0, sigma_end=1.0`

---

### 6. **Debug mode étendu** 🔍
**Informations supplémentaires:**
- GPU capability (compute version)
- Dtype sélectionné automatiquement
- Blocks patchés (indices)
- Détection de conflits de patches
- Shapes des embeddings à chaque étape

**Activation:**
```
Apply PuLID → debug_mode=True
```

---

## 📋 Nouveaux nodes

### **PuLIDCacheCleaner** 🗑️
Vide le cache global des modèles EVA-CLIP et InsightFace

**Usage:**
1. Ajoute le node au workflow
2. Set `clear=True`
3. Execute → cache vidé + VRAM libérée

**Cas d'usage:**
- Changement de workflow
- Memory leak suspect
- Avant training/fine-tuning

---

## 🚀 Installation

### Méthode 1: Remplacement direct
```bash
# Backup ancien fichier
copy C:\AI\ComfyUI\custom_nodes\ComfyUI-PuLID-Flux2\pulid_flux2.py C:\AI\ComfyUI\custom_nodes\ComfyUI-PuLID-Flux2\pulid_flux2.py.backup

# Remplace avec v2
copy pulid_flux2_v2_complete.py C:\AI\ComfyUI\custom_nodes\ComfyUI-PuLID-Flux2\pulid_flux2.py
```

### Méthode 2: Nouvelle installation
```bash
# Clone le repo (quand publié sur GitHub)
cd C:\AI\ComfyUI\custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git
cd ComfyUI-PuLID-Flux2
git checkout v2.0
```

---

## 🔧 Configuration recommandée

### Pour RTX 3090 (24GB VRAM)
```batch
# Launch script
python main.py --force-fp16 --normalvram --preview-method auto
```

**Nodes settings:**
- `use_cache=True` (économise temps de chargement)
- `cleanup_vram=True` (évite memory leak)
- `debug_mode=False` (sauf debug)

**VRAM budget:**
- Klein 4B + PuLID: ~16GB → confortable
- Klein 9B + PuLID: ~21GB → OK
- Klein 9B + PuLID + Consistency: ~24GB → limite (désactive LoRAs)

---

### Pour RTX 4090 (24GB VRAM)
```batch
# Launch script
python main.py --highvram --preview-method auto
```

**Différence vs RTX 3090:**
- bf16 natif → peut utiliser `--highvram` sans problème
- Même VRAM (24GB) mais meilleure gestion

---

## 🐛 Troubleshooting

### "Toujours lent au 2ème run"
**Solution:**
1. Set `cleanup_vram=True` dans Apply PuLID
2. Utilise `--normalvram` au lieu de `--highvram`
3. Ajoute node PuLIDCacheCleaner entre générations

### "Conflit avec Consistency Model"
**Solution:**
1. Désactive Consistency OU PuLID (pas les deux)
2. OU passe à Klein 4B (économise ~4.5GB)
3. OU rollback drivers NVIDIA 595.97 → 560.94

### "dtype bf16 toujours utilisé sur RTX 3090"
**Vérification:**
```python
# Dans ComfyUI console après loading
import torch
print(torch.cuda.get_device_capability())  # Devrait être (8, 6) pour RTX 3090
```

Si le dtype est toujours bf16 → vérifier que la v2 est bien chargée (check console log au démarrage)

---

## 📊 Benchmarks

### RTX 3090 + Klein 9B + PuLID

**Avant v2 (avec PyTorch 2.5.1 + xformers 0.0.35 + drivers 595.97):**
- Run 1: ~40s/it (bf16 forcé)
- Run 2+: CRASH ou 12s/it (memory leak + swap)

**Après v2 (avec PyTorch 2.4.0 + xformers 0.0.27 + drivers 560.94 + --normalvram):**
- Run 1: ~2s/it (fp16 auto)
- Run 2+: ~2s/it stable (cleanup VRAM)

**Gain:** ~20x plus rapide + stabilité complète

---

## 🎓 Migration depuis v1

### Changements breaking
**Aucun!** La v2 est **100% compatible** avec les workflows v1.

### Nouveaux paramètres (optionnels)
- `sigma_start` / `sigma_end`: défaut = `0.0` / `1.0` (comportement v1)
- `cleanup_vram`: défaut = `True` (améliore stabilité)
- `use_cache`: défaut = `True` (comportement v1)

### Nouveaux nodes (optionnels)
- **PuLIDCacheCleaner**: pour workflows avancés
- Display names changés (suffixe " v2") pour distinguer facilement

---

## 📝 Notes techniques

### get_optimal_dtype() logic
```python
capability = torch.cuda.get_device_capability(device)
major, minor = capability

if major >= 9:           # Hopper (H100)
    return torch.bfloat16
elif major == 8 and minor >= 9:  # Ada (RTX 40xx)
    return torch.bfloat16
elif major == 8:         # Ampere (RTX 30xx, A100)
    return torch.float16
else:                    # Older
    return torch.float16
```

### Patch conflict detection
Scanne tous les blocks pour ces markers:
- `_lora_patched`
- `_consistency_patched`
- `_ipadapter_patched`
- `_custom_forward`

Si trouvé → warning (continue quand même, mais prévient l'utilisateur)

---

## 🤝 Contribution

Pour contribuer ou reporter des bugs:
- GitHub: https://github.com/iFayens/ComfyUI-PuLID-Flux2
- Issues: Décris ton GPU, drivers, PyTorch version, log complet

---

## 📜 License

Même license que v1 (à définir selon le repo original)

---

## 🙏 Credits

**Développé par:** Mems (iFayens)  
**Basé sur:** PuLID paper + Flux.2 architecture  
**Debuggé avec:** Claude (Anthropic) — session marathon 27 mars 2026 🚀

---

**Version:** 2.0.0  
**Date:** 27 mars 2026  
**Tested on:** RTX 3090, PyTorch 2.4.0, ComfyUI 0.18.1
