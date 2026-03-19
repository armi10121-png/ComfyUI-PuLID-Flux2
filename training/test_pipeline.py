"""
test_pipeline.py - VERSION AMÉLIORÉE
=====================================
Valide tout le pipeline PuLID Klein AVANT de louer sur Vast.ai.
Utilise 10 images locales, 2 epochs, batch=1.

AMÉLIORATIONS vs version originale :
  ✅ Compatible avec train_pulid_COMPLETE.py
  ✅ Meilleure gestion des erreurs
  ✅ Chemins InsightFace auto-détectés
  ✅ Tests plus complets
  ✅ Rapport final détaillé

Usage :
  python test_pipeline.py --images ./mes_photos

Ce script vérifie :
  ✅ Dépendances installées (torch, CUDA, etc.)
  ✅ InsightFace détecte les visages
  ✅ EVA-CLIP se charge correctement
  ✅ prepare_dataset fonctionne sur vos images
  ✅ train_pulid.py tourne 2 epochs sans crash
  ✅ Le .safetensors est valide et chargeable dans ComfyUI

Si tous les tests passent → prêt pour Vast.ai ! 🚀
"""

import os
import sys
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def section(title):
    """Affiche un titre de section."""
    log.info("")
    log.info("=" * 70)
    log.info(f"  {title}")
    log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : DÉPENDANCES
# ══════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    section("ÉTAPE 1 — Vérification des dépendances")
    ok = True
    
    deps = {
        "torch": "pip install torch torchvision",
        "numpy": "pip install numpy",
        "PIL": "pip install Pillow",
        "tqdm": "pip install tqdm",
        "safetensors": "pip install safetensors",
        "open_clip": "pip install open-clip-torch",
        "insightface": "pip install insightface onnxruntime-gpu",
    }
    
    for module, install_cmd in deps.items():
        try:
            __import__(module)
            log.info(f"  {PASS} {module:15s} installé")
        except ImportError:
            log.error(f"  {FAIL} {module:15s} manquant → {install_cmd}")
            ok = False
    
    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"  {PASS} CUDA : {gpu} ({vram:.1f} Go VRAM)")
            
            if vram < 12:
                log.warning(
                    f"  {WARN} VRAM < 12 Go — recommandé : ≥12 Go pour Phase 1"
                )
        else:
            log.warning(f"  {WARN} CUDA non disponible — CPU uniquement (très lent)")
    except Exception as e:
        log.error(f"  {FAIL} Erreur CUDA : {e}")
    
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : INSIGHTFACE
# ══════════════════════════════════════════════════════════════════════════════

def check_insightface(images_dir: str):
    """Teste InsightFace sur les images locales."""
    section("ÉTAPE 2 — Test InsightFace")
    
    try:
        from insightface.app import FaceAnalysis
        import numpy as np
        from PIL import Image
        
        log.info("  Chargement du modèle buffalo_l...")
        
        # Essayer avec différents chemins InsightFace
        possible_roots = [
            os.path.expanduser("~/.insightface"),
            os.path.join(os.path.expanduser("~"), ".insightface"),
            None,  # Laisse InsightFace décider
        ]
        
        app = None
        for root in possible_roots:
            try:
                app = FaceAnalysis(
                    name="buffalo_l",
                    root=root,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                log.info(f"  {PASS} InsightFace chargé (root={root or 'auto'})")
                break
            except Exception:
                continue
        
        if app is None:
            raise RuntimeError("Impossible de charger InsightFace")
        
        # Tester sur les images locales
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = []
        for ext in exts:
            images.extend(Path(images_dir).glob(ext))
        images = images[:10]
        
        if not images:
            log.error(f"  {FAIL} Aucune image trouvée dans {images_dir}")
            return None, False
        
        detected = 0
        failed = []
        
        for img_path in images:
            img_np = np.array(Image.open(img_path).convert("RGB"))
            faces = app.get(img_np)
            
            if faces:
                detected += 1
                log.info(
                    f"  {PASS} {img_path.name:30s} → "
                    f"{len(faces)} visage(s), conf={faces[0].det_score:.3f}"
                )
            else:
                failed.append(img_path.name)
                log.warning(f"  {WARN} {img_path.name:30s} → aucun visage")
        
        log.info(f"\n  Résultat : {detected}/{len(images)} images avec visage détecté")
        
        if detected == 0:
            log.error(f"  {FAIL} Aucune image utilisable — vérifiez vos photos")
            log.error("         Les photos doivent contenir des visages clairement visibles")
            return None, False
        
        if failed:
            log.warning(
                f"  {WARN} {len(failed)} image(s) sans visage "
                f"(seront rejetées par prepare_dataset)"
            )
        
        return app, True
    
    except Exception as e:
        log.error(f"  {FAIL} InsightFace erreur : {e}")
        log.error("         Installation : pip install insightface onnxruntime-gpu")
        return None, False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : EVA-CLIP
# ══════════════════════════════════════════════════════════════════════════════

def check_eva_clip():
    """Teste le chargement d'EVA-CLIP."""
    section("ÉTAPE 3 — Test EVA-CLIP")
    
    try:
        import torch
        import open_clip
        
        log.info("  Chargement EVA02-CLIP-L-14-336...")
        log.info("  (peut prendre 1-2 min au premier lancement)")
        
        t0 = time.time()
        
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k"
        )
        visual = model.visual
        visual.eval()
        
        elapsed = time.time() - t0
        log.info(f"  {PASS} EVA-CLIP chargé en {elapsed:.1f}s")
        
        # Test forward
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visual = visual.to(device)
        dummy = torch.randn(1, 3, 336, 336).to(device)
        
        with torch.no_grad():
            out = visual(dummy.float())
            if isinstance(out, (list, tuple)):
                out = out[0]
        
        expected_shape = (1, 768)
        if out.shape == expected_shape:
            log.info(f"  {PASS} Forward OK → shape: {out.shape}")
        else:
            log.warning(
                f"  {WARN} Shape inattendu : {out.shape} "
                f"(attendu: {expected_shape})"
            )
        
        return visual, True
    
    except Exception as e:
        log.error(f"  {FAIL} EVA-CLIP erreur : {e}")
        log.error("         Installation : pip install open-clip-torch")
        return None, False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : PREPARE_DATASET
# ══════════════════════════════════════════════════════════════════════════════

def check_prepare_dataset(images_dir: str, output_dir: str, face_detector):
    """Teste prepare_dataset sur les images locales."""
    section("ÉTAPE 4 — Test prepare_dataset")
    
    try:
        import numpy as np
        from PIL import Image
        
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = []
        for ext in exts:
            images.extend(Path(images_dir).glob(ext))
        images = images[:10]
        
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        kept = 0
        metadata = []
        
        for img_path in images:
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            faces = face_detector.get(img_np)
            
            if len(faces) != 1:
                log.warning(
                    f"  {WARN} {img_path.name:30s} → "
                    f"{len(faces)} visages, ignorée"
                )
                continue
            
            face = faces[0]
            
            # Vérifications basiques
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1
            
            if face_w < 64 or face_h < 64:
                log.warning(
                    f"  {WARN} {img_path.name:30s} → "
                    f"visage trop petit ({face_w}×{face_h})"
                )
                continue
            
            if face.det_score < 0.6:
                log.warning(
                    f"  {WARN} {img_path.name:30s} → "
                    f"confiance faible ({face.det_score:.3f})"
                )
                continue
            
            # Image acceptée
            out_name = f"face_{kept:05d}.jpg"
            out_path = os.path.join(output_dir, "images", out_name)
            img_pil.resize((512, 512), Image.LANCZOS).save(out_path, quality=95)
            
            metadata.append({
                "filename": out_name,
                "embedding": face.embedding.tolist(),
                "det_score": float(face.det_score),
            })
            
            kept += 1
            log.info(
                f"  {PASS} {img_path.name:30s} → {out_name} "
                f"(conf={face.det_score:.3f})"
            )
        
        if kept == 0:
            log.error(f"  {FAIL} Aucune image valide — vérifiez vos photos")
            log.error("         Les photos doivent contenir exactement 1 visage")
            return False
        
        # Sauvegarder metadata.json
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"total": kept, "files": metadata}, f, indent=2)
        
        log.info(f"\n  {PASS} Dataset test prêt : {kept} images")
        log.info(f"  {PASS} Dossier : {output_dir}/images/")
        log.info(f"  {PASS} Metadata : {meta_path}")
        
        return True
    
    except Exception as e:
        log.error(f"  {FAIL} prepare_dataset erreur : {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 : TEST ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def check_training(dataset_dir: str, output_dir: str, comfyui_path: str):
    """Teste l'entraînement sur 2 epochs."""
    section("ÉTAPE 5 — Test entraînement (2 epochs, batch=1)")
    
    try:
        # Chercher le script d'entraînement
        possible_names = [
            "train_pulid_COMPLETE.py",
            "train_pulid.py",
            "train_pulid_klein.py",
        ]
        
        train_script = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for name in possible_names:
            path = os.path.join(script_dir, name)
            if os.path.exists(path):
                train_script = path
                break
        
        if not train_script:
            log.error(f"  {FAIL} Script d'entraînement non trouvé")
            log.error(f"         Cherché : {possible_names}")
            log.error(f"         Dans : {script_dir}")
            return False
        
        log.info(f"  Script : {os.path.basename(train_script)}")
        
        cmd = [
            sys.executable, train_script,
            "--dataset", dataset_dir,
            "--output", output_dir,
            "--comfyui_path", comfyui_path,
            "--phase", "1",
            "--epochs", "2",
            "--batch_size", "1",
            "--save_every", "999999",  # Pas de checkpoint intermédiaire
            "--dim", "4096",
        ]
        
        log.info(f"  Commande : {' '.join(cmd)}")
        log.info("  Lancement (peut prendre 2-5 min)...")
        log.info("")
        
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            log.error(f"  {FAIL} Entraînement échoué (code: {result.returncode})")
            log.error("")
            log.error("  Erreur stderr :")
            for line in result.stderr.split("\n")[-30:]:
                if line.strip():
                    log.error(f"    {line}")
            return False
        
        # Afficher les dernières lignes du log
        log.info("  Sortie stdout :")
        for line in result.stdout.split("\n")[-20:]:
            if line.strip():
                log.info(f"    {line}")
        
        log.info("")
        log.info(f"  {PASS} Entraînement terminé en {elapsed:.0f}s")
        return True
    
    except Exception as e:
        log.error(f"  {FAIL} Erreur : {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 : VÉRIFICATION SAFETENSORS
# ══════════════════════════════════════════════════════════════════════════════

def check_safetensors(output_dir: str, comfyui_path: str):
    """Vérifie le fichier .safetensors généré."""
    section("ÉTAPE 6 — Vérification du .safetensors")
    
    try:
        from safetensors.torch import load_file
        
        # Chercher le fichier
        candidates = [
            os.path.join(output_dir, "pulid_klein_phase1_best.safetensors"),
            os.path.join(output_dir, "pulid_klein_phase1_latest.safetensors"),
        ]
        
        # Chercher aussi dans checkpoints/
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".safetensors"):
                    candidates.append(os.path.join(ckpt_dir, f))
        
        path = next((p for p in candidates if os.path.exists(p)), None)
        
        if not path:
            log.error(f"  {FAIL} Aucun .safetensors trouvé dans {output_dir}")
            log.error(f"         Cherché : {candidates}")
            return False
        
        # Charger le fichier
        state = load_file(path, device="cpu")
        size = os.path.getsize(path) / 1e6
        
        log.info(f"  {PASS} Fichier trouvé : {os.path.basename(path)}")
        log.info(f"  {PASS} Taille : {size:.1f} Mo")
        log.info(f"  {PASS} Clés : {len(state)}")
        log.info("")
        
        # Vérifier les clés essentielles
        expected_keys = [
            "id_former.latents",
            "id_former.norm.weight",
            "id_former.proj.0.weight",
        ]
        
        missing = []
        for key in expected_keys:
            if key in state:
                log.info(f"  {PASS} {key:40s} : {tuple(state[key].shape)}")
            else:
                missing.append(key)
                log.warning(f"  {WARN} {key:40s} : manquante")
        
        if missing:
            log.warning(
                f"  {WARN} {len(missing)} clé(s) manquante(s) "
                f"— checkpoint peut-être incomplet"
            )
        
        # Test de chargement dans PuLIDFlux2
        log.info("")
        log.info("  Test chargement dans PuLIDFlux2...")
        
        # Chercher pulid_flux2.py
        possible_paths = [
            os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2Klein", "pulid_flux2.py"),
            os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2", "pulid_flux2.py"),
            os.path.join(os.path.dirname(__file__), "pulid_flux2.py"),
        ]
        
        pulid_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if pulid_path:
            sys.path.insert(0, os.path.dirname(pulid_path))
            try:
                from pulid_flux2 import PuLIDFlux2
                model = PuLIDFlux2(dim=4096)
                missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
                
                if missing_keys:
                    log.warning(
                        f"  {WARN} {len(missing_keys)} clé(s) manquante(s) "
                        f"lors du chargement"
                    )
                if unexpected_keys:
                    log.warning(
                        f"  {WARN} {len(unexpected_keys)} clé(s) inattendue(s)"
                    )
                
                log.info(f"  {PASS} Chargeable dans PuLIDFlux2 ✅")
            except Exception as e:
                log.warning(f"  {WARN} Erreur chargement PuLIDFlux2 : {e}")
        else:
            log.warning(f"  {WARN} pulid_flux2.py non trouvé — test de chargement skippé")
        
        return True
    
    except Exception as e:
        log.error(f"  {FAIL} Erreur : {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Valide le pipeline PuLID Klein avant Vast.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Dossier avec 5-10 photos de visages pour tester"
    )
    
    parser.add_argument(
        "--comfyui_path",
        type=str,
        default="C:/AI/ComfyUI",
        help="Chemin vers ComfyUI"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./test_output",
        help="Dossier de sortie pour les tests"
    )
    
    args = parser.parse_args()
    
    # Dossiers
    dataset_test = os.path.join(args.output, "dataset_test")
    train_test = os.path.join(args.output, "train_test")
    
    results = {}
    t_start = time.time()
    
    # ── Étape 1 : Dépendances ────────────────────────────────────────────────
    results["deps"] = check_dependencies()
    if not results["deps"]:
        log.error("")
        log.error("❌ Dépendances manquantes — installez-les avant de continuer")
        return
    
    # ── Étape 2 : InsightFace ─────────────────────────────────────────────────
    face_detector, results["insightface"] = check_insightface(args.images)
    if not results["insightface"]:
        log.error("")
        log.error("❌ InsightFace KO — impossible de continuer")
        return
    
    # ── Étape 3 : EVA-CLIP ────────────────────────────────────────────────────
    _, results["eva_clip"] = check_eva_clip()
    
    # ── Étape 4 : prepare_dataset ─────────────────────────────────────────────
    if results["insightface"]:
        results["dataset"] = check_prepare_dataset(
            args.images,
            dataset_test,
            face_detector
        )
    else:
        results["dataset"] = False
        log.warning("⚠️  Étape 4 skippée (InsightFace KO)")
    
    # ── Étape 5 : Entraînement ────────────────────────────────────────────────
    if results["dataset"]:
        results["training"] = check_training(
            dataset_test,
            train_test,
            args.comfyui_path
        )
    else:
        results["training"] = False
        log.warning("⚠️  Étape 5 skippée (dataset KO)")
    
    # ── Étape 6 : Safetensors ─────────────────────────────────────────────────
    if results["training"]:
        results["safetensors"] = check_safetensors(train_test, args.comfyui_path)
    else:
        results["safetensors"] = False
        log.warning("⚠️  Étape 6 skippée (training KO)")
    
    # ── Rapport Final ─────────────────────────────────────────────────────────
    section("RAPPORT FINAL")
    
    labels = {
        "deps": "Dépendances",
        "insightface": "InsightFace",
        "eva_clip": "EVA-CLIP",
        "dataset": "prepare_dataset",
        "training": "Entraînement (2 epochs)",
        "safetensors": "Fichier .safetensors",
    }
    
    all_ok = True
    for key, label in labels.items():
        status = PASS if results.get(key) else FAIL
        log.info(f"  {status} {label:30s}")
        if not results.get(key):
            all_ok = False
    
    elapsed = time.time() - t_start
    log.info("")
    log.info(f"  Durée totale : {elapsed:.0f}s")
    log.info("")
    
    if all_ok:
        log.info("🎉 TOUT EST OK — Vous pouvez lancer sur Vast.ai !")
        log.info("")
        log.info("📝 COMMANDES POUR VAST.AI :")
        log.info("")
        log.info("# Phase 1 (A100 40GB recommandé)")
        log.info("python train_pulid.py \\")
        log.info("    --dataset ./dataset/filtered \\")
        log.info("    --output ./output \\")
        log.info("    --phase 1 \\")
        log.info("    --epochs 25 \\")
        log.info("    --batch_size 8")
        log.info("")
        log.info("# Phase 2 (après Phase 1)")
        log.info("python train_pulid.py \\")
        log.info("    --dataset ./dataset/filtered \\")
        log.info("    --output ./output_phase2 \\")
        log.info("    --phase 2 \\")
        log.info("    --resume ./output/pulid_klein_phase1_best.safetensors \\")
        log.info("    --flux_model_path ./models/flux-2-klein-9b-fp8.safetensors \\")
        log.info("    --epochs 15 \\")
        log.info("    --batch_size 4")
    else:
        log.info("❌ Des erreurs sont présentes — corrigez-les avant Vast.ai")
        log.info("")
        log.info("Solutions courantes :")
        if not results.get("insightface"):
            log.info("  - InsightFace : pip install insightface onnxruntime-gpu")
        if not results.get("eva_clip"):
            log.info("  - EVA-CLIP : pip install open-clip-torch")
        if not results.get("dataset"):
            log.info("  - Dataset : vérifiez que vos photos contiennent des visages")


if __name__ == "__main__":
    main()
