"""
prepare_dataset.py - VERSION AMÉLIORÉE
========================================
Prépare un dataset de visages pour entraîner PuLID sur Flux.2 Klein.

AMÉLIORATIONS vs version originale :
  ✅ Support multi-sources (CelebA, FFHQ, vos propres photos)
  ✅ Filtrage qualité amélioré (taille visage, confiance, pose)
  ✅ Déduplication des identités similaires
  ✅ Statistiques détaillées du dataset
  ✅ Mode test avec 10 images
  ✅ Vérification qualité automatique

Ce script :
  1. Télécharge un dataset public OU utilise vos photos locales
  2. Filtre les images avec InsightFace (1 visage bien visible)
  3. Déduplique les identités trop similaires
  4. Crop et normalise les images
  5. Génère metadata.json avec embeddings
  6. Sauvegarde le dataset prêt pour train_pulid.py

Usage :
  # Avec dataset public (CelebA)
  python prepare_dataset.py --output ./dataset --max_images 2000
  
  # Avec vos propres photos
  python prepare_dataset.py --input_dir ./mes_photos --output ./dataset
  
  # Mode test rapide
  python prepare_dataset.py --input_dir ./mes_photos --output ./test --test_mode
"""

import os
import argparse
import logging
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT DATASETS PUBLICS
# ══════════════════════════════════════════════════════════════════════════════

def download_celeba(output_dir: str, max_images: int = 2000) -> int:
    """
    Télécharge CelebA dataset (visages de célébrités).
    Dataset public, ~200k images de haute qualité.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets non installé.\n"
            "Installation : pip install datasets"
        )
    
    log.info(f"Téléchargement CelebA ({max_images} images max)...")
    
    try:
        ds = load_dataset(
            "huggan/celeba-faces",
            split="train",
            streaming=True
        )
    except Exception as e:
        log.error(f"Erreur téléchargement CelebA : {e}")
        log.error("Alternative : utiliser --input_dir avec vos photos")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    
    for i, item in enumerate(tqdm(ds, total=max_images, desc="CelebA")):
        if saved >= max_images:
            break
        
        try:
            img = item["image"]
            img_path = os.path.join(output_dir, f"celeba_{i:05d}.jpg")
            img.save(img_path, quality=95)
            saved += 1
        except Exception as e:
            log.warning(f"Erreur image {i} : {e}")
            continue
    
    log.info(f"✅ {saved} images téléchargées dans {output_dir}")
    return saved


def download_ffhq(output_dir: str, max_images: int = 2000) -> int:
    """
    Télécharge FFHQ thumbnails (128x128, redimensionné à 512x512).
    Dataset public, 70k visages de haute qualité.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    
    log.info(f"Téléchargement FFHQ ({max_images} images max)...")
    
    try:
        ds = load_dataset(
            "asomoza/ffhq-thumbnails",
            split="train",
            streaming=True
        )
    except Exception as e:
        log.error(f"Erreur téléchargement FFHQ : {e}")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    
    for i, item in enumerate(tqdm(ds, total=max_images, desc="FFHQ")):
        if saved >= max_images:
            break
        
        try:
            img = item["image"]
            # FFHQ thumbnails sont 128x128, on les resize
            img = img.resize((512, 512), Image.LANCZOS)
            img_path = os.path.join(output_dir, f"ffhq_{i:05d}.jpg")
            img.save(img_path, quality=95)
            saved += 1
        except Exception:
            continue
    
    log.info(f"✅ {saved} images téléchargées dans {output_dir}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTFACE
# ══════════════════════════════════════════════════════════════════════════════

def load_face_detector():
    """Charge InsightFace AntelopeV2 (buffalo_l)."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "insightface non installé.\n"
            "Installation : pip install insightface onnxruntime-gpu"
        )
    
    log.info("Chargement InsightFace AntelopeV2...")
    
    # Buffalo_l est le modèle le plus précis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    log.info("✅ InsightFace chargé")
    return app


# ══════════════════════════════════════════════════════════════════════════════
# FILTRAGE ET TRAITEMENT
# ══════════════════════════════════════════════════════════════════════════════

def calculate_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calcule la similarité cosinus entre deux embeddings."""
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


def is_duplicate_identity(
    new_embedding: np.ndarray,
    existing_embeddings: List[np.ndarray],
    threshold: float = 0.6
) -> bool:
    """
    Vérifie si une identité est déjà présente dans le dataset.
    
    Args:
        new_embedding: Embedding du nouveau visage
        existing_embeddings: Liste des embeddings déjà dans le dataset
        threshold: Seuil de similarité (0.6 = même personne probable)
    
    Returns:
        True si duplicate, False sinon
    """
    for existing in existing_embeddings:
        similarity = calculate_embedding_similarity(new_embedding, existing)
        if similarity > threshold:
            return True
    return False


def filter_and_process_images(
    input_dir: str,
    output_dir: str,
    face_detector,
    min_face_size: int = 80,
    target_size: int = 512,
    min_confidence: float = 0.7,
    deduplicate: bool = True,
    dedup_threshold: float = 0.6,
    max_images: Optional[int] = None,
) -> int:
    """
    Filtre et traite les images pour créer un dataset de qualité.
    
    Critères de filtrage :
    - Exactement 1 visage détecté
    - Visage assez grand (min_face_size px)
    - Confiance détection suffisante (min_confidence)
    - Pas de duplicate d'identité (si deduplicate=True)
    - Pose raisonnablement frontale (angle < 45°)
    
    Args:
        input_dir: Dossier source avec images
        output_dir: Dossier de sortie
        face_detector: Instance InsightFace
        min_face_size: Taille min du visage en pixels
        target_size: Taille finale des images
        min_confidence: Score de confiance min (0-1)
        deduplicate: Activer la déduplication
        dedup_threshold: Seuil de similarité pour déduplication
        max_images: Limite du nombre d'images à traiter
    
    Returns:
        Nombre d'images gardées
    """
    # Créer les dossiers
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rejected"), exist_ok=True)
    
    # Lister les images
    input_path = Path(input_dir)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(input_path.glob(ext))
    
    if max_images:
        image_files = image_files[:max_images]
    
    if not image_files:
        log.error(f"Aucune image trouvée dans {input_dir}")
        return 0
    
    log.info(f"Traitement de {len(image_files)} images...")
    
    # Statistiques
    kept = 0
    rejected = {
        "no_face": 0,
        "multiple_faces": 0,
        "too_small": 0,
        "low_confidence": 0,
        "bad_pose": 0,
        "duplicate": 0,
    }
    
    embeddings_data = []
    existing_embeddings = []
    
    for img_path in tqdm(image_files, desc="Filtrage"):
        try:
            # Charger image
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            
            # Détecter visages
            faces = face_detector.get(img_np)
            
            # Filtre 1 : Exactement 1 visage
            if len(faces) == 0:
                rejected["no_face"] += 1
                shutil.copy(
                    img_path,
                    os.path.join(output_dir, "rejected", f"noface_{img_path.name}")
                )
                continue
            
            if len(faces) > 1:
                rejected["multiple_faces"] += 1
                shutil.copy(
                    img_path,
                    os.path.join(output_dir, "rejected", f"multi_{img_path.name}")
                )
                continue
            
            face = faces[0]
            
            # Filtre 2 : Taille du visage
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1
            
            if face_w < min_face_size or face_h < min_face_size:
                rejected["too_small"] += 1
                continue
            
            # Filtre 3 : Confiance de détection
            if face.det_score < min_confidence:
                rejected["low_confidence"] += 1
                continue
            
            # Filtre 4 : Pose (angle du visage)
            # Si pose existe, vérifier qu'elle n'est pas trop extrême
            if hasattr(face, 'pose') and face.pose is not None:
                # pose = [pitch, yaw, roll]
                pitch, yaw, roll = face.pose
                max_angle = 45  # degrés
                
                if abs(pitch) > max_angle or abs(yaw) > max_angle:
                    rejected["bad_pose"] += 1
                    continue
            
            # Filtre 5 : Déduplication (optionnel)
            if deduplicate:
                if is_duplicate_identity(
                    face.embedding,
                    existing_embeddings,
                    threshold=dedup_threshold
                ):
                    rejected["duplicate"] += 1
                    continue
            
            # Image acceptée !
            
            # Resize (on garde l'image complète, pas juste le visage)
            img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
            
            # Sauvegarder
            out_name = f"face_{kept:05d}.jpg"
            out_path = os.path.join(output_dir, "images", out_name)
            img_resized.save(out_path, quality=95)
            
            # Ajouter aux métadonnées
            embeddings_data.append({
                "filename": out_name,
                "original": img_path.name,
                "embedding": face.embedding.tolist(),
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score),
                "face_size": [int(face_w), int(face_h)],
            })
            
            # Ajouter à la liste des embeddings existants
            if deduplicate:
                existing_embeddings.append(face.embedding)
            
            kept += 1
        
        except Exception as e:
            log.warning(f"Erreur sur {img_path.name}: {e}")
            continue
    
    # Sauvegarder métadonnées
    total_rejected = sum(rejected.values())
    
    metadata = {
        "total_kept": kept,
        "total_rejected": total_rejected,
        "rejection_reasons": rejected,
        "target_size": target_size,
        "min_face_size": min_face_size,
        "min_confidence": min_confidence,
        "deduplicated": deduplicate,
        "files": embeddings_data,
    }
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Afficher statistiques
    log.info("")
    log.info("=" * 60)
    log.info("  STATISTIQUES DATASET")
    log.info("=" * 60)
    log.info(f"  Images gardées    : {kept}")
    log.info(f"  Images rejetées   : {total_rejected}")
    log.info("")
    log.info("  Raisons de rejet :")
    for reason, count in rejected.items():
        if count > 0:
            log.info(f"    - {reason:20s} : {count}")
    log.info("")
    log.info(f"  Dossier images    : {output_dir}/images/")
    log.info(f"  Métadonnées       : {meta_path}")
    log.info("=" * 60)
    
    # Évaluation qualité
    if kept > 0:
        log.info("")
        evaluate_dataset_quality(kept, total_rejected, metadata)
    
    return kept


def evaluate_dataset_quality(kept: int, rejected: int, metadata: dict):
    """Évalue et affiche la qualité du dataset."""
    log.info("📊 ÉVALUATION QUALITÉ")
    log.info("")
    
    # Taux de rejet
    total = kept + rejected
    reject_rate = (rejected / total * 100) if total > 0 else 0
    
    # Taille du dataset
    if kept < 200:
        size_quality = "❌ INSUFFISANT"
        size_note = "Recommandé : >500 images, idéal : 2000+"
    elif kept < 500:
        size_quality = "⚠️  MINIMAL"
        size_note = "Acceptable mais limité, idéal : 2000+"
    elif kept < 1000:
        size_quality = "✅ BON"
        size_note = "Suffisant pour Phase 1, plus = mieux"
    else:
        size_quality = "✅ EXCELLENT"
        size_note = "Taille idéale pour entraînement"
    
    # Qualité des détections
    avg_conf = np.mean([f["det_score"] for f in metadata["files"]])
    if avg_conf > 0.95:
        conf_quality = "✅ EXCELLENT"
    elif avg_conf > 0.85:
        conf_quality = "✅ BON"
    else:
        conf_quality = "⚠️  MOYEN"
    
    log.info(f"  Taille dataset    : {size_quality} ({kept} images)")
    log.info(f"                      {size_note}")
    log.info(f"  Taux de rejet     : {reject_rate:.1f}%")
    log.info(f"  Confiance moyenne : {conf_quality} ({avg_conf:.3f})")
    log.info("")
    
    # Recommandations
    if kept < 500:
        log.warning("⚠️  RECOMMANDATION : Augmenter le dataset")
        log.warning("   - Utiliser --max_images 5000 avec dataset public")
        log.warning("   - Ajouter plus de photos personnelles")
        log.warning("   - Télécharger dataset FFHQ ou CelebA complet")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prépare un dataset de visages pour PuLID Klein",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Source
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Utiliser vos propres images (dossier local)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="celeba",
        choices=["celeba", "ffhq"],
        help="Dataset public à télécharger (si --input_dir non spécifié)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=2000,
        help="Nombre max d'images à télécharger (dataset public)"
    )
    
    # Sortie
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset",
        help="Dossier de sortie du dataset"
    )
    
    # Qualité
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Taille finale des images"
    )
    
    parser.add_argument(
        "--min_face",
        type=int,
        default=80,
        help="Taille minimale du visage en pixels"
    )
    
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.7,
        help="Score de confiance minimum (0-1)"
    )
    
    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="Désactiver la déduplication des identités"
    )
    
    parser.add_argument(
        "--dedup_threshold",
        type=float,
        default=0.6,
        help="Seuil de similarité pour déduplication (0.6 = même personne)"
    )
    
    # Mode test
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Mode test : traite seulement 10 images"
    )
    
    args = parser.parse_args()
    
    # Dossiers
    raw_dir = os.path.join(args.output, "raw")
    filtered_dir = os.path.join(args.output, "filtered")
    
    # ── ÉTAPE 1 : Source des images ─────────────────────────────────────────
    
    if args.input_dir:
        log.info(f"📁 Source : images locales depuis {args.input_dir}")
        raw_dir = args.input_dir
    else:
        log.info(f"📁 Source : dataset public {args.source}")
        
        if args.source == "celeba":
            n = download_celeba(raw_dir, args.max_images)
        else:
            n = download_ffhq(raw_dir, args.max_images)
        
        if n == 0:
            log.error("❌ Téléchargement échoué")
            log.error("   Alternative : --input_dir avec vos propres photos")
            return
    
    # ── ÉTAPE 2 : Chargement InsightFace ────────────────────────────────────
    
    try:
        detector = load_face_detector()
    except Exception as e:
        log.error(f"❌ Erreur chargement InsightFace : {e}")
        return
    
    # ── ÉTAPE 3 : Filtrage et traitement ────────────────────────────────────
    
    max_to_process = 10 if args.test_mode else None
    
    if args.test_mode:
        log.info("🧪 MODE TEST : traitement de 10 images seulement")
    
    n_kept = filter_and_process_images(
        input_dir=raw_dir,
        output_dir=filtered_dir,
        face_detector=detector,
        min_face_size=args.min_face,
        target_size=args.target_size,
        min_confidence=args.min_confidence,
        deduplicate=not args.no_deduplicate,
        dedup_threshold=args.dedup_threshold,
        max_images=max_to_process,
    )
    
    # ── RAPPORT FINAL ────────────────────────────────────────────────────────
    
    if n_kept == 0:
        log.error("")
        log.error("❌ ÉCHEC : Aucune image valide dans le dataset")
        log.error("")
        log.error("Solutions possibles :")
        log.error("  1. Vérifier que les images contiennent des visages")
        log.error("  2. Réduire --min_confidence (ex: 0.5)")
        log.error("  3. Réduire --min_face (ex: 50)")
        log.error("  4. Essayer une autre source (--source ffhq)")
        return
    
    log.info("")
    log.info("🎉 DATASET PRÊT !")
    log.info("")
    log.info(f"   Chemin   : {filtered_dir}")
    log.info(f"   Images   : {n_kept}")
    log.info(f"   Metadata : {filtered_dir}/metadata.json")
    log.info("")
    log.info("📝 PROCHAINE ÉTAPE :")
    log.info("")
    log.info("   # Phase 1 (IDFormer)")
    log.info(f"   python train_pulid.py \\")
    log.info(f"       --dataset {filtered_dir} \\")
    log.info(f"       --output ./output \\")
    log.info(f"       --phase 1")
    log.info("")


if __name__ == "__main__":
    main()
