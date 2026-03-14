"""
prepare_dataset.py
==================
Prépare un dataset de visages pour entraîner PuLID sur Flux.2 Klein.

Ce script :
  1. Télécharge un dataset public de visages (FFHQ-thumbnails ou CelebA)
  2. Filtre les images avec InsightFace (garde uniquement 1 visage bien visible)
  3. Crop et normalise les images
  4. Sauvegarde le dataset prêt à l'emploi

Usage :
  python prepare_dataset.py --output ./dataset --max_images 2000
"""

import os
import argparse
import logging
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Téléchargement dataset public
# ──────────────────────────────────────────────────────────────────────────────

def download_ffhq_thumbnails(output_dir: str, max_images: int = 2000):
    """
    Télécharge FFHQ-thumbnails (128x128) depuis HuggingFace.
    Dataset public, 70 000 visages de haute qualité.
    On prend les max_images premiers.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    log.info(f"Téléchargement FFHQ thumbnails ({max_images} images max)...")
    
    # Télécharger via HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "mattmdjaga/human_parsing_dataset",
            split="train",
            streaming=True
        )
    except Exception:
        # Fallback : dataset plus simple
        from datasets import load_dataset
        ds = load_dataset(
            "asomoza/ffhq-thumbnails",
            split="train",
            streaming=True
        )

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for i, item in enumerate(tqdm(ds, total=max_images, desc="Download")):
        if saved >= max_images:
            break
        try:
            img = item["image"] if "image" in item else item["img"]
            if isinstance(img, dict):
                from PIL import Image as PILImage
                import io, base64
                img = PILImage.open(io.BytesIO(base64.b64decode(img["bytes"])))
            img_path = os.path.join(output_dir, f"face_{i:05d}.jpg")
            img.save(img_path, quality=95)
            saved += 1
        except Exception as e:
            continue

    log.info(f"✅ {saved} images téléchargées dans {output_dir}")
    return saved


def download_celeba(output_dir: str, max_images: int = 2000):
    """Alternative : CelebA dataset (visages de célébrités)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    log.info(f"Téléchargement CelebA ({max_images} images)...")
    ds = load_dataset("huggan/celeba-faces", split="train", streaming=True)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for i, item in enumerate(tqdm(ds, total=max_images, desc="CelebA")):
        if saved >= max_images:
            break
        try:
            img = item["image"]
            img_path = os.path.join(output_dir, f"face_{i:05d}.jpg")
            img.save(img_path, quality=95)
            saved += 1
        except Exception:
            continue

    log.info(f"✅ {saved} images téléchargées dans {output_dir}")
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Filtrage InsightFace
# ──────────────────────────────────────────────────────────────────────────────

def load_face_detector():
    """Charge InsightFace AntelopeV2."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="antelopev2",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def filter_and_process_images(
    input_dir: str,
    output_dir: str,
    face_detector,
    min_face_size: int = 64,
    target_size: int = 512,
    min_confidence: float = 0.6,
):
    """
    Filtre les images :
    - Garde uniquement celles avec exactement 1 visage
    - Visage suffisamment grand (min_face_size px)
    - Score de confiance suffisant
    - Resize à target_size
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rejected"), exist_ok=True)

    input_files = list(Path(input_dir).glob("*.jpg")) + \
                  list(Path(input_dir).glob("*.png")) + \
                  list(Path(input_dir).glob("*.jpeg"))

    log.info(f"Filtrage de {len(input_files)} images...")

    kept = 0
    rejected = 0
    embeddings_data = []

    for img_path in tqdm(input_files, desc="Filtrage"):
        try:
            # Charger image
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)

            # Détecter visages
            faces = face_detector.get(img_np)

            # Filtres
            if len(faces) != 1:
                shutil.copy(img_path, os.path.join(output_dir, "rejected", img_path.name))
                rejected += 1
                continue

            face = faces[0]

            # Vérifier taille du visage
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < min_face_size or face_h < min_face_size:
                rejected += 1
                continue

            # Vérifier confiance détection
            if face.det_score < min_confidence:
                rejected += 1
                continue

            # Resize image entière (pas crop — PuLID veut le contexte complet)
            img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)

            # Sauvegarder
            out_name = f"face_{kept:05d}.jpg"
            out_path = os.path.join(output_dir, "images", out_name)
            img_resized.save(out_path, quality=95)

            # Sauvegarder embedding InsightFace pour vérification
            embeddings_data.append({
                "filename": out_name,
                "embedding": face.embedding.tolist(),
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score),
            })

            kept += 1

        except Exception as e:
            log.warning(f"Erreur sur {img_path.name}: {e}")
            rejected += 1
            continue

    # Sauvegarder métadonnées
    import json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "total": kept,
            "rejected": rejected,
            "target_size": target_size,
            "files": embeddings_data
        }, f, indent=2)

    log.info(f"✅ Dataset prêt : {kept} images gardées, {rejected} rejetées")
    log.info(f"   Dossier : {output_dir}/images/")
    log.info(f"   Métadonnées : {meta_path}")
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prépare un dataset pour PuLID Klein")
    parser.add_argument("--output",      type=str, default="./dataset",
                        help="Dossier de sortie du dataset")
    parser.add_argument("--max_images",  type=int, default=2000,
                        help="Nombre max d'images à télécharger")
    parser.add_argument("--source",      type=str, default="celeba",
                        choices=["celeba", "ffhq"],
                        help="Source du dataset")
    parser.add_argument("--input_dir",   type=str, default=None,
                        help="Utiliser vos propres images (dossier local)")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Taille cible des images (512 recommandé)")
    parser.add_argument("--min_face",    type=int, default=64,
                        help="Taille minimale du visage en pixels")
    args = parser.parse_args()

    raw_dir      = os.path.join(args.output, "raw")
    filtered_dir = os.path.join(args.output, "filtered")

    # Étape 1 : Téléchargement ou utilisation d'images locales
    if args.input_dir:
        log.info(f"Utilisation des images locales depuis : {args.input_dir}")
        raw_dir = args.input_dir
    else:
        log.info(f"Source : {args.source}")
        if args.source == "celeba":
            download_celeba(raw_dir, args.max_images)
        else:
            download_ffhq_thumbnails(raw_dir, args.max_images)

    # Étape 2 : Chargement du détecteur de visages
    log.info("Chargement InsightFace...")
    detector = load_face_detector()

    # Étape 3 : Filtrage et traitement
    n = filter_and_process_images(
        input_dir=raw_dir,
        output_dir=filtered_dir,
        face_detector=detector,
        min_face_size=args.min_face,
        target_size=args.target_size,
    )

    if n < 500:
        log.warning(
            f"⚠️  Seulement {n} images — recommandé : 500 minimum. "
            "Augmentez --max_images ou ajoutez vos propres images."
        )
    else:
        log.info(f"🎉 Dataset prêt avec {n} images → {filtered_dir}/images/")
        log.info("Prochaine étape : python train_pulid_klein.py --dataset ./dataset/filtered")


if __name__ == "__main__":
    main()
