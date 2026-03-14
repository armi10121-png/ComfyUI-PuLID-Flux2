"""
train_pulid_klein.py
====================
Entraîne le module PuLID (IDFormer + PerceiverCA) sur Flux.2 Klein.

Ce script est le PREMIER script d'entraînement PuLID dédié à Flux.2 Klein.
Auteur original du custom node : toi 😄
Date : Mars 2026

Architecture entraînée :
  - IDFormer : projette InsightFace (512d) + EVA-CLIP (768d) → tokens [B, 4, dim]
  - perceiverCA double_blocks : injecté dans les double transformer blocks
  - perceiverCA single_blocks : injecté dans les single transformer blocks
  
Le modèle Flux.2 Klein est GELÉ — seul PuLID est entraîné.

Usage :
  python train_pulid_klein.py --dataset ./dataset/filtered --output ./output

Prérequis :
  pip install accelerate diffusers transformers datasets
  pip install open_clip_torch insightface
"""

import os
import argparse
import logging
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Dataset de visages pour entraîner PuLID.
    Chaque item = (image_tensor, id_embedding_insightface)
    """
    def __init__(self, dataset_dir: str, size: int = 512):
        self.size = size
        images_dir = os.path.join(dataset_dir, "images")
        meta_path   = os.path.join(dataset_dir, "metadata.json")

        # Charger les métadonnées (embeddings pré-calculés)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.items = []
        for item in meta["files"]:
            img_path = os.path.join(images_dir, item["filename"])
            if os.path.exists(img_path):
                self.items.append({
                    "path"     : img_path,
                    "embedding": np.array(item["embedding"], dtype=np.float32),
                })

        log.info(f"Dataset chargé : {len(self.items)} images")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Charger image
        img = Image.open(item["path"]).convert("RGB")

        # Data augmentation légère
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize + normalize [-1, 1] pour le VAE
        img = img.resize((self.size, self.size), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC → CHW

        # InsightFace embedding normalisé
        id_embed = torch.from_numpy(item["embedding"])
        id_embed = F.normalize(id_embed, dim=0)

        return img_tensor, id_embed


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ──────────────────────────────────────────────────────────────────────────────

def load_flux2_klein(model_path: str, device: torch.device):
    """
    Charge Flux.2 Klein via ComfyUI ou diffusers.
    Le modèle est complètement gelé.
    """
    log.info(f"Chargement Flux.2 Klein depuis : {model_path}")

    try:
        # Essayer via diffusers (si disponible)
        from diffusers import FluxTransformer2DModel
        model = FluxTransformer2DModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        # Charger safetensors directement
        from safetensors.torch import load_file
        state = load_file(model_path, device="cpu")
        log.info(f"Modèle chargé ({len(state)} clés)")
        # On ne peut pas instancier sans l'architecture
        # → utiliser comfy si disponible
        raise RuntimeError(
            "Flux.2 Klein doit être chargé via ComfyUI ou diffusers.\n"
            "Installez diffusers : pip install diffusers>=0.31.0\n"
            "Puis téléchargez depuis HuggingFace : black-forest-labs/FLUX.2-klein-9b"
        )

    # Geler complètement le modèle
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    log.info("✅ Flux.2 Klein chargé et gelé")
    return model.to(device)


def load_eva_clip(device: torch.device):
    """Charge EVA02-CLIP-L-14-336 via open_clip."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k"
    )
    model = model.visual
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    log.info("✅ EVA02-CLIP chargé et gelé")
    return model.to(device)


def load_vae(vae_path: str, device: torch.device):
    """Charge le VAE de Flux.2."""
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        for param in vae.parameters():
            param.requires_grad_(False)
        vae.eval()
        log.info("✅ VAE chargé et gelé")
        return vae.to(device)
    except Exception as e:
        log.warning(f"VAE non disponible: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Import PuLID module depuis notre custom node
# ──────────────────────────────────────────────────────────────────────────────

def load_pulid_module(comfyui_path: str, dim: int = 4096, device: torch.device = None):
    """
    Importe PuLIDFlux2Klein depuis notre custom node ComfyUI.
    """
    import sys
    pulid_path = os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2Klein")
    if pulid_path not in sys.path:
        sys.path.insert(0, pulid_path)

    from pulid_flux2klein import PuLIDFlux2Klein
    module = PuLIDFlux2Klein(dim=dim)
    log.info(f"✅ PuLIDFlux2Klein instancié (dim={dim})")

    # Compter les paramètres entraînables
    n_params = sum(p.numel() for p in module.parameters())
    log.info(f"   Paramètres entraînables : {n_params:,} ({n_params/1e6:.1f}M)")

    return module.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Loss : identité faciale
# ──────────────────────────────────────────────────────────────────────────────

class FaceIdentityLoss(nn.Module):
    """
    Loss principale : cosine similarity entre l'embedding InsightFace
    de l'image générée et celui de l'image de référence.
    
    On veut maximiser la similarité → minimiser (1 - cosine_sim)
    """
    def __init__(self, face_detector):
        super().__init__()
        self.detector = face_detector

    def get_embedding(self, images_np: np.ndarray) -> torch.Tensor:
        """Extrait les embeddings InsightFace d'un batch d'images numpy."""
        embeddings = []
        for img in images_np:
            faces = self.detector.get(img)
            if faces:
                embeddings.append(torch.from_numpy(faces[0].embedding))
            else:
                embeddings.append(torch.zeros(512))
        return torch.stack(embeddings)

    def forward(self, generated_embeds: torch.Tensor,
                target_embeds: torch.Tensor) -> torch.Tensor:
        gen_norm  = F.normalize(generated_embeds, dim=-1)
        tgt_norm  = F.normalize(target_embeds, dim=-1)
        cosine    = (gen_norm * tgt_norm).sum(dim=-1)
        return (1.0 - cosine).mean()


class FeatureMatchingLoss(nn.Module):
    """
    Loss secondaire : encourage les features intermédiaires à être cohérentes.
    Similaire à perceptual loss mais sur les tokens du transformer.
    """
    def forward(self, pred_tokens: torch.Tensor,
                target_tokens: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_tokens, target_tokens.detach())


# ──────────────────────────────────────────────────────────────────────────────
# Entraînement
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDKleinTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device : {self.device}")

        # Créer les dossiers de sortie
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, "checkpoints"), exist_ok=True)

        # Charger les modèles
        self._load_models()

        # Dataset et DataLoader
        dataset = FaceDataset(args.dataset, size=args.image_size)
        self.dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # Optimiseur — seulement les paramètres PuLID
        self.optimizer = AdamW(
            self.pulid.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        total_steps = len(self.dataloader) * args.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=args.lr * 0.1,
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        log.info(f"Steps par epoch : {len(self.dataloader)}")
        log.info(f"Total steps : {total_steps}")

    def _load_models(self):
        args = self.args

        # 1. PuLID module (seul à être entraîné)
        self.pulid = load_pulid_module(
            comfyui_path=args.comfyui_path,
            dim=args.dim,
            device=self.device,
        )
        self.pulid.train()

        # 2. EVA-CLIP (gelé)
        self.eva_clip = load_eva_clip(self.device)

        # 3. InsightFace pour la loss
        from insightface.app import FaceAnalysis
        self.face_app = FaceAnalysis(
            name="antelopev2",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 4. Flux.2 Klein (gelé) — optionnel pour phase 1
        if args.flux_model_path and os.path.exists(args.flux_model_path):
            self.flux = load_flux2_klein(args.flux_model_path, self.device)
        else:
            self.flux = None
            log.warning(
                "⚠️  Flux.2 Klein non chargé. "
                "Entraînement en mode 'embedding only' (phase 1)."
            )

    def get_eva_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        images : [B, C, H, W] float32 [-1, 1]
        returns: [B, 768]
        """
        # Renormaliser pour EVA-CLIP [0, 1] puis normalisation CLIP
        imgs = (images + 1.0) / 2.0
        imgs = F.interpolate(imgs, size=(336, 336), mode="bilinear",
                             align_corners=False)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=self.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std

        with torch.no_grad():
            features = self.eva_clip(imgs.float())
            if isinstance(features, (list, tuple)):
                features = features[0]
        return features.to(torch.bfloat16)

    def train_step_embedding_only(self, images: torch.Tensor,
                                  id_embeds: torch.Tensor) -> dict:
        """
        Phase 1 : Entraîner uniquement l'IDFormer.
        Loss = reconstruction de l'embedding à partir des features EVA-CLIP.
        Pas besoin de Flux pour cette phase.
        """
        images   = images.to(self.device, dtype=torch.bfloat16)
        id_embeds = id_embeds.to(self.device, dtype=torch.bfloat16)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Extraire features EVA-CLIP
            clip_features = self.get_eva_clip_features(images)

            # Passer par IDFormer
            id_tokens = self.pulid.id_former(id_embeds, clip_features)
            # id_tokens : [B, 4, dim]

            # Loss 1 : les tokens doivent encoder l'identité
            # On projette les tokens vers 512d et on compare avec id_embed
            proj = nn.Linear(self.args.dim, 512, bias=False).to(
                self.device, dtype=torch.bfloat16
            )
            projected = proj(id_tokens.mean(dim=1))  # [B, 512]
            projected_norm = F.normalize(projected, dim=-1)
            target_norm    = F.normalize(id_embeds, dim=-1)
            loss_identity  = (1.0 - (projected_norm * target_norm).sum(-1)).mean()

            # Loss 2 : régularisation sur les tokens (éviter explosion)
            loss_reg = id_tokens.pow(2).mean() * 0.01

            loss = loss_identity + loss_reg

        return {
            "loss"          : loss,
            "loss_identity" : loss_identity.item(),
            "loss_reg"      : loss_reg.item(),
        }

    def save_checkpoint(self, epoch: int, step: int, loss: float):
        """Sauvegarde les poids PuLID en .safetensors."""
        from safetensors.torch import save_file

        ckpt_dir  = os.path.join(self.args.output, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, f"pulid_klein_epoch{epoch:03d}_step{step:06d}.safetensors")
        latest    = os.path.join(self.args.output, "pulid_klein_latest.safetensors")

        state = {k: v.cpu().contiguous() for k, v in self.pulid.state_dict().items()}
        save_file(state, ckpt_path)
        save_file(state, latest)

        log.info(f"💾 Checkpoint sauvegardé : {ckpt_path}")
        log.info(f"   Latest : {latest}")

    def train(self):
        log.info("=" * 60)
        log.info("  Début de l'entraînement PuLID Flux.2 Klein")
        log.info(f"  Epochs        : {self.args.epochs}")
        log.info(f"  Batch size    : {self.args.batch_size}")
        log.info(f"  Learning rate : {self.args.lr}")
        log.info(f"  dim           : {self.args.dim}")
        log.info("=" * 60)

        global_step = 0
        best_loss   = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = 0.0
            self.pulid.train()

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.args.epochs}")

            for batch_idx, (images, id_embeds) in enumerate(pbar):
                self.optimizer.zero_grad()

                # Train step
                losses = self.train_step_embedding_only(images, id_embeds)
                loss   = losses["loss"]

                # Backward avec mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pulid.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                epoch_loss  += loss.item()
                global_step += 1

                # Log
                pbar.set_postfix({
                    "loss"    : f"{loss.item():.4f}",
                    "id_loss" : f"{losses['loss_identity']:.4f}",
                    "lr"      : f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

                # Sauvegarde intermédiaire
                if global_step % self.args.save_every == 0:
                    self.save_checkpoint(epoch, global_step, loss.item())

            avg_loss = epoch_loss / len(self.dataloader)
            log.info(f"Epoch {epoch} terminée — loss moyenne : {avg_loss:.4f}")

            # Sauvegarde de fin d'epoch
            self.save_checkpoint(epoch, global_step, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                from safetensors.torch import save_file
                best_path = os.path.join(self.args.output, "pulid_klein_best.safetensors")
                state = {k: v.cpu().contiguous() for k, v in self.pulid.state_dict().items()}
                save_file(state, best_path)
                log.info(f"🏆 Nouveau meilleur modèle : loss={best_loss:.4f} → {best_path}")

        log.info("=" * 60)
        log.info("✅ Entraînement terminé !")
        log.info(f"   Meilleur modèle : {self.args.output}/pulid_klein_best.safetensors")
        log.info(f"   Latest          : {self.args.output}/pulid_klein_latest.safetensors")
        log.info("")
        log.info("Pour utiliser le modèle dans ComfyUI :")
        log.info(f"  Copier pulid_klein_best.safetensors → ComfyUI/models/pulid/")
        log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne PuLID pour Flux.2 Klein",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Chemins
    parser.add_argument("--dataset",         type=str, required=True,
                        help="Dossier du dataset (contient images/ et metadata.json)")
    parser.add_argument("--output",          type=str, default="./output",
                        help="Dossier de sortie des poids")
    parser.add_argument("--comfyui_path",    type=str, default="C:/AI/ComfyUI",
                        help="Chemin vers ComfyUI (pour importer PuLIDFlux2Klein)")
    parser.add_argument("--flux_model_path", type=str, default=None,
                        help="Chemin vers flux-2-klein-9b-fp8.safetensors (optionnel phase 1)")

    # Architecture
    parser.add_argument("--dim",  type=int, default=4096,
                        choices=[3072, 4096],
                        help="Dimension cachée (3072=Klein4B, 4096=Klein9B)")

    # Hyperparamètres
    parser.add_argument("--epochs",      type=int,   default=20,
                        help="Nombre d'epochs")
    parser.add_argument("--batch_size",  type=int,   default=4,
                        help="Batch size (4 recommandé pour 24Go VRAM)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--image_size",  type=int,   default=512,
                        help="Taille des images d'entraînement")
    parser.add_argument("--save_every",  type=int,   default=500,
                        help="Sauvegarder un checkpoint tous les N steps")

    args = parser.parse_args()

    # Vérifications
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"Dataset non trouvé : {args.dataset}\n"
            "Lance d'abord : python prepare_dataset.py --output ./dataset"
        )

    meta_path = os.path.join(args.dataset, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"metadata.json non trouvé dans {args.dataset}\n"
            "Le dataset doit être préparé avec prepare_dataset.py"
        )

    # Lancer l'entraînement
    trainer = PuLIDKleinTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
