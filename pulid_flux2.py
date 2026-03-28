"""
ComfyUI-PuLID-Flux2 - VERSION AMÉLIORÉE
Custom node PuLID for FLUX.2 — Klein 4B, Klein 9B, Dev 32B
Version optimisée avec cache, gestion d'erreurs renforcée et paramètres configurables
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings

import comfy.model_management
import folder_paths

# Configuration des dossiers
PULID_DIR = os.path.join(folder_paths.models_dir, "pulid")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
os.makedirs(PULID_DIR, exist_ok=True)
os.makedirs(INSIGHTFACE_DIR, exist_ok=True)

# ============================================================================
# CACHE GLOBAL pour les modèles
# ============================================================================

_MODEL_CACHE = {
    "eva_clip": None,
    "insightface": None,
    "eva_device": None,
    "insightface_device": None,
}


def get_cached_model(model_type: str, device: torch.device, loader_func):
    """Récupère un modèle depuis le cache ou le charge si nécessaire"""
    cache_key = f"{model_type}"
    device_key = f"{model_type}_device"
    
    # Vérifier si le modèle existe et est sur le bon device
    if _MODEL_CACHE[cache_key] is not None:
        if _MODEL_CACHE[device_key] == device:
            return _MODEL_CACHE[cache_key]
        else:
            # Déplacer vers le nouveau device
            _MODEL_CACHE[cache_key] = _MODEL_CACHE[cache_key].to(device)
            _MODEL_CACHE[device_key] = device
            return _MODEL_CACHE[cache_key]
    
    # Charger le modèle
    model = loader_func(device)
    _MODEL_CACHE[cache_key] = model
    _MODEL_CACHE[device_key] = device
    return model


# ============================================================================
# Classes d'attention pour PuLID
# ============================================================================

class PerceiverAttentionCA(nn.Module):
    """Couche d'attention croisée pour le mécanisme Perceiver"""
    def __init__(self, dim: int = 3072, dim_head: int = 64, heads: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        target_dtype = self.norm1.weight.dtype
        
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if context.dtype != target_dtype:
            context = context.to(target_dtype)

        x_n = self.norm1(x)
        ctx = self.norm2(context)

        q = self.to_q(x_n)
        kv = self.to_kv(ctx)
        k, v = kv.chunk(2, dim=-1)

        def reshape(t):
            return t.view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(attn_out)


class IDFormer(nn.Module):
    """IDFormer - Transforme l'embedding d'identité en tokens"""
    def __init__(self, dim: int = 4096, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        
        self.proj = nn.Sequential(
            nn.Linear(512 + 768, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )
        self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.layers = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(4)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, id_embed: torch.Tensor, clip_embed: torch.Tensor) -> torch.Tensor:
        B = id_embed.shape[0]
        # Centrage du clip embedding
        clip_embed = clip_embed - clip_embed.mean(dim=-1, keepdim=True)
        clip_embed = 1 * clip_embed
        
        combined = torch.cat([id_embed, clip_embed], dim=-1)
        tokens = self.proj(combined).view(B, self.num_tokens, -1)
        
        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = latents + layer(latents, tokens)
        return self.norm(latents)


class PuLIDFlux2(nn.Module):
    """Modèle PuLID principal pour Flux.2"""
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.id_former = IDFormer(dim=dim)
        self.dim = dim
        self.double_ca = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(10)])
        self.single_ca = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(30)])

    @classmethod
    def from_pretrained(cls, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        dim = state["id_former.latents"].shape[-1]
        model = cls(dim=dim)
        model.load_state_dict(state, strict=False)
        return model


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def get_flux_inner(model):
    """Extrait le diffusion model interne de Flux"""
    if hasattr(model, "model"):
        model = model.model
    if hasattr(model, "diffusion_model"):
        model = model.diffusion_model
    return model


def detect_flux_variant(model) -> Tuple[str, int]:
    """Détecte la variante de Flux utilisée (Klein 4B, Klein 9B, Dev 32B)"""
    double = getattr(model, "transformer_blocks", None) or getattr(model, "double_blocks", [])
    single = getattr(model, "single_transformer_blocks", None) or getattr(model, "single_blocks", [])
    n_double, n_single = len(double), len(single)
    
    if n_double <= 5 and n_single <= 20:
        return "klein_4b", 3072
    elif n_double <= 8 and n_single <= 24:
        return "klein_9b", 4096
    elif n_single >= 40:
        return "flux2_dev", 6144
    else:
        logging.warning(f"[PuLID-Flux2] Unknown variant ({n_double}d/{n_single}s), defaulting to klein_9b")
        return "klein_9b", 4096


def load_eva_clip(device):
    """Charge le modèle EVA-CLIP pour l'extraction des features"""
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k",
        )
        visual = model.visual
        visual.eval().to(device)
        return visual
    except Exception as e:
        warnings.warn(f"[PuLID] Erreur lors du chargement d'EVA-CLIP: {e}")
        return None


def get_scale_factors(block_idx: int, total_blocks: int, block_type: str = "double") -> float:
    """
    Calcule le facteur d'échelle pour un bloc donné
    
    Args:
        block_idx: Index du bloc actuel
        total_blocks: Nombre total de blocs
        block_type: "double" ou "single"
    
    Returns:
        float: Facteur d'échelle pour ce bloc
    """
    if block_type == "double":
        # Facteurs plus forts au début, décroissants
        if block_idx < 3:
            return 8.0
        elif block_idx < 5:
            return 5.0
        else:
            return 3.0
    else:  # single
        if block_idx < 4:
            return 6.0
        elif block_idx < 8:
            return 4.0
        else:
            return 2.0


def patch_flux(model, pulid_module, id_tokens, strength, debug=False):
    """
    Patch le modèle Flux avec les tokens d'identité PuLID
    
    Args:
        model: Modèle Flux à patcher
        pulid_module: Module PuLID contenant les couches d'attention
        id_tokens: Tokens d'identité extraits
        strength: Intensité du PuLID (0-2)
        debug: Active les logs de debug
    
    Returns:
        Fonction unpatch pour restaurer le modèle original
    """
    dm = get_flux_inner(model)
    
    double_blocks = getattr(dm, "transformer_blocks", None) or getattr(dm, "double_blocks", [])
    single_blocks = getattr(dm, "single_transformer_blocks", None) or getattr(dm, "single_blocks", [])
    
    original_double = {}
    original_single = {}
    
    # Patch des double blocks
    for idx, block in enumerate(double_blocks):
        original_double[idx] = block.forward
        
        def make_double_patch(block_idx, ca_idx):
            def patched(img, txt, vec, **kwargs):
                out_img, out_txt = original_double[block_idx](img, txt, vec, **kwargs)
                
                if ca_idx < len(pulid_module.double_ca):
                    factor = get_scale_factors(block_idx, len(double_blocks), "double")
                    
                    ca = pulid_module.double_ca[ca_idx]
                    correction = ca(out_img, id_tokens)
                    
                    # Normalisation optimisée avec F.normalize
                    correction = F.normalize(correction, p=2, dim=-1)
                    out_img = out_img + strength * factor * correction
                    
                    if debug:
                        print(f"[Double Block {block_idx}] factor={factor:.1f}, correction_norm={correction.norm():.4f}")
                
                return out_img, out_txt
            return patched
        
        ca_idx = min(idx, len(pulid_module.double_ca) - 1)
        block.forward = make_double_patch(idx, ca_idx)
    
    # Patch des single blocks
    for idx, block in enumerate(single_blocks):
        original_single[idx] = block.forward
        
        def make_single_patch(block_idx, ca_idx):
            def patched(x, vec, pe, *args, **kwargs):
                try:
                    if len(args) > 0:
                        out = original_single[block_idx](x, vec, pe, args[0], **kwargs)
                    else:
                        out = original_single[block_idx](x, vec, pe, **kwargs)
                except Exception as e:
                    if debug:
                        print(f"[Single Block {block_idx}] Fallback appelé: {e}")
                    out = original_single[block_idx](x, vec, pe, **kwargs)
                
                if ca_idx < len(pulid_module.single_ca):
                    factor = get_scale_factors(block_idx, len(single_blocks), "single")
                    
                    ca = pulid_module.single_ca[ca_idx]
                    correction = ca(out, id_tokens)
                    
                    # Normalisation optimisée
                    correction = F.normalize(correction, p=2, dim=-1)
                    out = out + strength * factor * correction
                    
                    if debug:
                        print(f"[Single Block {block_idx}] factor={factor:.1f}, correction_norm={correction.norm():.4f}")
                
                return out
            return patched
        
        ca_idx = min(idx, len(pulid_module.single_ca) - 1)
        block.forward = make_single_patch(idx, ca_idx)
    
    # Fonction pour restaurer l'état original
    def unpatch():
        for idx, block in enumerate(double_blocks):
            if idx in original_double:
                block.forward = original_double[idx]
        for idx, block in enumerate(single_blocks):
            if idx in original_single:
                block.forward = original_single[idx]
    
    return unpatch


# ============================================================================
# Nodes ComfyUI
# ============================================================================

class PuLIDInsightFaceLoader:
    """Charge InsightFace pour la détection et l'analyse des visages"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"],),
            }
        }
    
    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load"
    CATEGORY = "PuLID-Flux2"
    
    def load(self, provider):
        def _load_insightface(device):
            try:
                from insightface.app import FaceAnalysis
                providers = {
                    "CPU": "CPUExecutionProvider",
                    "CUDA": "CUDAExecutionProvider", 
                    "ROCM": "ROCMExecutionProvider",
                }
                
                app = FaceAnalysis(
                    name="antelopev2",
                    root=INSIGHTFACE_DIR,
                    providers=[providers.get(provider, "CPUExecutionProvider")]
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                return app
            except Exception as e:
                raise RuntimeError(f"[PuLID] Erreur lors du chargement d'InsightFace: {e}")
        
        device = comfy.model_management.get_torch_device()
        model = get_cached_model("insightface", device, _load_insightface)
        
        print(f"✅ InsightFace chargé (provider={provider}, cache={'HIT' if _MODEL_CACHE['insightface'] else 'MISS'})")
        return (model,)


class PuLIDEVACLIPLoader:
    """Charge EVA-CLIP pour l'extraction des features visuelles"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load"
    CATEGORY = "PuLID-Flux2"
    
    def load(self):
        device = comfy.model_management.get_torch_device()
        model = get_cached_model("eva_clip", device, load_eva_clip)
        
        if model is None:
            raise RuntimeError("[PuLID] EVA-CLIP non disponible. Vérifiez l'installation d'open_clip.")
        
        print(f"✅ EVA-CLIP chargé (cache={'HIT' if _MODEL_CACHE['eva_clip'] else 'MISS'})")
        return (model,)


class PuLIDModelLoader:
    """Charge le modèle PuLID depuis un fichier .safetensors ou .pt"""
    @classmethod
    def INPUT_TYPES(cls):
        if not os.path.exists(PULID_DIR):
            os.makedirs(PULID_DIR, exist_ok=True)
        
        files = [f for f in os.listdir(PULID_DIR) if f.endswith((".safetensors", ".pt", ".bin"))]

        if not files:
            files = ["__create_new__"]
        
        return {
            "required": {
                "pulid_file": (files,),
            }
        }
    
    RETURN_TYPES = ("PULID_MODEL",)
    FUNCTION = "load"
    CATEGORY = "PuLID-Flux2"
    
    def load(self, pulid_file):
        if pulid_file == "__create_new__":
            print("⚠️  Création d'un nouveau modèle PuLID (non entraîné)")
            return (PuLIDFlux2(dim=4096),)
        
        path = os.path.join(PULID_DIR, pulid_file)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"[PuLID] Fichier introuvable: {path}")
        
        try:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(path, device="cpu")
                dim = state["id_former.latents"].shape[-1]
                model = PuLIDFlux2(dim=dim)
                model.load_state_dict(state, strict=False)
            else:
                model = PuLIDFlux2.from_pretrained(path)
            
            model.eval()
            print(f"✅ Modèle PuLID chargé: {pulid_file} (dim={model.dim})")
            return (model,)
        except Exception as e:
            raise RuntimeError(f"[PuLID] Erreur lors du chargement du modèle: {e}")


class ApplyPuLIDFlux2:
    """
    Applique PuLID au modèle Flux.2
    VERSION AMÉLIORÉE avec gestion d'erreurs renforcée
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_model": ("PULID_MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "eva_clip": ("EVA_CLIP",),
                "face_analysis": ("INSIGHTFACE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "face_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "PuLID-Flux2"
    
    def apply(self, model, pulid_model, strength, eva_clip, face_analysis, image, 
              face_index=0, debug_mode=False):
        
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16
        
        # Validation de l'image
        if image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0:
            raise ValueError("[PuLID] Image invalide (dimensions nulles)")
        
        # Extraction du visage depuis l'image
        img_np = (image[0].numpy() * 255).astype(np.uint8)
        faces = face_analysis.get(img_np)
        
        # Gestion du cas "aucun visage détecté"
        if not faces:
            warning_msg = "[PuLID] ⚠️  AUCUN VISAGE DÉTECTÉ dans l'image"
            print(warning_msg)
            print("[PuLID] → Mode fallback activé: modèle retourné sans modification")
            return (model,)
        
        # Tri des visages par taille (du plus grand au plus petit)
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        
        # Validation de face_index
        if face_index >= len(faces):
            warning_msg = f"[PuLID] ⚠️  face_index={face_index} hors limites (max={len(faces)-1})"
            print(f"{warning_msg} → Utilisation du plus grand visage (index 0)")
            face_index = 0
        
        face = faces[face_index]
        
        if debug_mode:
            print(f"[PuLID Debug] Visages détectés: {len(faces)}")
            print(f"[PuLID Debug] Visage sélectionné: index={face_index}, bbox={face.bbox}")
        
        # Embedding d'identité (512 dims)
        id_embed = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
        id_embed = F.normalize(id_embed, dim=-1)
        
        # Crop du visage pour EVA-CLIP avec marge
        x1, y1, x2, y2 = face.bbox.astype(int)
        margin = int(max(x2-x1, y2-y1) * 0.2)
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(img_np.shape[1], x2+margin), min(img_np.shape[0], y2+margin)
        
        face_crop = image[:1, y1:y2, x1:x2, :]
        
        # Validation du crop
        if face_crop.shape[1] == 0 or face_crop.shape[2] == 0:
            print("[PuLID] ⚠️  Crop invalide, utilisation de l'image complète")
            face_crop = image[:1]
        
        # Préparation pour EVA-CLIP (resize + normalisation)
        face_crop = F.interpolate(face_crop.permute(0,3,1,2), size=(336,336), mode="bilinear").to(device)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        face_crop = (face_crop - mean) / std
        
        # Extraction des features EVA-CLIP (768 dims)
        with torch.no_grad():
            clip_out = eva_clip(face_crop.float())
            if isinstance(clip_out, (list, tuple)):
                clip_out = clip_out[0]
            if clip_out.dim() == 3:
                clip_out = clip_out[:, 0, :]
            clip_embed = clip_out.to(device, dtype=dtype)
        
        if debug_mode:
            print(f"[PuLID Debug] id_embed shape: {id_embed.shape}, clip_embed shape: {clip_embed.shape}")
        
        # Génération des tokens d'identité
        pulid_model = pulid_model.to(device, dtype=dtype)
        
        with torch.no_grad():
            id_tokens = pulid_model.id_former(id_embed, clip_embed)
            id_tokens = F.normalize(id_tokens, p=2, dim=-1)
        
        if debug_mode:
            print(f"[PuLID Debug] id_tokens shape: {id_tokens.shape}, norm: {id_tokens.norm():.4f}")
        
        # Clonage du modèle pour ne pas affecter l'original
        work_model = model.clone()
        dm = get_flux_inner(work_model)
        
        # Nettoie l'ancien patch s'il existe
        if hasattr(dm, "_pulid_unpatcher"):
            try:
                dm._pulid_unpatcher()
                if debug_mode:
                    print("[PuLID Debug] Ancien patch nettoyé")
            except Exception as e:
                if debug_mode:
                    print(f"[PuLID Debug] Erreur lors du nettoyage: {e}")
        
        # Détection de la variante Flux
        variant, detected_dim = detect_flux_variant(dm)
        
        # Projection si nécessaire
        if id_tokens.shape[-1] != detected_dim:
            print(f"[PuLID] 🔄 Projection nécessaire: {id_tokens.shape[-1]} → {detected_dim}")
            proj = nn.Linear(id_tokens.shape[-1], detected_dim, bias=False).to(device, dtype=dtype)
            nn.init.normal_(proj.weight, std=0.01)
            id_tokens = proj(id_tokens)
            
            if debug_mode:
                print(f"[PuLID Debug] id_tokens après projection: {id_tokens.shape}")
        
        # Application du patch
        unpatch = patch_flux(work_model, pulid_model, id_tokens, strength, debug=debug_mode)
        dm._pulid_unpatcher = unpatch
        
        # Affichage des informations
        if strength == 0:
            print("⚪ PuLID: OFF (strength=0)")
        else:
            emoji = "🟢" if strength >= 1.0 else "🟡"
            print(f"{emoji} PuLID: ON | {variant} | strength={strength:.2f} | face={face_index}/{len(faces)-1}")
        
        return (work_model,)


class PuLIDFacePreview:
    """Affiche les visages détectés avec leur index et informations de confiance"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_analysis": ("INSIGHTFACE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "show_confidence": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "PuLID-Flux2"
    OUTPUT_NODE = True
    
    def preview(self, face_analysis, image, show_confidence=True):
        try:
            import cv2
            img_np = (image[0].numpy() * 255).astype(np.uint8).copy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            faces = face_analysis.get(img_np)
            
            if not faces:
                # Ajouter un message "No faces detected"
                h, w = img_bgr.shape[:2]
                cv2.putText(img_bgr, "No faces detected", (w//2-100, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            else:
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    
                    # Rectangle vert autour du visage
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                    
                    # Label avec index
                    label = f"Face {i}"
                    if show_confidence and hasattr(face, 'det_score'):
                        label += f" ({face.det_score:.2f})"
                    
                    cv2.putText(img_bgr, label, (x1, y1-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
                    # Afficher la taille du visage
                    size = f"{x2-x1}x{y2-y1}"
                    cv2.putText(img_bgr, size, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            
            print(f"[PuLID Preview] Visages détectés: {len(faces)}")
            return (out,)
        except Exception as e:
            print(f"[PuLID Preview] Erreur: {e}")
            return (image,)


# ============================================================================
# Enregistrement des nodes
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PuLIDInsightFaceLoader": PuLIDInsightFaceLoader,
    "PuLIDEVACLIPLoader": PuLIDEVACLIPLoader,
    "PuLIDModelLoader": PuLIDModelLoader,
    "ApplyPuLIDFlux2": ApplyPuLIDFlux2,
    "PuLIDFacePreview": PuLIDFacePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLIDInsightFaceLoader": "Load InsightFace (PuLID) ⚡",
    "PuLIDEVACLIPLoader": "Load EVA-CLIP (PuLID) ⚡",
    "PuLIDModelLoader": "Load PuLID ✦ Flux.2",
    "ApplyPuLIDFlux2": "Apply PuLID ✦ Flux.2 [IMPROVED]",
    "PuLIDFacePreview": "PuLID — Face Preview 🔍",
}
