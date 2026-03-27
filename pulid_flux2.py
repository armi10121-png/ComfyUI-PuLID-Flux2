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
        self.double_interval = 3
        self.single_interval = 6
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


def _get_flux2_inner_model(model):
    """Remonte jusqu'au vrai objet transformer de ComfyUI."""
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "diffusion_model"):
        return m.diffusion_model
    return m


def _detect_flux2_variant(dm) -> tuple:
    """
    Détecte automatiquement le variant Flux.2 et sa dimension cachée.
    Retourne (variant_name, hidden_dim)
    - Klein 4B  : 5 double,  20 single, dim=3072
    - Klein 9B  : 8 double,  24 single, dim=4096
    - Dev 32B   : 8 double,  48 single, dim=6144
    """
    double_blocks = getattr(dm, "transformer_blocks", None) or getattr(dm, "double_blocks", [])
    single_blocks = getattr(dm, "single_transformer_blocks", None) or getattr(dm, "single_blocks", [])

    n_double = len(double_blocks)
    n_single = len(single_blocks)

    print(f"[PuLID-Flux2] Detected: {n_double} double blocks, {n_single} single blocks")

    if n_double <= 5 and n_single <= 20:
        return "klein_4b", 3072
    elif n_double <= 8 and n_single <= 24:
        return "klein_9b", 4096
    elif n_single >= 40:
        return "flux2_dev", 6144
    else:
        warnings.warn(f"[PuLID-Flux2] Unknown variant ({n_double}d/{n_single}s), defaulting to klein_9b")
        return "klein_9b", 4096


def patch_flux2_forward(flux_model, pulid_module, id_embedding,
                        weight, sigma_start, sigma_end):

    # récupérer le vrai modèle flux
    dm = _get_flux2_inner_model(flux_model)

    # détecter automatiquement le variant
    variant, detected_dim = _detect_flux2_variant(dm)
    print(f"[PuLID-Flux2] Model variant: {variant}, dim: {detected_dim}")

    # adapter les modules CA si la dimension change
    current_dim = pulid_module.id_former.latents.shape[-1]

    # Sauvegarder les CA originaux si pas encore fait
    if not hasattr(pulid_module, "_original_pulid_ca_double"):
        pulid_module._original_pulid_ca_double = pulid_module.double_ca
        pulid_module._original_pulid_ca_single = pulid_module.single_ca
        pulid_module._original_dim = current_dim

    if detected_dim != getattr(pulid_module, "_current_adapted_dim", current_dim):
        device = pulid_module.id_former.latents.device
        dtype  = pulid_module.id_former.latents.dtype

        if detected_dim == pulid_module._original_dim:
            # Restaurer les CA originaux
            pulid_module.double_ca = pulid_module._original_pulid_ca_double
            pulid_module.single_ca = pulid_module._original_pulid_ca_single
            print(f"[PuLID-Flux2] PerceiverCA restored to original dim={detected_dim}")
        else:
            # Créer de nouveaux CA pour la dim cible
            warnings.warn(
                f"[PuLID-Flux2] Adapting PerceiverCA {pulid_module._original_dim} → {detected_dim}"
            )
            pulid_module.double_ca = nn.ModuleList([
                PerceiverAttentionCA(dim=detected_dim).to(device, dtype=dtype)
                for _ in range(len(pulid_module._original_pulid_ca_double))
            ])
            pulid_module.single_ca = nn.ModuleList([
                PerceiverAttentionCA(dim=detected_dim).to(device, dtype=dtype)
                for _ in range(len(pulid_module._original_pulid_ca_single))
            ])
            print(f"[PuLID-Flux2] PerceiverCA adapted to dim={detected_dim} ✅")

        pulid_module._current_adapted_dim = detected_dim

    # stocker données runtime
    dm._pulid_flux2_data = {
        "module": pulid_module,
        "embedding": id_embedding,
        "weight": weight,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }
    
    # Variable pour stocker le sigma courant
    current_sigma = [None]

    def _sigma_out_of_range(data):
        """Retourne True si on doit skipper ce step (sigma hors plage active)."""
        if current_sigma[0] is None:
            return False
        
        s = float(current_sigma[0])
        s_norm = s
        
        in_range = data["sigma_end"] <= s_norm <= data["sigma_start"]
        
        return not in_range

    original_model_function = None

    def capture_sigma_wrapper(original_func):
        def wrapper(x, timestep, *args, **kwargs):
            if timestep is not None:
                if isinstance(timestep, torch.Tensor):
                    current_sigma[0] = float(timestep.max().item())
                else:
                    current_sigma[0] = float(timestep)
            return original_func(x, timestep, *args, **kwargs)
        return wrapper

    if hasattr(dm, 'forward'):
        original_model_function = dm.forward
        dm.forward = capture_sigma_wrapper(original_model_function)

    # patch double blocks
    original_double_forwards = {}

    def make_double_patch(block_idx, ca_idx):
        def patched_forward(img, txt, vec, **kwargs):

            data = getattr(dm, "_pulid_flux2_data", None)

            out_img, out_txt = original_double_forwards[block_idx](
                img, txt, vec, **kwargs
            )

            if data is None:
                return out_img, out_txt

            # Extraire sigma depuis transformer_options
            if 'transformer_options' in kwargs and 'sigmas' in kwargs['transformer_options']:
                sig = kwargs['transformer_options']['sigmas']
                if sig is not None:
                    current_sigma[0] = float(sig[0]) if hasattr(sig, '__getitem__') else float(sig)
            elif 'timestep' in kwargs:
                current_sigma[0] = float(kwargs['timestep'].max()) if hasattr(kwargs['timestep'], 'max') else float(kwargs['timestep'])

            if _sigma_out_of_range(data):
                return out_img, out_txt

            embed = data["embedding"].to(out_img.device, dtype=out_img.dtype)
            ca_mod = data["module"].double_ca

            if ca_idx < len(ca_mod):
                correction = ca_mod[ca_idx](out_img, embed)

                correction = correction / (correction.norm(dim=-1, keepdim=True) + 1e-6)

                alpha = data["weight"] * 0.3

                out_img = out_img + alpha * correction

            return out_img, out_txt

        return patched_forward

    # récupérer blocks flux
    if hasattr(dm, "transformer_blocks"):
        double_blocks = dm.transformer_blocks
    elif hasattr(dm, "double_blocks"):
        double_blocks = dm.double_blocks
    else:
        double_blocks = []

    double_interval = pulid_module.double_interval
    ca_idx = 0

    for i, block in enumerate(double_blocks):
        if i % double_interval == 0:
            original_double_forwards[i] = block.forward
            block.forward = make_double_patch(i, ca_idx)
            ca_idx += 1

    # patch single blocks
    original_single_forwards = {}

    def make_single_patch(block_idx, ca_idx):
        def patched_forward(x, vec, pe, *args, **kwargs):
            data = getattr(dm, "_pulid_flux2_data", None)
            
            out = original_single_forwards[block_idx](x, vec, pe, *args, **kwargs)
            
            if data is None:
                return out
                
            if 'transformer_options' in kwargs and 'sigmas' in kwargs['transformer_options']:
                sigmas = kwargs['transformer_options']['sigmas']
                if sigmas is not None and len(sigmas) > 0:
                    current_sigma[0] = float(sigmas[0]) if isinstance(sigmas, (list, tuple)) else float(sigmas.max())
            elif 'timestep' in kwargs:
                current_sigma[0] = float(kwargs['timestep'].max()) if hasattr(kwargs['timestep'], 'max') else float(kwargs['timestep'])
                
            if _sigma_out_of_range(data):
                return out
                
            # Injection PuLID
            if isinstance(out, tuple):
                out_hidden = out[0]
                embed = data["embedding"].to(out_hidden.device, dtype=out_hidden.dtype)
                ca_mod = data["module"].single_ca
                if ca_idx < len(ca_mod):
                    correction = ca_mod[ca_idx](out_hidden, embed)
                    correction = correction / (correction.norm(dim=-1, keepdim=True) + 1e-6)
                    alpha = data["weight"] * 0.3
                    out_hidden = out_hidden + alpha * correction
                return (out_hidden,) + out[1:]
            else:
                embed = data["embedding"].to(out.device, dtype=out.dtype)
                ca_mod = data["module"].single_ca
                if ca_idx < len(ca_mod):
                    correction = ca_mod[ca_idx](out, embed)
                    out = out + data["weight"] * correction
                return out
        
        return patched_forward

    # récupérer single blocks
    if hasattr(dm, "single_transformer_blocks"):
        single_blocks = dm.single_transformer_blocks
    elif hasattr(dm, "single_blocks"):
        single_blocks = dm.single_blocks
    else:
        single_blocks = []
        warnings.warn("[PuLID-Flux2] Aucun single_block trouvé — injection partielle seulement")

    single_interval = pulid_module.single_interval
    ca_idx_single = 0

    for i, block in enumerate(single_blocks):
        if i % single_interval == 0:
            original_single_forwards[i] = block.forward
            block.forward = make_single_patch(i, ca_idx_single)
            ca_idx_single += 1

    print(
        f"[PuLID-Flux2] Patched: {len(original_double_forwards)} double blocks, "
        f"{len(original_single_forwards)} single blocks"
    )

    # cleanup / unpatch
    def unpatch():
        for i, fn in original_double_forwards.items():
            double_blocks[i].forward = fn

        for i, fn in original_single_forwards.items():
            single_blocks[i].forward = fn
            
        if original_model_function is not None:
            dm.forward = original_model_function

        if hasattr(dm, "_pulid_flux2_data"):
            del dm._pulid_flux2_data

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
        dm = _get_flux2_inner_model(work_model)
        
        # Nettoie l'ancien patch s'il existe
        if hasattr(dm, "_pulid_flux2_unpatchers"):
            for old_unpatch in dm._pulid_flux2_unpatchers:
                try:
                    old_unpatch()
                    if debug_mode:
                        print("[PuLID Debug] Ancien patch nettoyé")
                except Exception as e:
                    if debug_mode:
                        print(f"[PuLID Debug] Erreur lors du nettoyage: {e}")
            dm._pulid_flux2_unpatchers = []
        
        if hasattr(dm, "_pulid_flux2_data"):
            del dm._pulid_flux2_data
        
        # Détection de la variante Flux
        variant, detected_dim = _detect_flux2_variant(dm)
        
        # Application du patch avec sigma range
        sigma_start = 0.0
        sigma_end = 1.0
        unpatch = patch_flux2_forward(work_model, pulid_model, id_tokens, strength, sigma_start, sigma_end)
        
        # Stocker unpatch pour cleanup
        if not hasattr(dm, "_pulid_flux2_unpatchers"):
            dm._pulid_flux2_unpatchers = []
        dm._pulid_flux2_unpatchers.append(unpatch)
        
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
