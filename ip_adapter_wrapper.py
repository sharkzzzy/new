"""
zsrag/core/ip_adapter_wrapper.py
IP-Adapter wrapper for SDXL (Training-Free).

Features:
- Encodes reference images using CLIP Vision Model.
- Projects features to SDXL token space (1280dim + 768dim).
- Merges image tokens into encoder_hidden_states for cross-attention.
- Supports multiple reference images per subject with weighting.
"""

import os
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dependency Checks
try:
    from transformers import CLIPModel, CLIPProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from PIL import Image
    import numpy as np
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _ensure_deps():
    missing = []
    if not _HAS_TRANSFORMERS: missing.append("transformers")
    if not _HAS_PIL: missing.append("Pillow")
    if missing:
        raise ImportError(f"ip_adapter_wrapper requires: {', '.join(missing)}. Please install them.")

def _to_uint8_image(t: torch.Tensor) -> Image.Image:
    """
    Converts [C,H,W] tensor (0..1 or -1..1) to PIL Image.
    """
    if t.ndim != 3 or t.shape[0] < 1:
        raise ValueError("Expected tensor of shape [C,H,W] where C>=1.")
    
    img = t.detach().cpu()
    if img.min() < 0.0:
        img = (img + 1.0) / 2.0
    
    img = img.clamp(0.0, 1.0)
    
    # Drop alpha if present
    if img.shape[0] == 4:
        img = img[:3]
    
    img = (img * 255.0).round().to(torch.uint8)
    
    # Import locally to avoid top-level dependency crash if torchvision missing
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(img)


class MLPProjector(nn.Module):
    """
    Simple MLP Projector for IP-Adapter.
    Structure: Linear -> GELU -> Linear -> LayerNorm
    """
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens = int(num_tokens)
        
        hidden = max(out_dim, in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim * self.num_tokens)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        # img_feat: [B, in_dim]
        x = self.fc1(img_feat)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)  # [B, out_dim*num_tokens]
        x = x.view(x.shape[0], self.num_tokens, self.out_dim)  # [B, M, out_dim]
        return self.ln(x)


class IPAdapterXL:
    """
    IP-Adapter SDXL Wrapper.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_image_tokens: int = 4,
        # Pretrained weights source
        projector_weights_path_1280: Optional[str] = None,
        projector_weights_path_768: Optional[str] = None,
        hf_repo_id_1280: Optional[str] = "h94/IP-Adapter",
        hf_filename_1280: Optional[str] = "sdxl_image_proj_1280.bin",
        hf_repo_id_768: Optional[str] = "h94/IP-Adapter",
        hf_filename_768: Optional[str] = "sdxl_image_proj_768.bin",
    ):
        _ensure_deps()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.num_image_tokens = int(num_image_tokens)

        # Load CLIP Vision Encoder
        self.clip = CLIPModel.from_pretrained(clip_model_name).vision_model.to(self.device, dtype=self.dtype)
        self.clip.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Dimension Mapping
        self.clip_feat_dim = self.clip.config.hidden_size  # e.g. 1024 for ViT-L/14

        # Init Projectors
        self.proj_1280 = MLPProjector(in_dim=self.clip_feat_dim, out_dim=1280, num_tokens=self.num_image_tokens).to(self.device, dtype=self.dtype)
        self.proj_768 = MLPProjector(in_dim=self.clip_feat_dim, out_dim=768, num_tokens=self.num_image_tokens).to(self.device, dtype=self.dtype)

        # Load Weights
        self._load_projector_weights(self.proj_1280, projector_weights_path_1280, hf_repo_id_1280, hf_filename_1280)
        self._load_projector_weights(self.proj_768, projector_weights_path_768, hf_repo_id_768, hf_filename_768)

        self.default_weight = 1.0

    def _load_projector_weights(self, projector: MLPProjector, local_path: Optional[str], repo_id: Optional[str], filename: Optional[str]):
        """ Loads weights from local path or HF Hub. """
        sd = None
        
        # Try local
        if local_path and os.path.exists(local_path):
            try:
                sd = torch.load(local_path, map_location=self.device)
            except Exception as e:
                print(f"[IPAdapterXL] Failed to load local weights from {local_path}: {e}")
        
        # Try HF Hub
        if sd is None and repo_id and filename:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                sd = torch.load(path, map_location=self.device)
            except Exception as e:
                print(f"[IPAdapterXL] HF Hub download failed for {repo_id}/{filename}: {e}")
        
        if sd is not None:
            try:
                # IP-Adapter weights usually have specific keys like "image_proj.weight"
                # If loading raw state dict of the module, keys should match.
                # Often official weights are a dict {"image_proj": ...} or similar.
                # We assume the weights file is the direct state_dict of the projector module.
                # If using h94 official bin files, they contain just the state dict.
                projector.load_state_dict(sd, strict=False)
                print(f"[IPAdapterXL] Weights loaded for projector.")
            except Exception as e:
                print(f"[IPAdapterXL] load_state_dict failed: {e}. Using random init.")
        else:
            print("[IPAdapterXL] No weights found. Using random initialization (will produce noise).")

    @torch.no_grad()
    def _encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """ Encodes image to CLIP features. """
        if isinstance(image, torch.Tensor):
            pil = _to_uint8_image(image)
        elif isinstance(image, Image.Image):
            pil = image
        else:
            raise ValueError("image must be PIL.Image or torch.Tensor.")

        inputs = self.clip_processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        
        out = self.clip(pixel_values)
        
        # Use pooler_output if available (CLS token projected), else mean pool last hidden state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            img_feat = out.pooler_output
        else:
            img_feat = out.last_hidden_state.mean(dim=1)
            
        return img_feat

    @torch.no_grad()
    def compute_image_prompt_embeds(
        self,
        image: Union[Image.Image, torch.Tensor],
        weight: Optional[float] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (emb_1280, emb_768).
        """
        w = float(self.default_weight if weight is None else weight)
        img_feat = self._encode_image(image)  # [B, D_clip]
        
        if normalize:
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
            
        emb1 = self.proj_1280(img_feat)  # [B, M, 1280]
        emb2 = self.proj_768(img_feat)   # [B, M, 768]
        
        emb1 = emb1 * w
        emb2 = emb2 * w
        
        return emb1, emb2

    @torch.no_grad()
    def merge_with_text_tokens(
        self,
        text_prompt_embeds: torch.Tensor,   # [B, N, 2048]
        image_prompt_embeds: torch.Tensor,  # [B, M, 1280]
        image_prompt_embeds_2: torch.Tensor,# [B, M, 768]
    ) -> torch.Tensor:
        """
        Merges image tokens into text tokens sequence.
        """
        if text_prompt_embeds.ndim != 3 or text_prompt_embeds.shape[-1] != 2048:
            raise ValueError("text_prompt_embeds must be [B, N, 2048].")
        if image_prompt_embeds.ndim != 3 or image_prompt_embeds_2.ndim != 3:
            raise ValueError("image embeds dimension mismatch.")
            
        # Concat the two projections to match SDXL 2048 dim
        img_tokens = torch.cat([image_prompt_embeds, image_prompt_embeds_2], dim=-1) # [B, M, 2048]
        
        # Concat sequence
        merged = torch.cat([text_prompt_embeds, img_tokens], dim=1) # [B, N+M, 2048]
        return merged


def build_added_cond_kwargs_ip(
    image_prompt_embeds: torch.Tensor,
    image_prompt_embeds_2: torch.Tensor,
    time_ids: Optional[torch.Tensor] = None,
    text_embeds_pool: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Constructs added_cond_kwargs dictionary.
    """
    out = {
        "image_prompt_embeds": image_prompt_embeds,
        "image_prompt_embeds_2": image_prompt_embeds_2,
    }
    if time_ids is not None:
        out["time_ids"] = time_ids
    if text_embeds_pool is not None:
        out["text_embeds"] = text_embeds_pool
    return out


class IPAdapterManager:
    """
    Manages references for multiple subjects.
    """
    def __init__(self, ip_adapter: IPAdapterXL):
        self.ip = ip_adapter
        self.refs: Dict[str, List[Dict[str, Any]]] = {}

    def clear(self):
        self.refs.clear()

    def add_reference(self, name: str, image: Union[Image.Image, torch.Tensor], weight: float = 1.0):
        if name not in self.refs:
            self.refs[name] = []
        self.refs[name].append({"image": image, "weight": float(weight)})

    @torch.no_grad()
    def compute_embeds_for(self, name: str, agg: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes aggregated embeddings for a subject.
        """
        if name not in self.refs or len(self.refs[name]) == 0:
            raise ValueError(f"No references for subject '{name}'.")
            
        emb1_list = []
        emb2_list = []
        
        for ref in self.refs[name]:
            img = ref["image"]
            w = ref.get("weight", 1.0)
            e1, e2 = self.ip.compute_image_prompt_embeds(img, weight=w)
            emb1_list.append(e1)
            emb2_list.append(e2)
            
        # Stack: [K, B, M, D]
        emb1 = torch.stack(emb1_list, dim=0)
        emb2 = torch.stack(emb2_list, dim=0)
        
        if agg == "mean":
            emb1 = emb1.mean(dim=0)
            emb2 = emb2.mean(dim=0)
        elif agg == "sum":
            emb1 = emb1.sum(dim=0)
            emb2 = emb2.sum(dim=0)
        elif agg == "max":
            emb1 = emb1.max(dim=0).values
            emb2 = emb2.max(dim=0).values
        else:
            raise ValueError("Unsupported agg mode.")
            
        return emb1, emb2

    @torch.no_grad()
    def merge_subject_refs_into_text(
        self,
        name: str,
        text_prompt_embeds: torch.Tensor,
        agg: str = "mean",
    ) -> torch.Tensor:
        """
        Compute refs and merge into text embeds for UNet cross-attention.
        """
        emb1, emb2 = self.compute_embeds_for(name, agg=agg)
        merged = self.ip.merge_with_text_tokens(text_prompt_embeds, emb1, emb2)
        return merged

    @torch.no_grad()
    def build_added_cond_kwargs_for(
        self,
        name: str,
        time_ids: Optional[torch.Tensor] = None,
        text_embeds_pool: Optional[torch.Tensor] = None,
        agg: str = "mean",
    ) -> Dict[str, Any]:
        """
        Helper to get added_cond_kwargs.
        """
        emb1, emb2 = self.compute_embeds_for(name, agg=agg)
        return build_added_cond_kwargs_ip(emb1, emb2, time_ids=time_ids, text_embeds_pool=text_embeds_pool)
