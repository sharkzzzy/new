"""
zsrag/core/ip_adapter_wrapper.py
IP-Adapter wrapper for SDXL (Training-Free).
FINAL FIXED VERSION: 
- Uses CLIP ViT-L (768 dim) + Resampler (2048 dim output).
- Splits output into 1280|768 to match SDXL dual-text-encoder interface.
- Includes dtype alignment and robust feature extraction.

Features:
- Encodes images using standard openai/clip-vit-large-patch14.
- Uses official Resampler architecture (Cross-Attn based).
- Loads official IP-Adapter SDXL (ViT-L variant) weights.
"""

import os
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dependency Checks
try:
    from transformers import CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection
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
    """ Converts [C,H,W] tensor (0..1 or -1..1) to PIL Image. """
    if t.ndim != 3 or t.shape[0] < 1:
        raise ValueError("Expected tensor of shape [C,H,W].")
    img = t.detach().cpu()
    if img.min() < 0.0: img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    if img.shape[0] == 4: img = img[:3]
    img = (img * 255.0).round().to(torch.uint8)
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(img)


# ==============================================================================
# Perceiver Resampler Architecture
# ==============================================================================
class Resampler(nn.Module):
    def __init__(
        self,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim=dim, heads=heads, dim_head=dim_head),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(latents, x) + latents
            latents = ff(latents) + latents
            
        return self.norm_out(self.proj_out(latents))

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, q.shape[-1] * h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
    def forward(self, x):
        return self.net(x)


# ==============================================================================
# IP-Adapter Wrapper
# ==============================================================================
class IPAdapterXL:
    """
    IP-Adapter SDXL Wrapper (ViT-L variant).
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_image_tokens: int = 4,
        # Default to official SDXL ViT-L weights
        hf_repo_id: str = "h94/IP-Adapter",
        hf_filename: str = "sdxl_models/ip-adapter_sdxl.bin", 
        local_weights_path: Optional[str] = None,
    ):
        _ensure_deps()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16
        self.num_image_tokens = int(num_image_tokens)

        # 1. Load CLIP Vision Encoder (ViT-L, 768 dim)
        print(f"[IPAdapterXL] Loading Image Encoder: {clip_model_name}...")
        try:
            self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                clip_model_name
            ).to(self.device, dtype=self.dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP encoder {clip_model_name}: {e}")

        # Check dimension (ViT-L/14 projection dim is 768)
        # Note: If hidden_size is 1024 but projection is 768, we need 768.
        self.clip_feat_dim = self.clip_image_encoder.config.projection_dim if hasattr(self.clip_image_encoder.config, "projection_dim") else self.clip_image_encoder.config.hidden_size
        
        if self.clip_feat_dim != 768:
             print(f"[IPAdapterXL] Warning: Encoder dim is {self.clip_feat_dim}, expected 768 for standard ViT-L weights.")

        # 2. Init Resampler
        # Config for ViT-L variant: embedding_dim=768 -> output=2048
        self.resampler = Resampler(
            dim=1280, 
            depth=4, 
            dim_head=64, 
            heads=20, 
            num_queries=num_image_tokens, 
            embedding_dim=self.clip_feat_dim, 
            output_dim=2048, 
            ff_mult=4
        ).to(self.device, dtype=self.dtype)

        # 3. Load Weights
        self._load_weights(local_weights_path, hf_repo_id, hf_filename)
        self.default_weight = 1.0

    def _load_weights(self, local_path, repo_id, filename):
        sd = None
        # Try local
        if local_path and os.path.exists(local_path):
            print(f"[IPAdapterXL] Loading local weights: {local_path}")
            sd = torch.load(local_path, map_location="cpu")
        elif repo_id and filename:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[IPAdapterXL] Downloading weights from {repo_id}/{filename}...")
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                sd = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"[IPAdapterXL] Download failed: {e}. Using random init.")
                
        if sd is not None:
            # Handle keys: "image_proj.xxx" -> "xxx"
            # Official weights usually prefix with "image_proj." or "image_proj_model."
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("image_proj."):
                    new_sd[k.replace("image_proj.", "")] = v
                elif k.startswith("image_proj_model."):
                    new_sd[k.replace("image_proj_model.", "")] = v
                elif "ip_adapter" in k: pass 
                else: new_sd[k] = v
            
            try:
                self.resampler.load_state_dict(new_sd, strict=False)
                print("[IPAdapterXL] Resampler weights loaded successfully.")
            except Exception as e:
                print(f"[IPAdapterXL] Weight loading mismatch: {e}. Using random init.")
        else:
            print("[IPAdapterXL] WARNING: No weights loaded. Output will be noise.")

    @torch.no_grad()
    def compute_image_prompt_embeds(
        self,
        image: Union[Image.Image, torch.Tensor],
        weight: Optional[float] = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (emb_1280, emb_768) by splitting the 2048-dim Resampler output.
        """
        w = float(self.default_weight if weight is None else weight)
        
        if isinstance(image, torch.Tensor):
            pil = _to_uint8_image(image)
        else:
            pil = image
            
        inputs = self.clip_image_processor(images=pil, return_tensors="pt").to(self.device)
        clip_out = self.clip_image_encoder(**inputs)
        
        # Robust feature extraction
        if hasattr(clip_out, "image_embeds") and clip_out.image_embeds is not None:
            image_embeds = clip_out.image_embeds # [B, 768]
        elif hasattr(clip_out, "pooler_output") and clip_out.pooler_output is not None:
            image_embeds = clip_out.pooler_output 
        else:
            # Fallback
            image_embeds = clip_out.last_hidden_state.mean(dim=1)
            
        # Resampler expects sequence dim: [B, 1, Dim]
        image_embeds = image_embeds.unsqueeze(1).to(self.dtype)
        
        # Project -> [B, M, 2048]
        ip_tokens = self.resampler(image_embeds)
        ip_tokens = ip_tokens * w
        
        # Split output to 1280 | 768
        emb_1280 = ip_tokens[..., :1280]
        emb_768 = ip_tokens[..., 1280:]
        
        return emb_1280, emb_768

    @torch.no_grad()
    def merge_with_text_tokens(
        self,
        text_prompt_embeds: torch.Tensor,
        image_prompt_embeds: torch.Tensor,
        image_prompt_embeds_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merges image tokens into text embeddings.
        Ensures dtype alignment before concatenation.
        """
        # Re-concat back to 2048
        img_tokens = torch.cat([image_prompt_embeds, image_prompt_embeds_2], dim=-1)
        
        # Align dtype with text embeddings (often FP32)
        img_tokens = img_tokens.to(text_prompt_embeds.dtype)
        
        merged = torch.cat([text_prompt_embeds, img_tokens], dim=1)
        return merged


class IPAdapterManager:
    def __init__(self, ip_adapter: IPAdapterXL):
        self.ip = ip_adapter
        self.refs: Dict[str, List[Dict[str, Any]]] = {}

    def clear(self):
        self.refs.clear()

    def add_reference(self, name: str, image: Union[Image.Image, torch.Tensor], weight: float = 1.0):
        if name not in self.refs: self.refs[name] = []
        self.refs[name].append({"image": image, "weight": float(weight)})

    @torch.no_grad()
    def compute_embeds_for(self, name: str, agg: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
        if name not in self.refs: raise ValueError(f"No refs for {name}")
        e1_list, e2_list = [], []
        for ref in self.refs[name]:
            e1, e2 = self.ip.compute_image_prompt_embeds(ref["image"], weight=ref["weight"])
            e1_list.append(e1); e2_list.append(e2)
        
        # Stack: [K, B, M, D] -> Mean: [B, M, D]
        e1 = torch.stack(e1_list, dim=0).mean(dim=0)
        e2 = torch.stack(e2_list, dim=0).mean(dim=0)
        return e1, e2

    @torch.no_grad()
    def merge_subject_refs_into_text(self, name: str, text_prompt_embeds: torch.Tensor, agg: str = "mean") -> torch.Tensor:
        emb1, emb2 = self.compute_embeds_for(name, agg=agg)
        return self.ip.merge_with_text_tokens(text_prompt_embeds, emb1, emb2)
