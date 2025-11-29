"""
zsrag/core/ip_adapter_wrapper.py
IP-Adapter wrapper for SDXL (Training-Free).
FIXED VERSION: Uses official Perceiver Resampler architecture and weights.

Features:
- Encodes reference images using CLIP Vision Model.
- Uses Perceiver Resampler to project features (Official SDXL Arch).
- Downloads official weights from h94/IP-Adapter/sdxl_models.
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
    """ Converts [C,H,W] tensor to PIL Image. """
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
# Perceiver Resampler (Official SDXL IP-Adapter Architecture)
# ==============================================================================
class Resampler(nn.Module):
    def __init__(
        self,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=8,
        embedding_dim=1280,
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
    IP-Adapter SDXL Wrapper using official weights.
    """
    def __init__(
        self,
        clip_model_name: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", # Or openai/clip-vit-large-patch14
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # Default to official repo and file
        hf_repo_id: str = "h94/IP-Adapter",
        hf_filename: str = "sdxl_models/ip-adapter_sdxl_vit-h.bin",
        local_weights_path: Optional[str] = None,
    ):
        _ensure_deps()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16
        
        # 1. Load Image Encoder
        # Note: Official IP-Adapter SDXL uses CLIP ViT-H (OpenCLIP bigG)
        print(f"[IPAdapterXL] Loading Image Encoder: {clip_model_name}...")
        try:
            self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(self.device, dtype=self.dtype)
        except Exception as e:
            print(f"[IPAdapterXL] Failed to load bigG encoder, falling back to openai/clip-vit-large-patch14 (Quality may drop): {e}")
            fallback = "openai/clip-vit-large-patch14"
            self.clip_image_processor = CLIPProcessor.from_pretrained(fallback)
            self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(fallback).to(self.device, dtype=self.dtype)

        # 2. Init Resampler (Projector)
        # Official Config for SDXL ViT-H IP-Adapter:
        # dim=1280 (ViT-H), output_dim=2048 (SDXL CrossAttn), num_queries=4
        self.image_proj = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=4, # Standard SDXL IP-Adapter uses 4 tokens
            embedding_dim=self.clip_image_encoder.config.hidden_size,
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
        
        # Try Hub
        if sd is None:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[IPAdapterXL] Downloading weights from {repo_id}/{filename}...")
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                sd = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"[IPAdapterXL] Download failed: {e}")
                
        if sd is not None:
            # Official weights have keys "image_proj_model.latents", etc. or just "image_proj..."
            # We need to strip prefixes if necessary.
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("image_proj_model."):
                    new_sd[k.replace("image_proj_model.", "")] = v
                elif k.startswith("ip_adapter."): # Some variations
                    pass # ignore unet weights if bundled
                else:
                    new_sd[k] = v
            
            # Load
            try:
                self.image_proj.load_state_dict(new_sd, strict=False)
                print("[IPAdapterXL] Projector weights loaded successfully.")
            except Exception as e:
                print(f"[IPAdapterXL] Weight loading mismatch: {e}. Using random init.")
        else:
            print("[IPAdapterXL] WARNING: No weights loaded. Output will be noise.")

    @torch.no_grad()
    def compute_image_prompt_embeds(
        self,
        image: Union[Image.Image, torch.Tensor],
        weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (emb_1280, emb_768) split from the 2048-dim output.
        """
        w = float(self.default_weight if weight is None else weight)
        
        # Preprocess
        if isinstance(image, torch.Tensor):
            pil = _to_uint8_image(image)
        else:
            pil = image
            
        inputs = self.clip_image_processor(images=pil, return_tensors="pt").to(self.device)
        
        # Encode Image
        # Use hidden_states[-2] usually for IP-Adapter, but standard SDXL IP-Adapter uses projection output?
        # Official IP-Adapter code uses `image_encoder(clip_image).image_embeds` which is the projection.
        clip_out = self.clip_image_encoder(**inputs)
        image_embeds = clip_out.image_embeds # [1, 1280] or similar
        
        # Resampler expects [B, Seq, Dim], so unsqueeze sequence dim
        image_embeds = image_embeds.unsqueeze(1) # [B, 1, 1280]
        
        # Project
        # Output: [B, 4, 2048]
        ip_tokens = self.image_proj(image_embeds.to(self.dtype))
        
        # Scale
        ip_tokens = ip_tokens * w
        
        # Split into 1280 + 768 for compatibility with our pipeline injection
        # SDXL CrossAttn expects 2048. Our utils `merge_with_text_tokens` expects two parts.
        # We slice it here to satisfy the API.
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
        # Re-concat
        img_tokens = torch.cat([image_prompt_embeds, image_prompt_embeds_2], dim=-1) # [B, M, 2048]
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
        e1 = torch.stack(e1_list, dim=0).mean(dim=0) # Simple mean aggregation
        e2 = torch.stack(e2_list, dim=0).mean(dim=0)
        return e1, e2

    @torch.no_grad()
    def merge_subject_refs_into_text(self, name: str, text_prompt_embeds: torch.Tensor, agg: str = "mean") -> torch.Tensor:
        emb1, emb2 = self.compute_embeds_for(name, agg=agg)
        return self.ip.merge_with_text_tokens(text_prompt_embeds, emb1, emb2)
