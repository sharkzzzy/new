"""
zsrag/core/utils.py
Common utilities for ZS-RAG (Image handling, SDXL helpers, Seeding).
"""

import os
import random
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn.functional as F

# Core Dependencies
try:
    from PIL import Image
    import numpy as np
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import torchvision.transforms.functional as TF
    _HAS_TV = True
except ImportError:
    _HAS_TV = False

# Diffusers Dependency
try:
    from diffusers import StableDiffusionXLPipeline
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


def _ensure_deps():
    missing = []
    if not _HAS_PIL: missing.append("Pillow")
    if not _HAS_TV: missing.append("torchvision")
    if not _HAS_DIFFUSERS: missing.append("diffusers")
    
    if missing:
        raise ImportError(f"utils requires: {', '.join(missing)}")

def seed_everything(seed: int):
    """
    Sets seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if _HAS_PIL:
        np.random.seed(seed)
        
    random.seed(seed)

def to_pil(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    """
    Converts tensor or PIL to PIL Image.
    """
    _ensure_deps()
    if isinstance(image, Image.Image):
        return image.convert("RGB")
        
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim != 3 or img.shape[0] < 1:
            raise ValueError("Expected tensor of shape [C,H,W].")
        
        if img.min() < 0.0:
            img = (img + 1.0) / 2.0
        
        img = img.clamp(0.0, 1.0)
        
        if img.shape[0] == 4:
            img = img[:3]
            
        return TF.to_pil_image((img * 255.0).round().to(torch.uint8))
        
    raise ValueError("image must be PIL.Image or torch.Tensor.")

def pil_to_tensor_uint8(img: Image.Image) -> torch.Tensor:
    """
    PIL.Image -> [3,H,W] uint8 tensor.
    """
    if not isinstance(img, Image.Image):
        raise ValueError("img must be PIL.Image")
    
    # Ensure RGB
    img = img.convert("RGB")
    arr = np.array(img)
    
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.to(torch.uint8)

def tensor_to_float01(img: torch.Tensor) -> torch.Tensor:
    """
    Converts [C,H,W] tensor (uint8 or float) to 0..1 float32.
    """
    if img.ndim != 3:
        raise ValueError("img must be [C,H,W]")
    
    if img.dtype == torch.uint8:
        return (img.float() / 255.0).clamp(0.0, 1.0)
    
    if img.min() < 0.0:
        # Assume -1..1
        img = (img + 1.0) / 2.0
        
    return img.float().clamp(0.0, 1.0)

def resize_pil(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """ Resizes PIL image to (W, H). """
    w, h = int(size[0]), int(size[1])
    return img.resize((w, h), resample=Image.BICUBIC)

def interpolate_bilinear(t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """ Bilinear interpolation for tensors. """
    return F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)

def get_base_latent_size(width: int, height: int, downsample: int = 8) -> Tuple[int, int]:
    """ Returns (H_lat, W_lat). """
    return height // downsample, width // downsample

def load_sdxl_base(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: Optional[torch.device] = None,
    torch_dtype: Optional[torch.dtype] = None,
    vae_tiling: bool = True,
    vae_slicing: bool = True,
) -> StableDiffusionXLPipeline:
    """
    Loads SDXL Base Pipeline.
    """
    _ensure_deps()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype or torch.float16
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True
    ).to(dev)

    # VAE Optimization
    try:
        if vae_tiling:
            pipe.enable_vae_tiling()
        if vae_slicing:
            pipe.enable_vae_slicing()
    except Exception:
        pass
        
    return pipe

def force_fp32(pipe: StableDiffusionXLPipeline):
    """
    Forces critical components to FP32 for stability.
    """
    pipe.vae.to(dtype=torch.float32)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to(dtype=torch.float32)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.to(dtype=torch.float32)
    pipe.unet.to(dtype=torch.float32)

def make_time_ids(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Constructs SDXL time_ids: [original_size, crop_coords, target_size].
    """
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    return torch.tensor([add_time_ids], dtype=torch.float32, device=device)

def attach_attention_processor(unet, processor):
    """
    Attaches custom processor to UNet and injects 'layer_id' attribute to modules.
    """
    for n, m in unet.named_modules():
        if hasattr(m, "to_q") and hasattr(m, "to_k"):
            setattr(m, "layer_id", n)
    unet.set_attn_processor(processor)
def prepare_latents(
    batch_size: int,
    channels: int,
    height_lat: int,
    width_lat: int,
    dtype: torch.dtype,
    device: torch.device,
    scheduler, # 保留参数位，但如果 init_image 存在则不用来加噪
    seed: Optional[int] = None,
    vae=None,
    init_image=None,
) -> torch.Tensor:
    """
    Returns clean latents (if init_image) OR random scaled noise (if no init_image).
    Does NOT handle SDEdit noise addition logic.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        
    shape = (batch_size, channels, height_lat, width_lat)
    
    # 1. Pure Noise (Text-to-Image mode base)
    if init_image is None or vae is None:
        noise = torch.randn(shape, device=device, dtype=dtype, generator=gen)
        return noise * scheduler.init_noise_sigma

    # 2. Image-to-Image: Encode to Clean Latent
    from zsrag.core.utils import pil_to_tensor_uint8, tensor_to_float01
    
    if not isinstance(init_image, torch.Tensor):
        img_t = pil_to_tensor_uint8(init_image)
        img_t = tensor_to_float01(img_t).unsqueeze(0).to(device=device, dtype=vae.dtype)
    else:
        img_t = init_image.to(device=device, dtype=vae.dtype)
        if img_t.ndim == 3: img_t = img_t.unsqueeze(0)
    
    img_t = (img_t * 2.0) - 1.0
    
    with torch.no_grad():
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
             latents_clean = vae.encode(img_t).latent_dist.sample(generator=gen)
             latents_clean = (latents_clean - vae.config.shift_factor) * vae.config.scaling_factor
        else:
             latents_clean = vae.encode(img_t).latent_dist.sample(generator=gen) * vae.config.scaling_factor
    
    return latents_clean.to(dtype=dtype)


@torch.no_grad()
def decode_vae(vae, latents: torch.Tensor) -> torch.Tensor:
    """
    Decodes latents to [B,3,H,W] float image (0..1).
    """
    latents = latents / vae.config.scaling_factor
    latents = latents.to(dtype=vae.dtype)
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def build_added_kwargs_sdxl(
    pooled_prompt_embeds: torch.Tensor,
    time_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """ Helper for added_cond_kwargs. """
    return {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

def encode_prompt_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str = "",
    device: Optional[torch.device] = None,
    do_classifier_free_guidance: bool = True,
    num_images_per_prompt: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wraps SDXL prompt encoding.
    Returns: (pos_embeds, neg_embeds, pooled_pos, pooled_neg)
    """
    dev = device or pipe.device
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=dev,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
    )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def sdxl_token_indices_for_keywords(
    tokenizer,
    prompt: str,
    keywords: List[str],
    offset: int = 1,
) -> List[int]:
    """
    Helper to find token indices for keywords.
    Note: SDXL uses dual encoders; this only checks the first tokenizer.
    Use with caution or for heuristic probing.
    """
    try:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception:
        return []
        
    decoded = [tokenizer.decode([i]).strip().lower() for i in input_ids]
    indices = []
    kw_lower = [k.lower() for k in keywords]
    
    for i, token_str in enumerate(decoded):
        if any(k in token_str for k in kw_lower):
            indices.append(i + int(offset))
            
    return indices

def save_image_tensor(image: torch.Tensor, path: str):
    """
    Saves the first image in batch to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = image[0].detach().cpu().clamp(0.0, 1.0)
    
    if img.shape[0] == 4:
        img = img[:3]
        
    pil = TF.to_pil_image((img * 255.0).round().to(torch.uint8))
    pil.save(path)
