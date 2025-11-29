"""
zsrag/core/controlnet_utils.py
ControlNet utilities for SDXL Inpainting and SDEdit (Training-Free).

Features:
- Wraps SDXL ControlNet Inpaint Pipeline.
- Handles image preprocessing (Tensor <-> PIL, resizing).
- Supports Multi-ControlNet (e.g., Canny + Depth).
- Provides SDEdit (global refinement) wrapper.
"""

import torch
from typing import Optional, Dict, Any, Tuple, List, Union

# Dependency Checks
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

try:
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        ControlNetModel,
    )
    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


def _ensure_deps():
    missing = []
    if not _HAS_PIL: missing.append("Pillow")
    if not _HAS_TV: missing.append("torchvision")
    if not _HAS_DIFFUSERS: missing.append("diffusers>=0.21")
    if missing:
        raise ImportError(f"controlnet_utils requires: {', '.join(missing)}")

def _to_pil(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    """
    Converts [C,H,W] tensor (0..1 or -1..1) to PIL Image.
    """
    if isinstance(image, Image.Image):
        return image
        
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        
        if img.ndim != 3 or img.shape[0] < 1:
             raise ValueError("Expected tensor of shape [C,H,W] where C>=1.")
        
        if img.min() < 0.0:
            img = (img + 1.0) / 2.0
            
        img = img.clamp(0.0, 1.0)
        
        if img.shape[0] == 4:
            img = img[:3]
            
        return TF.to_pil_image((img * 255.0).round().to(torch.uint8))
        
    raise ValueError("image must be PIL.Image or torch.Tensor.")

def _resize_pil(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """ Resizes PIL image to (W, H) using Bicubic. """
    w, h = size
    return img.resize((w, h), resample=Image.BICUBIC)

def preprocess_control_image(
    control_image: Union[torch.Tensor, Image.Image],
    target_size: Tuple[int, int],
    normalize: bool = True,
) -> Image.Image:
    """
    Prepares control image: resize to target (W,H) and convert to PIL.
    """
    pil = _to_pil(control_image)
    pil = _resize_pil(pil, target_size)
    return pil

def preprocess_control_batch(
    control_images: List[Union[torch.Tensor, Image.Image]],
    target_size: Tuple[int, int],
) -> List[Image.Image]:
    return [preprocess_control_image(ci, target_size, normalize=True) for ci in control_images]

def load_controlnets(
    canny_model_id: Optional[str] = "diffusers/controlnet-canny-sdxl-1.0",
    depth_model_id: Optional[str] = "diffusers/controlnet-depth-sdxl-1.0",
    device: Optional[torch.device] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> Dict[str, ControlNetModel]:
    """
    Loads pre-trained SDXL ControlNet models.
    Default IDs are set to official Diffusers SDXL models.
    """
    _ensure_deps()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype or torch.float16
    
    nets: Dict[str, ControlNetModel] = {}
    
    # Using Canny as general Lineart structure for SDXL
    if canny_model_id:
        try:
            nets["canny"] = ControlNetModel.from_pretrained(canny_model_id, torch_dtype=dtype).to(dev)
        except Exception as e:
            print(f"Warning: Failed to load Canny ControlNet: {e}")
            
    if depth_model_id:
        try:
            nets["depth"] = ControlNetModel.from_pretrained(depth_model_id, torch_dtype=dtype).to(dev)
        except Exception as e:
             print(f"Warning: Failed to load Depth ControlNet: {e}")
             
    return nets


class SDXLControlNetBuilder:
    """
    Helper to build SDXL ControlNet Inpaint Pipeline.
    """
    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        controlnets: Optional[Dict[str, ControlNetModel]] = None,
    ):
        _ensure_deps()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or torch.float16
        self.base_model_id = base_model_id
        self.controlnets = controlnets or {}
        
        # Order matters! We store keys list to ensure consistency.
        self.controlnet_keys = list(self.controlnets.keys())
        self.pipe_inpaint: Optional[StableDiffusionXLControlNetInpaintPipeline] = None

    def build_inpaint_pipeline(self):
        """
        Builds the pipeline.
        WARNING: The order of ControlNets in the pipeline is determined by self.controlnet_keys.
        You must feed control_images in the SAME order.
        """
        _ensure_deps()
        
        cn_list = []
        if len(self.controlnets) == 0:
            # Fallback if no controlnet provided? 
            # Ideally InpaintPipeline can work without ControlNet if loaded differently, 
            # but SDXLControlNetInpaintPipeline expects at least one.
            raise ValueError("No ControlNet models provided for ControlNetPipeline.")
        
        for k in self.controlnet_keys:
            cn_list.append(self.controlnets[k])
            
        # If single model, pass object; if multiple, pass list
        cn_arg = cn_list[0] if len(cn_list) == 1 else cn_list

        self.pipe_inpaint = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.base_model_id,
            controlnet=cn_arg,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None
        ).to(self.device)

        # Optimization
        try:
            self.pipe_inpaint.enable_vae_slicing()
            self.pipe_inpaint.enable_vae_tiling()
        except Exception:
            pass
            
        return self.pipe_inpaint

    def get_inpaint_pipeline(self) -> StableDiffusionXLControlNetInpaintPipeline:
        if self.pipe_inpaint is None:
            return self.build_inpaint_pipeline()
        return self.pipe_inpaint


def run_controlnet_inpaint(
    pipe_inpaint: StableDiffusionXLControlNetInpaintPipeline,
    init_image: Union[Image.Image, torch.Tensor],
    mask_image: Union[Image.Image, torch.Tensor],
    prompts: Dict[str, str],
    control_images: List[Image.Image],
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.99, # High strength for inpainting (almost full redraw inside mask)
    ip_adapter_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Runs single-pass inpainting with ControlNet.
    strength defaults to 0.99 (basically 1.0) to allow full generation inside mask.
    """
    _ensure_deps()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe_inpaint.device).manual_seed(int(seed))
        
    init_pil = _to_pil(init_image)
    mask_pil = _to_pil(mask_image)
    
    # Check control images length
    # Note: Pipe expects control_image to match num_controlnets
    
    out = pipe_inpaint(
        prompt=prompts.get("pos", ""),
        negative_prompt=prompts.get("neg", ""),
        image=init_pil,
        mask_image=mask_pil,
        control_image=control_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
        **(ip_adapter_kwargs or {}),
    )
    
    imgs = out.images if hasattr(out, "images") else out[0]
    return imgs[0]


def sdxl_sde_edit(
    pipe_inpaint: StableDiffusionXLControlNetInpaintPipeline,
    image: Union[Image.Image, torch.Tensor],
    prompt: str,
    control_images: List[Image.Image],
    strength: float = 0.3,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Global SDEdit refinement.
    Uses Inpaint pipeline with a full-white mask (or just img2img mode) to refine the whole image.
    Low strength (0.2-0.35) preserves structure while unifying lighting.
    """
    _ensure_deps()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe_inpaint.device).manual_seed(int(seed))
        
    img_pil = _to_pil(image)
    H, W = img_pil.height, img_pil.width
    
    # Full white mask = edit everything
    # (InpaintPipeline treats white as "inpainting area", black as "keep")
    mask = Image.new("L", (W, H), color=255)
    
    out = pipe_inpaint(
        prompt=prompt,
        image=img_pil,
        mask_image=mask,
        control_image=control_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    )
    
    imgs = out.images if hasattr(out, "images") else out[0]
    return imgs[0]


class ControlImagePack:
    """
    Helper to bundle control images and ensure correct order/size.
    Assumes standard order: [Canny, Depth] or similar, defined by user usage.
    """
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.images: List[Image.Image] = []
        self.names: List[str] = []

    def add(self, name: str, img: Union[Image.Image, torch.Tensor]):
        """ Generic add. """
        pil = preprocess_control_image(img, self.target_size)
        self.images.append(pil)
        self.names.append(name)

    def get(self) -> List[Image.Image]:
        return list(self.images)

    def __len__(self):
        return len(self.images)
