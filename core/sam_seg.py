"""
zsrag/core/sam_seg.py
Segment Anything Model (SAM) wrapper for zero-shot mask generation.

Features:
- Automatic model loading (Local checkpoint or HF Hub).
- Mask generation from Boxes or Points.
- Mask post-processing (Standardization, Erosion, Dilation, Feathering).
- Utilities for mask resizing and clipping.
"""

import os
from typing import Dict, Tuple, List, Optional, Union

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

# SAM Dependency
try:
    from segment_anything import sam_model_registry, SamPredictor
    _HAS_SAM = True
except ImportError:
    _HAS_SAM = False

# HF Hub Dependency (Optional)
try:
    from huggingface_hub import hf_hub_download
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


def _ensure_deps():
    missing = []
    if not _HAS_PIL: missing.append("Pillow")
    if not _HAS_TV: missing.append("torchvision")
    
    if missing:
        raise ImportError(f"sam_seg requires: {', '.join(missing)}")
    
    if not _HAS_SAM:
        raise ImportError("segment_anything not installed. Please install 'segment-anything'.")

def _to_pil(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    """ Converts tensor or PIL to PIL Image. """
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
            
        img = (img * 255.0).round().to(torch.uint8)
        return TF.to_pil_image(img)
    raise ValueError("image must be PIL.Image or torch.Tensor.")

def _pil_to_np_uint8(img: Image.Image) -> np.ndarray:
    """ PIL.Image -> numpy uint8 HxWx3 """
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)

def _resize_pil(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """ Resizes PIL image to (W, H). """
    w, h = int(size[0]), int(size[1])
    return img.resize((w, h), resample=Image.BICUBIC)

def _gaussian_kernel(ksize: int, sigma: float, device, dtype):
    if ksize % 2 == 0:
        ksize += 1
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    xx = ax[None, :]
    yy = ax[:, None]
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[None, None, :, :]

def _gaussian_blur(m: torch.Tensor, ksize: int = 7, sigma: float = 1.5) -> torch.Tensor:
    kernel = _gaussian_kernel(ksize, sigma, m.device, m.dtype)
    pad = ksize // 2
    m_padded = F.pad(m, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(m_padded, kernel)

def _dilate(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)

def _erode(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad)

def _clip_to_box(mask: torch.Tensor, box_xyxy: Tuple[int, int, int, int]) -> torch.Tensor:
    """ Clips mask to safety box boundaries. """
    H, W = mask.shape[-2], mask.shape[-1]
    x1, y1, x2, y2 = box_xyxy
    clipped = torch.zeros_like(mask)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    
    if x2 > x1 and y2 > y1:
        clipped[..., y1:y2, x1:x2] = mask[..., y1:y2, x1:x2]
    return clipped

def _bbox_from_mask(mask: torch.Tensor, min_area: int = 64) -> Optional[Tuple[int, int, int, int]]:
    """ Extracts bbox from mask. Returns None if area too small. """
    if mask.ndim == 4:
        m = mask[0, 0]
    elif mask.ndim == 2:
        m = mask
    else:
        raise ValueError("mask must be [1,1,H,W] or [H,W].")
        
    ys, xs = torch.where(m > 0.5)
    if ys.numel() == 0:
        return None
        
    y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
    x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
    
    area = (y2 - y1) * (x2 - x1)
    if area < min_area:
        return None
        
    H, W = m.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return (x1, y1, x2, y2)

def simple_mask_from_box(
    size: Tuple[int, int], 
    box_xyxy: Tuple[int, int, int, int], 
    device: torch.device, 
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Creates a binary rectangular mask from a box. Fallback method.
    size: (Width, Height)
    Returns: [1, 1, H, W]
    """
    W, H = int(size[0]), int(size[1])
    x1, y1, x2, y2 = box_xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    
    m = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    if x2 > x1 and y2 > y1:
        m[..., y1:y2, x1:x2] = 1.0
    return m


class SAMSegmenter:
    """
    Wrapper for Segment Anything Model (Zero-Shot Segmentation).
    """
    def __init__(
        self,
        model_type: str = "vit_h",  # "vit_h" | "vit_l" | "vit_b"
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        auto_download: bool = True,
    ):
        _ensure_deps()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.auto_download = auto_download

        self.predictor: Optional[SamPredictor] = None
        self.image_np: Optional[np.ndarray] = None  # HxWx3 uint8
        self.image_size: Optional[Tuple[int, int]] = None  # (W,H)

        self._load_model()

    def _resolve_checkpoint(self) -> str:
        """
        Resolves checkpoint path. Tries HF Hub download if local path is missing.
        """
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return self.checkpoint_path
            
        if not self.auto_download or not _HAS_HF:
            raise FileNotFoundError("SAM checkpoint not provided and auto_download disabled or huggingface_hub missing.")
            
        # Official checkpoints (usually mirrored or available in various repos)
        # Using a reliable repo ID for standard SAM checkpoints.
        # Often 'ybelkada/segment-anything' or similar. 
        # Here we assume a user might need to set up the correct repo if default fails.
        fname_map = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        fname = fname_map.get(self.model_type, fname_map["vit_h"])
        
        # Note: This Repo ID is hypothetical for "official" weights directly.
        # You might need to change this to a valid HF Model Hub ID that hosts these files.
        # e.g. "facebook/sam-vit-huge" might provide safetensors now.
        # For legacy .pth, we assume the user provides path or we try a known mirror.
        # Let's try to assume local file first. If download fails, user must provide path.
        try:
             # Try a known mirror if official doesn't work directly via Hub
             path = hf_hub_download(repo_id="ybelkada/segment-anything", filename=f"checkpoints/{fname}")
             return path
        except Exception:
             # Fallback attempt
             print("Failed to download from mirror. Please providing local checkpoint_path.")
             raise
             
    def _load_model(self):
        ckpt = self._resolve_checkpoint()
        sam = sam_model_registry[self.model_type](checkpoint=ckpt).to(device=self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image: Union[torch.Tensor, Image.Image]):
        """
        Sets the image for segmentation.
        """
        pil = _to_pil(image)
        arr = _pil_to_np_uint8(pil)
        self.image_np = arr
        self.image_size = (pil.width, pil.height)
        self.predictor.set_image(arr)

    def predict_from_boxes(
        self,
        boxes: Dict[str, Tuple[int, int, int, int]],
        multimask_output: bool = False,
        box_expansion: float = 0.0,
        postprocess: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict masks from boxes.
        boxes: {name: (x1, y1, x2, y2)}
        """
        if self.predictor is None or self.image_np is None:
            raise RuntimeError("Predictor not initialized or image not set.")

        H, W = self.image_np.shape[:2]
        out_masks: Dict[str, torch.Tensor] = {}
        
        for name, box in boxes.items():
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Expand box
            if box_expansion and box_expansion > 0.0:
                w_box, h_box = x2 - x1, y2 - y1
                ex, ey = int(w_box * box_expansion), int(h_box * box_expansion)
                x1, y1 = max(0, x1 - ex), max(0, y1 - ey)
                x2, y2 = min(W, x2 + ex), min(H, y2 + ey)

            input_box = np.array([x1, y1, x2, y2], dtype=np.int32)
            
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=bool(multimask_output),
            )
            
            if not multimask_output:
                # Best score mask
                idx = int(np.argmax(scores))
                m = masks[idx].astype(np.float32)
            else:
                # Union of masks
                m = masks.max(axis=0).astype(np.float32)
                
            mt = torch.from_numpy(m)[None, None].to(self.device, self.dtype)
            
            # Post-processing
            if postprocess:
                if "erode" in postprocess and postprocess["erode"] > 1:
                    mt = _erode(mt, k=int(postprocess["erode"]))
                if "dilate" in postprocess and postprocess["dilate"] > 1:
                    mt = _dilate(mt, k=int(postprocess["dilate"]))
                if "blur_ksize" in postprocess and postprocess["blur_ksize"] >= 3:
                    sig = float(postprocess.get("blur_sigma", max(0.5, postprocess["blur_ksize"] / 3.0)))
                    mt = _gaussian_blur(mt, ksize=int(postprocess["blur_ksize"]), sigma=sig)
                    
            out_masks[name] = mt.clamp(0.0, 1.0)

        return out_masks

    def predict_from_points(
        self,
        prompts: Dict[str, Dict[str, List[Tuple[int, int]]]],
        multimask_output: bool = False,
        postprocess: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict masks from points.
        prompts: {name: {"pos": [(x,y)], "neg": [(x,y)]}}
        """
        if self.predictor is None or self.image_np is None:
            raise RuntimeError("Predictor not initialized or image not set.")

        out_masks: Dict[str, torch.Tensor] = {}
        for name, pn in prompts.items():
            pos_points = pn.get("pos", [])
            neg_points = pn.get("neg", [])
            
            points_list = pos_points + neg_points
            if len(points_list) == 0:
                continue
                
            points = np.array(points_list, dtype=np.int32)
            labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=bool(multimask_output),
            )
            
            if not multimask_output:
                idx = int(np.argmax(scores))
                m = masks[idx].astype(np.float32)
            else:
                m = masks.max(axis=0).astype(np.float32)
                
            mt = torch.from_numpy(m)[None, None].to(self.device, self.dtype)
            
            if postprocess:
                if "erode" in postprocess and postprocess["erode"] > 1:
                    mt = _erode(mt, k=int(postprocess["erode"]))
                if "dilate" in postprocess and postprocess["dilate"] > 1:
                    mt = _dilate(mt, k=int(postprocess["dilate"]))
                if "blur_ksize" in postprocess and postprocess["blur_ksize"] >= 3:
                    sig = float(postprocess.get("blur_sigma", max(0.5, postprocess["blur_ksize"] / 3.0)))
                    mt = _gaussian_blur(mt, ksize=int(postprocess["blur_ksize"]), sigma=sig)
                    
            out_masks[name] = mt.clamp(0.0, 1.0)

        return out_masks

    @staticmethod
    def resize_masks(
        masks_img: Dict[str, torch.Tensor],
        size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolates masks to size (W, H).
        """
        H, W = size[1], size[0]
        out = {}
        for name, m in masks_img.items():
            mm = m
            if mm.shape[-2:] != (H, W):
                mm = F.interpolate(mm, size=(H, W), mode="bilinear", align_corners=False)
            out[name] = mm.clamp(0.0, 1.0)
        return out

    @staticmethod
    def clip_masks_to_boxes(
        masks_img: Dict[str, torch.Tensor],
        boxes: Dict[str, Tuple[int, int, int, int]],
    ) -> Dict[str, torch.Tensor]:
        """ Clips masks to safety boxes. """
        out = {}
        for name, m in masks_img.items():
            box = boxes.get(name, None)
            if box is None:
                out[name] = m
            else:
                out[name] = _clip_to_box(m, box).clamp(0.0, 1.0)
        return out

    @staticmethod
    def feather_masks(
        masks_img: Dict[str, torch.Tensor],
        radius: int = 15,
        sigma: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """ Feathers image-scale masks. """
        out = {}
        for name, m in masks_img.items():
            ksize = max(3, int(radius) | 1)
            sig = sigma if sigma and sigma > 0 else max(0.5, ksize / 3.0)
            out[name] = _gaussian_blur(m, ksize=ksize, sigma=sig).clamp(0.0, 1.0)
        return out

    @staticmethod
    def merge_masks_union(
        masks_img: Dict[str, torch.Tensor],
        names_order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """ Returns union of multiple masks. """
        keys = names_order if names_order else list(masks_img.keys())
        base = None
        for k in keys:
            if k in masks_img:
                m = masks_img[k]
                base = m.clone() if base is None else torch.max(base, m)
        return base if base is not None else None
