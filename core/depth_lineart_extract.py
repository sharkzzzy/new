"""
zsrag/core/depth_lineart_extract.py
Extracts geometric control signals (Lineart & Depth) from images.
Prioritizes 'controlnet_aux' for high-quality extraction.
Fallback to OpenCV/Canny for basic edge detection.
"""

import torch
from typing import Optional, Tuple, Union, List

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

# Optional: controlnet_aux (Recommended)
try:
    from controlnet_aux import CannyDetector, LineartDetector, ZoeDetector, MidasDetector
    _HAS_CNAUX = True
except ImportError:
    _HAS_CNAUX = False

# Optional: OpenCV
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Optional: Matplotlib (for depth colormap)
try:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt # safe import
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _ensure_basic_deps():
    missing = []
    if not _HAS_PIL: missing.append("Pillow")
    if not _HAS_TV: missing.append("torchvision")
    if missing:
        raise ImportError(f"depth_lineart_extract requires: {', '.join(missing)}")

def _to_pil(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    """ Converts tensor or PIL to PIL Image. """
    _ensure_basic_deps()
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim != 3 or img.shape[0] < 1:
            raise ValueError("Expected tensor image of shape [C,H,W] where C>=1.")
        
        if img.min() < 0.0:
            img = (img + 1.0) / 2.0
        
        img = img.clamp(0.0, 1.0)
        
        if img.shape[0] == 4:
            img = img[:3]
            
        return TF.to_pil_image((img * 255.0).round().to(torch.uint8))
    raise ValueError("image must be PIL.Image or torch.Tensor.")

def _resize_pil(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """ Resizes PIL image to (W, H). """
    w, h = int(size[0]), int(size[1])
    return img.resize((w, h), resample=Image.BICUBIC)

def _pil_to_np_uint8(img: Image.Image) -> np.ndarray:
    """ PIL.Image -> numpy uint8 HxWxC """
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """ numpy uint8 HxWxC -> PIL.Image """
    arr = np.ascontiguousarray(arr)
    return Image.fromarray(arr)

def extract_canny(
    image: Union[torch.Tensor, Image.Image],
    target_size: Tuple[int, int],
    low_threshold: int = 100,
    high_threshold: int = 200,
    aperture_size: int = 3,
) -> Image.Image:
    """
    Extracts Canny edges.
    Priority: controlnet_aux -> cv2 -> numpy gradient fallback.
    """
    _ensure_basic_deps()
    pil = _to_pil(image)
    pil = _resize_pil(pil, target_size)

    if _HAS_CNAUX:
        # controlnet_aux CannyDetector
        try:
            detector = CannyDetector()
            # detector expects np uint8 HWC
            arr = detector(_pil_to_np_uint8(pil))
            
            # Ensure 3 channels
            if isinstance(arr, Image.Image):
                arr = np.array(arr)
            
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return _np_to_pil(arr)
        except Exception as e:
            print(f"[CannyDetector] Failed: {e}. Falling back to OpenCV.")
    
    if _HAS_CV2:
        arr = _pil_to_np_uint8(pil)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold, apertureSize=aperture_size)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        return _np_to_pil(edges_rgb)
    else:
        # Fallback: Simple gradient magnitude
        arr = _pil_to_np_uint8(pil).astype(np.float32)
        gray = arr.mean(axis=-1)
        
        # Calculate gradients manually
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = (mag / (mag.max() + 1e-6) * 255.0).astype(np.uint8)
        edges_rgb = np.stack([mag] * 3, axis=-1)
        return _np_to_pil(edges_rgb)

def extract_lineart(
    image: Union[torch.Tensor, Image.Image],
    target_size: Tuple[int, int],
    coarse: bool = False,
) -> Image.Image:
    """
    Extracts Lineart using controlnet_aux.
    """
    _ensure_basic_deps()
    pil = _to_pil(image)
    pil = _resize_pil(pil, target_size)

    if _HAS_CNAUX:
        try:
            # Note: LineartDetector download model on first init
            detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
            # The API might differ slightly across versions, usually __call__ works
            arr = detector(pil, coarse=bool(coarse))
            
            # If returns PIL
            if isinstance(arr, Image.Image):
                return arr
            
            # If returns numpy
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return _np_to_pil(arr)
        except Exception as e:
            print(f"[LineartDetector] Failed, fallback to Canny: {e}")
            return extract_canny(pil, target_size)
    else:
        return extract_canny(pil, target_size)

def _colorize_depth_gray(depth: np.ndarray) -> np.ndarray:
    """
    Colorizes normalized depth map (0..1) to RGB using 'turbo' or 'inferno' colormap.
    """
    d = np.clip(depth, 0.0, 1.0)
    if _HAS_MPL:
        # Use new API if available, fallback to old
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps['turbo']
        except Exception:
            cmap = cm.get_cmap("turbo")
            
        rgb = cmap(d)[:, :, :3] # Drop alpha
        rgb255 = (rgb * 255.0).round().astype(np.uint8)
        return rgb255
    else:
        # Grayscale replication
        g = (d * 255.0).round().astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

def extract_depth(
    image: Union[torch.Tensor, Image.Image],
    target_size: Tuple[int, int],
    method: str = "zoe", # "zoe" or "midas"
    normalize: bool = True,
) -> Image.Image:
    """
    Extracts depth map using ZoeDepth or MiDaS.
    """
    _ensure_basic_deps()
    pil = _to_pil(image)
    pil = _resize_pil(pil, target_size)

    if not _HAS_CNAUX:
        raise ImportError("controlnet_aux is required for depth extraction (Zoe/MiDaS). Please install controlnet-aux.")

    if method.lower() == "zoe":
        try:
            detector = ZoeDetector.from_pretrained("lllyasviel/Annotators")
            depth = detector(pil)
        except Exception as e:
            print(f"[ZoeDetector] Failed, fallback to MiDaS: {e}")
            detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            depth = detector(pil)
    else:
        detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        depth = detector(pil)

    # Convert to numpy for normalization and colorization
    depth_np = np.array(depth).astype(np.float32)
    
    # If already RGB (some detectors return colorized), convert to gray
    if depth_np.ndim == 3:
        depth_np = depth_np.mean(axis=-1)

    if normalize:
        mn, mx = float(depth_np.min()), float(depth_np.max())
        depth_np = (depth_np - mn) / (mx - mn + 1e-6)

    rgb = _colorize_depth_gray(depth_np)
    return _np_to_pil(rgb)


class ControlSignalExtractor:
    """
    Wrapper for extracting control signals.
    """
    def __init__(self, target_size: Tuple[int, int]):
        _ensure_basic_deps()
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.image: Optional[Image.Image] = None
        self.lineart_img: Optional[Image.Image] = None
        self.depth_img: Optional[Image.Image] = None

    def set_image(self, image: Union[torch.Tensor, Image.Image]):
        self.image = _to_pil(image)

    def compute_lineart(self, coarse: bool = False):
        if self.image is None:
            raise ValueError("image not set. Call set_image(image) first.")
        self.lineart_img = extract_lineart(self.image, self.target_size, coarse=coarse)
        return self.lineart_img

    def compute_canny(self, low_threshold: int = 100, high_threshold: int = 200, aperture_size: int = 3):
        if self.image is None:
            raise ValueError("image not set. Call set_image(image) first.")
        self.lineart_img = extract_canny(self.image, self.target_size, low_threshold, high_threshold, aperture_size)
        return self.lineart_img

    def compute_depth(self, method: str = "zoe", normalize: bool = True):
        if self.image is None:
            raise ValueError("image not set. Call set_image(image) first.")
        self.depth_img = extract_depth(self.image, self.target_size, method=method, normalize=normalize)
        return self.depth_img

    def get_control_images(self) -> List[Image.Image]:
        """
        Returns list of control images [Lineart, Depth] if they exist.
        """
        outs: List[Image.Image] = []
        if self.lineart_img is not None:
            outs.append(self.lineart_img)
        if self.depth_img is not None:
            outs.append(self.depth_img)
        return outs
