"""
zsrag/core/mask_manager.py
Dynamic Mask Manager for ZS-RAG (Zero-Shot Region-Aware Generation).

Features:
- Dual-scale mask management (Image & Latent).
- Softmax mutual exclusion with temperature control.
- Z-Order occlusion handling.
- Boundary feathering and safety box clipping.
- Latent fusion weight construction.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

def _gaussian_kernel(ksize: int, sigma: float, device, dtype):
    """ Generates a 2D Gaussian kernel. """
    if ksize % 2 == 0:
        ksize += 1
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    xx = ax[None, :]
    yy = ax[:, None]
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[None, None, :, :]  # [1,1,ksize,ksize]

def _gaussian_blur(m: torch.Tensor, ksize: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """ Applies Gaussian blur for feathering. """
    kernel = _gaussian_kernel(ksize, sigma, m.device, m.dtype)
    pad = ksize // 2
    # Reflect padding to avoid border artifacts
    m_padded = F.pad(m, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(m_padded, kernel)

def _dilate(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    """ Dilation using MaxPool. """
    pad = k // 2
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)

def _erode(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    """ Erosion using inverted MaxPool. """
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

def _feather(mask: torch.Tensor, radius: int, sigma: Optional[float] = None) -> torch.Tensor:
    """ Feathers mask edges. """
    if radius <= 0:
        return mask
    ksize = max(3, int(radius) | 1)
    sig = sigma if sigma and sigma > 0 else max(0.5, ksize / 3.0)
    return _gaussian_blur(mask, ksize=ksize, sigma=sig).clamp(0.0, 1.0)

def _normalize_stack(stack: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """ Normalizes a stack of masks (channels) to sum to 1. """
    denom = stack.sum(dim=0, keepdim=True) + eps
    return stack / denom

class DynamicMaskManager:
    """
    Manages masks for multi-subject generation without training.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        latent_size: Tuple[int, int],
        subject_names: List[str],
        init_masks_img: Optional[Dict[str, torch.Tensor]] = None,
        safety_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        tau: float = 0.8,
        bg_floor: float = 0.05,
        feather_radius_img: int = 15,
        feather_radius_lat: int = 3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.image_size = image_size
        self.latent_size = latent_size
        self.subject_names = list(subject_names)
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        self.tau = float(tau)
        self.bg_floor = float(bg_floor)
        self.feather_radius_img = int(feather_radius_img)
        self.feather_radius_lat = int(feather_radius_lat)

        H, W = image_size
        self.masks_img: Dict[str, torch.Tensor] = {}
        for name in subject_names:
            if init_masks_img and name in init_masks_img:
                m = init_masks_img[name].to(self.device, self.dtype)
            else:
                m = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
            self.masks_img[name] = m.clamp(0.0, 1.0)

        self.safety_boxes = safety_boxes or {name: (0, 0, W, H) for name in subject_names}

        # Z-Order: Higher index = Foreground (drawn later / on top)
        self.z_order: List[str] = list(subject_names)

        # Background masks
        self.bg_mask_img = torch.ones(1, 1, H, W, device=self.device, dtype=self.dtype)
        self.bg_mask_latent = torch.ones(1, 1, *latent_size, device=self.device, dtype=self.dtype)

        self.masks_latent: Dict[str, torch.Tensor] = {}

        self._recompute_all()

    # -----------------------
    # Public API
    # -----------------------
    def set_z_order(self, order: List[str]):
        """
        Sets occlusion priority. Later elements cover earlier ones.
        """
        valid = [n for n in order if n in self.subject_names]
        if len(valid) == len(self.subject_names):
            self.z_order = valid
            self._recompute_all()

    def set_tau(self, tau: float):
        self.tau = float(tau)
        self._recompute_all()

    def set_bg_floor(self, bg_floor: float):
        self.bg_floor = float(bg_floor)
        self._recompute_all()

    def set_feather_radii(self, img_radius: int, lat_radius: int):
        self.feather_radius_img = int(img_radius)
        self.feather_radius_lat = int(lat_radius)
        self._recompute_all()

    def replace_masks_img(self, new_masks: Dict[str, torch.Tensor]):
        """
        Replaces current image-scale masks with external masks.
        """
        H, W = self.image_size
        for name, m in new_masks.items():
            if name in self.masks_img:
                mm = m.to(self.device, self.dtype)
                if mm.shape[-2:] != (H, W):
                    mm = F.interpolate(mm, size=(H, W), mode="bilinear", align_corners=False)
                self.masks_img[name] = mm.clamp(0.0, 1.0)
        self._recompute_all()

    def get_masks_img(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.masks_img.items()}

    def get_masks_latent(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return ({k: v.clone() for k, v in self.masks_latent.items()}, self.bg_mask_latent.clone())

    def get_feathered_masks_latent(self, radius: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        r = self.feather_radius_lat if radius is None else int(radius)
        masks_lat_f = {k: _feather(v, r) for k, v in self.masks_latent.items()}
        bg_lat_f = _feather(self.bg_mask_latent, r)
        return masks_lat_f, bg_lat_f

    def update_from_attn(self, attn_maps_img: Dict[str, torch.Tensor], beta: float = 0.6, thresh: float = 0.5, erode_k: int = 5, blur_ksize: int = 15, blur_sigma: float = 2.5):
        """
        Updates masks based on Cross-Attention heatmaps.
        Applies thresholding, erosion, and smoothing before blending.
        """
        H, W = self.image_size
        for name, amap in attn_maps_img.items():
            if name not in self.masks_img:
                continue
            m_prev = self.masks_img[name]
            a = amap.to(self.device, self.dtype)
            
            if a.shape[-2:] != (H, W):
                a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
            
            if a.max() > 1e-6:
                a = a / a.max()
            
            a_bin = (a > float(thresh)).to(self.dtype)
            a_erode = _erode(a_bin, k=erode_k)
            a_smooth = _gaussian_blur(a_erode, ksize=blur_ksize, sigma=blur_sigma)
            
            box = self.safety_boxes.get(name, (0, 0, W, H))
            a_boxed = _clip_to_box(a_smooth, box)
            
            m_new = beta * a_boxed + (1.0 - beta) * m_prev
            self.masks_img[name] = m_new.clamp(0.0, 1.0)

        self._recompute_all()

    def build_latent_fusion_weights(self, feather_radius: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Constructs normalized weights for latent space fusion.
        """
        r = self.feather_radius_lat if feather_radius is None else int(feather_radius)
        masks_lat_f, bg_lat_f = self.get_feathered_masks_latent(r)
        
        stack = torch.stack([masks_lat_f[n] for n in self.subject_names] + [bg_lat_f], dim=0)
        stack = _normalize_stack(stack)
        
        w_list = stack[:-1]
        w_bg = stack[-1]
        
        weights = {name: w_list[i] for i, name in enumerate(self.subject_names)}
        return weights, w_bg

    def energy_minimize_blend(self, eps_bg: torch.Tensor, eps_subjects: Dict[str, torch.Tensor], kappa: float = 1.0) -> torch.Tensor:
        """
        Performs "Energy Minimization" blending in latent space.
        Applies extra smoothing to blending weights to avoid hard seams.
        """
        weights, w_bg = self.build_latent_fusion_weights()
        
        def smooth_w(w: torch.Tensor):
            return _feather(w, radius=max(1, self.feather_radius_lat)).clamp(0.0, 1.0)

        w_bg_s = smooth_w(w_bg)
        eps_final = eps_bg * w_bg_s
        
        for name, eps_s in eps_subjects.items():
            w = smooth_w(weights[name])
            diff = eps_s - eps_bg
            eps_final = eps_final + kappa * w * diff

        # Normalize final combination
        stack_w = torch.stack([weights[n] for n in self.subject_names] + [w_bg_s], dim=0)
        stack_w = _normalize_stack(stack_w)
        
        eps_out = torch.zeros_like(eps_bg)
        for i, name in enumerate(self.subject_names):
            eps_out = eps_out + stack_w[i] * eps_subjects[name]
        
        eps_out = eps_out + stack_w[-1] * eps_bg
        return eps_out

    # -----------------------
    # Internal Logic
    # -----------------------
    def _recompute_all(self):
        """
        统一重算：
        - 安全框裁剪
        - z-order 去重叠
        - 温度 softmax 前景互斥 + 背景底权重与归一化
        - 羽化（图像尺度）并下采样到 latent 尺度
        """
        H, W = self.image_size
        
        # 1) 应用安全框裁剪 (保持不变)
        for name in self.subject_names:
            box = self.safety_boxes.get(name, (0, 0, W, H))
            self.masks_img[name] = _clip_to_box(self.masks_img[name], box).clamp(0.0, 1.0)

        # 2) Z-Order 去重叠 (保持不变)
        occupied = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
        masks_img_no_overlap: Dict[str, torch.Tensor] = {}
        for name in self.z_order:
            m = self.masks_img[name].clamp(0.0, 1.0)
            effective = m * (1.0 - occupied)
            masks_img_no_overlap[name] = effective
            occupied = (occupied + effective).clamp(0.0, 1.0)
        self.masks_img = masks_img_no_overlap

        # 3) 前景互斥
        if len(self.subject_names) > 0:
            stack = torch.cat([self.masks_img[name] for name in self.subject_names], dim=0)
            stack = _feather(stack, radius=self.feather_radius_img)
            fg_sharp = torch.softmax(stack / max(self.tau, 1e-6), dim=0)
            
            for i, name in enumerate(self.subject_names):
                self.masks_img[name] = fg_sharp[i:i+1]
            
            # 【修复点】确保维度是 [1, 1, H, W]
            sum_fg = fg_sharp.sum(dim=0, keepdim=True) 
            # 注意：如果 dim=0 是 N，keepdim=True 会得到 [1, 1, H, W] 吗？
            # 假设 fg_sharp 是 [N, 1, H, W]。
            # sum(dim=0, keepdim=True) -> [1, 1, H, W]。是对的。
            # 之前的 sum(dim=0) 得到了 [1, H, W]。
        else:
            sum_fg = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)

        # 4) 背景计算与归一化 (保持不变)
        w_bg = (1.0 - sum_fg).clamp(0.0, 1.0) + self.bg_floor
        denom = w_bg + sum_fg + 1e-6
        self.bg_mask_img = w_bg / denom
        for name in self.subject_names:
            self.masks_img[name] = self.masks_img[name] / denom

        # 5) 下采样到 latent 尺度 (保持不变)
        H_lat, W_lat = self.latent_size
        self.masks_latent = {}
        for name in self.subject_names:
            self.masks_latent[name] = F.interpolate(self.masks_img[name], size=(H_lat, W_lat), mode="bilinear", align_corners=False)
        self.bg_mask_latent = F.interpolate(self.bg_mask_img, size=(H_lat, W_lat), mode="bilinear", align_corners=False)


    # -----------------------
    # Factories
    # -----------------------
    @staticmethod
    def init_from_positions(
        image_size: Tuple[int, int],
        latent_size: Tuple[int, int],
        subjects: List[Dict[str, str]],
        safety_expand: float = 0.05,
        tau: float = 0.8,
        bg_floor: float = 0.05,
        gap_ratio: float = 0.06,
        feather_radius_img: int = 15,
        feather_radius_lat: int = 3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize masks from positional keywords (left/right/top/bottom/center).
        """
        H, W = image_size
        dev = device or torch.device("cpu")
        dt = dtype or torch.float32

        def _make_positional_mask(position: str, sharpness: float = 6.0) -> torch.Tensor:
            yy = torch.linspace(0, 1, steps=H, device=dev, dtype=dt).unsqueeze(1).repeat(1, W)
            xx = torch.linspace(0, 1, steps=W, device=dev, dtype=dt).unsqueeze(0).repeat(H, 1)
            pos = position.lower()
            if "left" in pos:
                mask = torch.sigmoid((0.5 - xx) * sharpness)
            elif "right" in pos:
                mask = torch.sigmoid((xx - 0.5) * sharpness)
            elif "top" in pos:
                mask = torch.sigmoid((0.5 - yy) * sharpness)
            elif "bottom" in pos:
                mask = torch.sigmoid((yy - 0.5) * sharpness)
            else:
                mask = torch.ones(H, W, device=dev, dtype=dt)
            return mask[None, None]

        init_masks = {}
        boxes = {}
        
        gap_w = max(2, int(W * gap_ratio))
        center_l, center_r = W // 2 - gap_w // 2, W // 2 + gap_w // 2

        for subj in subjects:
            name = subj["name"]
            position = subj.get("position", "center")
            m = _make_positional_mask(position, sharpness=6.0)
            
            if "left" in position or "right" in position:
                m[..., :, center_l:center_r] = 0.0
                
            m = _gaussian_blur(m, ksize=31, sigma=6.0)
            init_masks[name] = m.clamp(0.0, 1.0)

            x1, y1, x2, y2 = 0, 0, W, H
            if "left" in position:
                x2 = int(0.5 * W)
            elif "right" in position:
                x1 = int(0.5 * W)
            
            w_box, h_box = x2 - x1, y2 - y1
            ex, ey = int(w_box * safety_expand), int(h_box * safety_expand)
            boxes[name] = (max(0, x1 - ex), max(0, y1 - ey), min(W, x2 + ex), min(H, y2 + ey))

        return DynamicMaskManager(
            image_size=image_size,
            latent_size=latent_size,
            subject_names=[s["name"] for s in subjects],
            init_masks_img=init_masks,
            safety_boxes=boxes,
            tau=tau,
            bg_floor=bg_floor,
            feather_radius_img=feather_radius_img,
            feather_radius_lat=feather_radius_lat,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def init_from_external_masks(
        image_size: Tuple[int, int],
        latent_size: Tuple[int, int],
        masks_img: Dict[str, torch.Tensor],
        safety_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        tau: float = 0.8,
        bg_floor: float = 0.05,
        feather_radius_img: int = 15,
        feather_radius_lat: int = 3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        z_order: Optional[List[str]] = None,
    ):
        """
        Initialize from external binary masks (e.g. SAM output).
        """
        H, W = image_size
        names = list(masks_img.keys())
        init_masks = {}
        for name, m in masks_img.items():
            mm = m.to(device or torch.device("cpu"), dtype or torch.float32)
            if mm.shape[-2:] != (H, W):
                mm = F.interpolate(mm, size=(H, W), mode="bilinear", align_corners=False)
            init_masks[name] = mm.clamp(0.0, 1.0)

        mgr = DynamicMaskManager(
            image_size=image_size,
            latent_size=latent_size,
            subject_names=names,
            init_masks_img=init_masks,
            safety_boxes=safety_boxes,
            tau=tau,
            bg_floor=bg_floor,
            feather_radius_img=feather_radius_img,
            feather_radius_lat=feather_radius_lat,
            device=device,
            dtype=dtype,
        )
        if z_order:
            mgr.set_z_order(z_order)
        return mgr
