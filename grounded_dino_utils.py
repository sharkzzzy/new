"""
zsrag/core/grounded_dino_utils.py
Wrapper for Grounded-DINO text-guided detection.
Handles model loading, image preprocessing, prediction, and post-processing (NMS, filtering).
"""

import os
from typing import Dict, List, Tuple, Optional, Union

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

# GroundingDINO Dependency
try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    _HAS_GDINO = True
except ImportError:
    _HAS_GDINO = False

# NMS Dependency
try:
    from torchvision.ops import nms as tv_nms
    _HAS_NMS = True
except ImportError:
    _HAS_NMS = False

# HF Hub Dependency
try:
    from huggingface_hub import hf_hub_download
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


def _ensure_deps():
    missing = []
    if not _HAS_PIL: missing.append("Pillow")
    if not _HAS_TV: missing.append("torchvision")
    if not _HAS_GDINO: missing.append("groundingdino")
    
    if missing:
        raise ImportError(f"grounded_dino_utils requires: {', '.join(missing)}")

def _to_pil(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim != 3 or img.shape[0] < 1:
            raise ValueError("Expected tensor image [C,H,W].")
        
        if img.min() < 0.0:
            img = (img + 1.0) / 2.0
        
        img = img.clamp(0.0, 1.0)
        
        if img.shape[0] == 4:
             img = img[:3]
             
        return TF.to_pil_image((img * 255.0).round().to(torch.uint8))
    raise ValueError("image must be PIL.Image or torch.Tensor.")

def _format_text_prompt(names: List[str], punct: str = ".") -> str:
    """ Formats text prompt for Grounded-DINO (e.g. "cat . dog ."). """
    toks = []
    for n in names:
        n2 = n.strip()
        if not n2: continue
        # GroundingDINO usually expects lowercase
        toks.append(f"{n2.lower()} {punct}")
    return " ".join(toks)

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """ Converts normalized cxcywh to normalized xyxy. """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> List[int]:
    """ NMS with fallback. """
    if _HAS_NMS:
        keep = tv_nms(boxes_xyxy, scores, iou_thresh)
        return keep.tolist()
    
    # Fallback: Greedy NMS
    order = scores.argsort(descending=True)
    keep = []
    
    def iou(box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0.0, (xa2 - xa1) * (ya2 - ya1))
        area_b = max(0.0, (xb2 - xb1) * (yb2 - yb1))
        union = area_a + area_b - inter + 1e-6
        return inter / union

    selected = []
    boxes_list = boxes_xyxy.tolist()
    for idx in order.tolist():
        b = boxes_list[idx]
        if all(iou(b, boxes_list[j]) < iou_thresh for j in selected):
            selected.append(idx)
    return selected

def _rescale_boxes(boxes_xyxy: torch.Tensor, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> torch.Tensor:
    """ Rescales boxes from src_size to dst_size. """
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    out = boxes_xyxy.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * scale_x
    out[:, [1, 3]] = out[:, [1, 3]] * scale_y
    return out

def _clip_boxes_to_image(boxes_xyxy: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    w, h = img_size
    boxes = boxes_xyxy.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w)
    boxes[:, 2] = boxes[:, 2].clamp(0, w)
    boxes[:, 1] = boxes[:, 1].clamp(0, h)
    boxes[:, 3] = boxes[:, 3].clamp(0, h)
    return boxes


class GroundedDINODetector:
    """
    Wrapper for Grounded-DINO Detection.
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cfg_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        auto_download: bool = True,
        repo_id: str = "IDEA-Research/GroundingDINO",
        cfg_filename: str = "GroundingDINO_SwinT_OGC.py",
        weights_filename: str = "groundingdino_swint_ogc.pth",
    ):
        if not _HAS_GDINO:
            raise ImportError("groundingdino is not installed. Please install it from https://github.com/IDEA-Research/GroundingDINO")
            
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or torch.float32 # GDINO usually FP32
        
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.auto_download = auto_download
        self.repo_id = repo_id
        self.cfg_filename = cfg_filename
        self.weights_filename = weights_filename

        self.model = None
        self._load_model()

    def _resolve_file(self, local_path: Optional[str], filename: str) -> str:
        if local_path and os.path.exists(local_path):
            return local_path
        if not self.auto_download or not _HAS_HF:
            raise FileNotFoundError(f"File {filename} not found and auto_download disabled or huggingface_hub missing.")
            
        # Try to download config/weights from HF Hub
        # Note: The official repo ID might differ or require token.
        # "ShilongLiu/GroundingDINO" is a common mirror for HF.
        try:
            return hf_hub_download(repo_id="ShilongLiu/GroundingDINO", filename=filename)
        except Exception:
             # Fallback to provided repo_id
            return hf_hub_download(repo_id=self.repo_id, filename=filename)

    def _load_model(self):
        cfg = self._resolve_file(self.cfg_path, self.cfg_filename)
        ckpt = self._resolve_file(self.weights_path, self.weights_filename)
        self.model = load_model(cfg, ckpt, device=self.device)

    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Applies GroundingDINO transforms: Resize, ToTensor, Normalize.
        """
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_image, None)
        return image_transformed.to(self.device)

    @torch.no_grad()
    def detect(
        self,
        image: Union[Image.Image, torch.Tensor],
        names: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        nms_iou: float = 0.5,
        topk_per_class: int = 1,
        resize_for_detector: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detects objects by text prompt.
        """
        _ensure_deps()
        pil = _to_pil(image)
        img_w, img_h = pil.width, pil.height

        # Optional resize for input (though GDINO transform handles resize too)
        pil_det = pil
        if resize_for_detector is not None:
             pil_det = pil.resize(resize_for_detector, resample=Image.BICUBIC)

        # Prepare input tensor
        img_tensor = self._preprocess_image(pil_det)

        # Format Text
        text_prompt = _format_text_prompt(names, punct=".")

        # Run Prediction
        # boxes: [N, 4] normalized cxcywh
        # logits: [N]
        # phrases: [N]
        boxes, logits, phrases = predict(
            model=self.model,
            image=img_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        if boxes.shape[0] == 0:
            return {name: [] for name in names}

        # Convert cxcywh -> xyxy (normalized)
        boxes_xyxy_norm = _cxcywh_to_xyxy(boxes)
        
        # Scale to Original Image Size
        boxes_img = boxes_xyxy_norm * torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)

        # NMS
        keep_idx = _nms(boxes_img, logits, iou_thresh=nms_iou)
        boxes_img = boxes_img[keep_idx]
        scores_t = logits[keep_idx]
        phrases_kept = [phrases[i] for i in keep_idx]

        boxes_img = _clip_boxes_to_image(boxes_img, img_size=(img_w, img_h))

        # Filter by name
        out: Dict[str, List[Tuple[int, int, int, int]]] = {name: [] for name in names}
        phrases_lower = [p.lower() for p in phrases_kept]
        
        for name in names:
            nm = name.lower()
            # Simple string containment matching
            idxs = [i for i, p in enumerate(phrases_lower) if nm in p]
            
            if len(idxs) == 0:
                continue
                
            # Top-K
            idxs_sorted = sorted(idxs, key=lambda i: float(scores_t[i].item()), reverse=True)
            idxs_top = idxs_sorted[:max(1, int(topk_per_class))]
            
            boxes_class = []
            for i in idxs_top:
                x1, y1, x2, y2 = boxes_img[i].round().int().tolist()
                boxes_class.append((x1, y1, x2, y2))
            out[name] = boxes_class

        return out


def boxes_from_positions(
    image_size: Tuple[int, int],
    subjects: List[Dict[str, str]],
    gap_ratio: float = 0.08,
    safety_expand: float = 0.05,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Fallback: Generate boxes based on position keywords.
    """
    W, H = int(image_size[0]), int(image_size[1])
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    
    gap_w = max(2, int(W * gap_ratio))

    for subj in subjects:
        name = subj["name"]
        pos = subj.get("position", "center").lower()
        
        x1, y1, x2, y2 = 0, 0, W, H
        
        if "left" in pos:
            x2 = int(0.5 * W)
            x2 = max(x2 - gap_w // 2, 0)
        elif "right" in pos:
            x1 = int(0.5 * W)
            x1 = min(x1 + gap_w // 2, W)
        elif "top" in pos:
            y2 = int(0.5 * H)
        elif "bottom" in pos:
            y1 = int(0.5 * H)
            
        # Safety Expand
        w_box, h_box = x2 - x1, y2 - y1
        ex, ey = int(w_box * safety_expand), int(h_box * safety_expand)
        
        x1 = max(0, x1 - ex)
        y1 = max(0, y1 - ey)
        x2 = min(W, x2 + ex)
        y2 = min(H, y2 + ey)
        
        boxes[name] = (x1, y1, x2, y2)
        
    return boxes
