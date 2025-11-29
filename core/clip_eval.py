"""
zsrag/core/clip_eval.py
Region-aware attribute consistency evaluation and parameter suggestion.

Features:
- Zero-shot region scoring using pretrained CLIP/BLIP.
- Supports Masked Crop and BBox Crop modes.
- Color consistency metrics.
- Feedback loop: Proposes CFG/IP-Adapter weight updates based on scores.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

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

try:
    import torchvision.transforms.functional as TF
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


@dataclass
class RegionScore:
    name: str
    text: str
    score: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    area: int = 0

@dataclass
class EvalResult:
    per_region: List[RegionScore]
    avg_score: float
    min_score: float
    max_score: float
    details: Dict[str, Any]

def _ensure_dependencies():
    missing = []
    if not _HAS_TRANSFORMERS: missing.append("transformers")
    if not _HAS_PIL: missing.append("Pillow")
    if not _HAS_TORCHVISION: missing.append("torchvision")
    if missing:
        raise ImportError(f"clip_eval requires: {', '.join(missing)}. Please install them.")

def _to_uint8_image(t: torch.Tensor) -> Image.Image:
    """
    Converts [C,H,W] tensor (0..1 or -1..1) to PIL Image.
    """
    if t.ndim != 3 or t.shape[0] not in (3, 4):
        raise ValueError("Expected tensor of shape [3,H,W] or [4,H,W].")
    
    img = t.detach().cpu()
    if img.min() < 0.0:
        img = (img + 1.0) / 2.0
    
    img = img.clamp(0.0, 1.0)
    if img.shape[0] == 4:
        img = img[:3]
    
    img = (img * 255.0).round().to(torch.uint8)
    return TF.to_pil_image(img)

def _mask_to_bbox(mask: torch.Tensor, min_area: int = 64, dilate_k: int = 7) -> Optional[Tuple[int, int, int, int]]:
    """
    Converts binary mask to bounding box (xyxy).
    Supports dilation to include boundary context.
    """
    if mask.ndim == 4:
        m = mask[0, 0]
    elif mask.ndim == 2:
        m = mask
    else:
        raise ValueError("mask must be [1,1,H,W] or [H,W].")
        
    H, W = m.shape
    m = m.float()
    
    if dilate_k and dilate_k > 1:
        pad = dilate_k // 2
        m = F.max_pool2d(m[None, None], kernel_size=dilate_k, stride=1, padding=pad)[0, 0]
    
    ys, xs = torch.where(m > 0.5)
    if ys.numel() == 0:
        return None
        
    y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
    x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
    
    area = (y2 - y1) * (x2 - x1)
    if area < min_area:
        return None
        
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return (x1, y1, x2, y2)

def _crop_by_bbox(image: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    """ Crops [C,H,W] image tensor by bbox (xyxy). """
    C, H, W = image.shape
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        # Fallback to full image if bbox is invalid
        return image
        
    return image[:, y1:y2, x1:x2]

def _apply_mask(image: torch.Tensor, mask: torch.Tensor, background: float = 0.0) -> torch.Tensor:
    """ Applies mask to image, setting background to constant value. """
    if mask.ndim == 4:
        m = mask[0, 0]
    else:
        m = mask
    m = m.float().clamp(0.0, 1.0)
    return image * m + background * (1.0 - m)


class CLIPRegionEvaluator:
    """
    Zero-shot region-level evaluator using CLIP.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[torch.device] = None,
        max_side: int = 512,
        crop_mode: str = "mask",  # "mask" or "bbox"
        image_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
    ):
        _ensure_dependencies()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.max_side = int(max_side)
        self.crop_mode = crop_mode
        self.image_mean_std = image_mean_std

    def _resize_pil(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        s = max(H, W)
        if s <= self.max_side:
            return img
        scale = self.max_side / float(s)
        new_w, new_h = int(W * scale), int(H * scale)
        return img.resize((new_w, new_h), resample=Image.BICUBIC)

    def _prepare_region_image(self, image: torch.Tensor, mask: Optional[torch.Tensor], bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
        """
        Prepares a single PIL image for a region.
        """
        C, H, W = image.shape
        if self.crop_mode == "mask":
            if mask is None:
                raise ValueError("mask is required in crop_mode='mask'.")
            
            # Zero out background
            img_masked = _apply_mask(image, mask, background=0.0)
            
            # Crop to bounding box of mask for better resolution
            bbox2 = _mask_to_bbox(mask)
            if bbox2 is None:
                img_crop = image
            else:
                img_crop = _crop_by_bbox(img_masked, bbox2)
                
            pil = _to_uint8_image(img_crop)
            pil = self._resize_pil(pil)
            return pil
            
        elif self.crop_mode == "bbox":
            if bbox is None:
                if mask is None:
                    raise ValueError("bbox or mask required for crop_mode='bbox'.")
                bbox = _mask_to_bbox(mask)
                if bbox is None:
                    bbox = (0, 0, W, H)
                    
            img_crop = _crop_by_bbox(image, bbox)
            pil = _to_uint8_image(img_crop)
            pil = self._resize_pil(pil)
            return pil
        else:
            raise ValueError("Invalid crop_mode. Use 'mask' or 'bbox'.")

    @torch.no_grad()
    def score_regions(
        self,
        image: torch.Tensor,
        texts: List[str],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        bboxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        names: Optional[List[str]] = None,
    ) -> EvalResult:
        """
        Scores multiple regions in a batch.
        """
        if names is None:
            names = [f"subj_{i}" for i in range(len(texts))]
        if len(texts) != len(names):
            raise ValueError("texts and names length mismatch.")

        region_pils = []
        region_names = []
        region_bboxes = []
        areas = []
        
        C, H, W = image.shape
        
        # Prepare Images
        for i, name in enumerate(names):
            mask_i = masks[name] if masks and name in masks else None
            bbox_i = bboxes[name] if bboxes and name in bboxes else None
            
            pil = self._prepare_region_image(image, mask_i, bbox_i)
            region_pils.append(pil)
            region_names.append(name)
            
            # Track BBox for metadata
            if bbox_i is None and mask_i is not None:
                bbox_i2 = _mask_to_bbox(mask_i)
            else:
                bbox_i2 = bbox_i
            region_bboxes.append(bbox_i2)
            
            if bbox_i2 is not None:
                x1, y1, x2, y2 = bbox_i2
                areas.append(max(0, (x2 - x1) * (y2 - y1)))
            else:
                areas.append(H * W)

        # Encode Texts
        inputs_text = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs_text = {k: v.to(self.device) for k, v in inputs_text.items()}
        text_embeds = self.model.get_text_features(**inputs_text)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Encode Images (One by one to support different sizes if not resized to same batch)
        # Here we loop because `region_pils` might vary slightly or if batching is complex.
        # But `processor` usually handles batch of PILs. Let's try batching if sizes are uniform.
        # Since we resize to max_side, aspect ratios might differ. Safest is loop.
        
        region_scores: List[RegionScore] = []
        for i, name in enumerate(region_names):
            inputs_img = self.processor(images=region_pils[i], return_tensors="pt")
            inputs_img = {k: v.to(self.device) for k, v in inputs_img.items()}
            
            img_embeds = self.model.get_image_features(**inputs_img)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine Sim
            sim = (img_embeds @ text_embeds[i].unsqueeze(1)).squeeze().item()
            
            region_scores.append(
                RegionScore(
                    name=name,
                    text=texts[i],
                    score=float(sim),
                    bbox=region_bboxes[i],
                    area=int(areas[i]),
                )
            )

        scores = [rs.score for rs in region_scores]
        avg_score = float(sum(scores) / max(len(scores), 1))
        min_score = float(min(scores) if scores else 0.0)
        max_score = float(max(scores) if scores else 0.0)
        
        details = {
            "names": region_names,
            "texts": texts,
            "bboxes": region_bboxes,
            "areas": areas,
        }
        return EvalResult(per_region=region_scores, avg_score=avg_score, min_score=min_score, max_score=max_score, details=details)

    @torch.no_grad()
    def score_single(
        self,
        image: torch.Tensor,
        text: str,
        mask: Optional[torch.Tensor] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """ Helper for single region scoring. """
        pil = self._prepare_region_image(image, mask, bbox)
        
        inputs_text = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs_img = self.processor(images=pil, return_tensors="pt")
        
        inputs_text = {k: v.to(self.device) for k, v in inputs_text.items()}
        inputs_img = {k: v.to(self.device) for k, v in inputs_img.items()}
        
        text_embeds = self.model.get_text_features(**inputs_text)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        img_embeds = self.model.get_image_features(**inputs_img)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        
        sim = (img_embeds @ text_embeds.T).squeeze().item()
        return float(sim)


class BLIPRegionEvaluator:
    """
    Optional BLIP-CLIP scorer. Better for caption alignment than standard CLIP.
    """
    def __init__(self, model_name: str = "Salesforce/blip-itm-large-coco", device: Optional[torch.device] = None):
        if not _HAS_TRANSFORMERS or not _HAS_PIL or not _HAS_TORCHVISION:
            raise ImportError("BLIPRegionEvaluator requires transformers, Pillow, torchvision.")
        
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)

    def _prepare_region_image(self, image: torch.Tensor, mask: Optional[torch.Tensor], bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
        # Reuses logic similar to CLIP evaluator
        if mask is not None:
            img_masked = _apply_mask(image, mask, background=0.0)
            bbox2 = _mask_to_bbox(mask)
            img_crop = _crop_by_bbox(img_masked, bbox2) if bbox2 else image
        else:
            img_crop = _crop_by_bbox(image, bbox) if bbox else image
        return _to_uint8_image(img_crop)

    @torch.no_grad()
    def score_single(self, image: torch.Tensor, text: str, mask: Optional[torch.Tensor] = None, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
        pil = self._prepare_region_image(image, mask, bbox)
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Use ITM score (logits)
        score = float(outputs.logits_per_image.squeeze().item())
        return score

    @torch.no_grad()
    def score_regions(self, image: torch.Tensor, texts: List[str], masks: Optional[Dict[str, torch.Tensor]] = None, bboxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None, names: Optional[List[str]] = None) -> EvalResult:
        if names is None:
            names = [f"subj_{i}" for i in range(len(texts))]
        
        region_scores = []
        for i, name in enumerate(names):
            mask_i = masks[name] if masks and name in masks else None
            bbox_i = bboxes[name] if bboxes and name in bboxes else None
            score = self.score_single(image, texts[i], mask_i, bbox_i)
            
            bbox2 = bbox_i if bbox_i is not None else (_mask_to_bbox(mask_i) if mask_i is not None else None)
            area = 0
            if bbox2 is not None:
                x1, y1, x2, y2 = bbox2
                area = max(0, (x2 - x1) * (y2 - y1))
            
            region_scores.append(RegionScore(name=name, text=texts[i], score=score, bbox=bbox2, area=area))
            
        scores = [rs.score for rs in region_scores]
        avg_score = float(sum(scores) / max(len(scores), 1))
        min_score = float(min(scores) if scores else 0.0)
        max_score = float(max(scores) if scores else 0.0)
        
        details = {"names": names, "texts": texts}
        return EvalResult(per_region=region_scores, avg_score=avg_score, min_score=min_score, max_score=max_score, details=details)


def region_mean_rgb(image: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float, float]:
    """ Calculates mean RGB in the masked region. """
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("image must be [3,H,W].")
    
    if mask.ndim == 4:
        m = mask[0, 0]
    else:
        m = mask
        
    img = image[:3]
    if img.min() < 0.0:
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    m = m.float().clamp(0.0, 1.0)
    
    w = m.sum()
    if w.item() < 1e-6:
        return (0.0, 0.0, 0.0)
        
    r = (img[0] * m).sum() / w
    g = (img[1] * m).sum() / w
    b = (img[2] * m).sum() / w
    return (float(r.item()), float(g.item()), float(b.item()))

def rgb_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """ Simple Euclidean distance in RGB space. """
    return float(((torch.tensor(a) - torch.tensor(b)) ** 2).sum().sqrt().item())

def propose_param_updates(
    eval_result: EvalResult,
    target_threshold: float = 0.30,
    base_cfg_pos: float = 7.5,
    base_cfg_neg: float = 3.0,
    base_ip_weight: float = 1.0,
    max_increase: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    """
    Proposes parameter updates based on CLIP scores.
    """
    suggestions: Dict[str, Dict[str, float]] = {}
    
    for rs in eval_result.per_region:
        deficit = max(0.0, target_threshold - rs.score)
        
        if deficit <= 0.0:
            suggestions[rs.name] = {
                "cfg_pos": base_cfg_pos,
                "cfg_neg": base_cfg_neg,
                "ip_weight": base_ip_weight
            }
            continue
            
        # Linear scaling with safety clamping
        pos_new = min(base_cfg_pos + deficit * max_increase, base_cfg_pos + 5.0)
        neg_new = min(base_cfg_neg + deficit * (max_increase * 0.5), base_cfg_neg + 2.0)
        # IP-Adapter shouldn't go too high to avoid burning the image
        ip_new = min(base_ip_weight + deficit * max_increase, 1.3)
        
        suggestions[rs.name] = {
            "cfg_pos": float(pos_new),
            "cfg_neg": float(neg_new),
            "ip_weight": float(ip_new),
        }
        
    return suggestions

def evaluate_and_suggest(
    image: torch.Tensor,
    subjects: List[Dict[str, Any]],
    masks_img: Dict[str, torch.Tensor],
    evaluator: Optional[CLIPRegionEvaluator] = None,
    threshold: float = 0.30,
    base_cfg_pos: float = 7.5,
    base_cfg_neg: float = 3.0,
    base_ip_weight: float = 1.0,
) -> Tuple[EvalResult, Dict[str, Dict[str, float]]]:
    """
    High-level interface for evaluation and parameter suggestion.
    WARNING: Pass a pre-initialized evaluator to avoid reloading models every step.
    """
    if evaluator is None:
        # Only for fallback/testing; very slow in loops
        evaluator = CLIPRegionEvaluator()
        
    names = [s["name"] for s in subjects]
    texts = [s.get("text", s["name"]) for s in subjects]
    
    result = evaluator.score_regions(image, texts, masks=masks_img, bboxes=None, names=names)
    
    suggestions = propose_param_updates(
        result,
        target_threshold=threshold,
        base_cfg_pos=base_cfg_pos,
        base_cfg_neg=base_cfg_neg,
        base_ip_weight=base_ip_weight
    )
    return result, suggestions
