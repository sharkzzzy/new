"""
zsrag/pipelines/runner.py
Command Line Interface (CLI) runner for ZS-RAG.
Handles configuration loading, subject parsing, and pipeline execution.
"""

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Pipeline Imports
from zsrag.pipelines.pipeline_zsrag_inpaint import ZSRAGPipeline
from zsrag.core.sam_seg import SAMSegmenter
from zsrag.core.grounded_dino_utils import GroundedDINODetector, boxes_from_positions
from zsrag.core.utils import save_image_tensor


# -----------------------------
# Config loading utilities
# -----------------------------
def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """ Load YAML configuration if path provided, else return empty dict. """
    if not path:
        return {}
    try:
        import yaml
    except ImportError as e:
        print(f"[runner] YAML not available: {e}. Skipping config file.")
        return {}
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                print("[runner] YAML config is not a dict. Ignored.")
                return {}
            return cfg
    except Exception as e:
        print(f"[runner] Failed to load YAML config: {e}.")
        return {}

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """ Shallow merge two dicts: override keys take precedence. """
    out = dict(base or {})
    for k, v in (override or {}).items():
        if v is not None: # Only override if value is present
            out[k] = v
    return out


# -----------------------------
# Subject parsing utilities
# -----------------------------
def parse_subjects_cli(subjects_str: Optional[str]) -> List[Dict[str, str]]:
    """ 
    Parse CLI subjects string into list of dicts.
    Format: "name:cat|prompt:a red cat|position:left; name:dog|prompt:a blue dog"
    """
    if not subjects_str:
        return []
    subjects: List[Dict[str, str]] = []
    for subj_seg in subjects_str.split(";"):
        seg = subj_seg.strip()
        if not seg: continue
        fields = {}
        for kv in seg.split("|"):
            kv = kv.strip()
            if not kv: continue
            parts = kv.split(":", 1)
            if len(parts) != 2: continue
            key = parts[0].strip()
            val = parts[1].strip()
            fields[key] = val
        if "name" in fields and "prompt" in fields:
            subjects.append(fields)
    return subjects

def load_subjects_json(path: Optional[str]) -> List[Dict[str, Any]]:
    """ Load subjects from JSON file. """
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"[runner] Failed to load subjects JSON: {e}")
    return []

def attach_ref_images(subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ Attach ref_image tensor/PIL from file paths if provided. """
    from PIL import Image
    out = []
    for s in subjects:
        ss = dict(s)
        ref_path = ss.get("ref_image", None)
        if ref_path and isinstance(ref_path, str) and os.path.exists(ref_path):
            try:
                ss["ref_image"] = Image.open(ref_path).convert("RGB")
            except Exception as e:
                print(f"[runner] Failed to load ref_image '{ref_path}': {e}")
                ss["ref_image"] = None
        # If ref_image is already an object (not string), leave it
        elif ref_path and not isinstance(ref_path, str):
             pass 
        else:
            ss["ref_image"] = None
        out.append(ss)
    return out


# -----------------------------
# Mask building utilities
# -----------------------------
def simple_rect_mask(size: Tuple[int, int], box_xyxy: Tuple[int, int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """ Build a simple rectangular mask [1,1,H,W] from box coordinates. """
    W, H = size
    x1, y1, x2, y2 = box_xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    
    m = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    if x2 > x1 and y2 > y1:
        m[:, :, y1:y2, x1:x2] = 1.0
    return m

def build_masks_with_sam_and_dino(
    image_pil,
    subjects: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    use_grounded_dino: bool = True,
    dino_box_threshold: float = 0.3,
    dino_text_threshold: float = 0.25,
    dino_nms_iou: float = 0.5,
    topk_per_class: int = 1,
    sam_model_type: str = "vit_h",
    sam_checkpoint: Optional[str] = None,
    sam_box_expansion: float = 0.05,
    postprocess: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build external masks using Grounded-DINO (boxes) + SAM (segmentation).
    Fallback to position-based boxes if DINO is not available or fails.
    """
    names = [s["name"] for s in subjects]
    W, H = image_size
    boxes_dict: Dict[str, List[Tuple[int, int, int, int]]] = {}

    # 1. Get Boxes (DINO or Fallback)
    if use_grounded_dino:
        try:
            detector = GroundedDINODetector()
            det_out = detector.detect(
                image=image_pil,
                names=names,
                box_threshold=dino_box_threshold,
                text_threshold=dino_text_threshold,
                nms_iou=dino_nms_iou,
                topk_per_class=topk_per_class,
                resize_for_detector=None,
            )
            boxes_dict = {n: det_out.get(n, []) for n in names}
        except Exception as e:
            print(f"[runner] Grounded-DINO failed: {e}. Falling back to position-based boxes.")
            pos_boxes = boxes_from_positions(image_size=(W, H), subjects=subjects, gap_ratio=0.08, safety_expand=0.05)
            boxes_dict = {n: [pos_boxes[n]] for n in names}
    else:
        pos_boxes = boxes_from_positions(image_size=(W, H), subjects=subjects, gap_ratio=0.08, safety_expand=0.05)
        boxes_dict = {n: [pos_boxes[n]] for n in names}

    # 2. SAM Segmentation
    try:
        seg = SAMSegmenter(model_type=sam_model_type, checkpoint_path=sam_checkpoint)
        seg.set_image(image_pil)
        
        # Select one best box per subject
        boxes_one = {}
        for n in names:
            if len(boxes_dict.get(n, [])) > 0:
                boxes_one[n] = boxes_dict[n][0]
            else:
                # Fallback if DINO returned empty list for a name
                fallback_box = boxes_from_positions((W, H), [{"name": n, "position": "center"}])[n]
                boxes_one[n] = fallback_box

        masks = seg.predict_from_boxes(
            boxes=boxes_one,
            multimask_output=False,
            box_expansion=sam_box_expansion,
            postprocess=postprocess or {"erode": 5, "blur_ksize": 15, "blur_sigma": 2.5},
        )
        return masks
        
    except Exception as e:
        print(f"[runner] SAM segmentation failed: {e}. Using rectangular masks from boxes as fallback.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        masks_fallback: Dict[str, torch.Tensor] = {}
        for n in names:
            box = boxes_dict.get(n, [None])[0] 
            if box is None:
                 box = boxes_from_positions((W, H), [{"name": n, "position": "center"}])[n]
            masks_fallback[n] = simple_rect_mask((W, H), box, device=device, dtype=dtype)
        return masks_fallback


# -----------------------------
# Main runner logic
# -----------------------------
def run(
    config: Dict[str, Any],
    save_dir: Optional[str] = None,
):
    """ Execute ZS-RAG pipeline according to config. """
    out_dir = save_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Basic parameters
    global_prompt = config.get("global_prompt", "a scenic landscape, cinematic lighting, photorealistic, high quality")
    subjects = config.get("subjects", [])
    width = int(config.get("width", 1024))
    height = int(config.get("height", 1024))
    seed = int(config.get("seed", 42))

    # Stage A params
    bootstrap_steps = int(config.get("bootstrap_steps", 20))
    bootstrap_guidance = float(config.get("bootstrap_guidance", 5.5))

    # Stage B params
    use_sam = bool(config.get("use_sam", False))
    use_grounded_dino = bool(config.get("use_grounded_dino", False))
    safety_expand = float(config.get("safety_expand", 0.05))
    tau = float(config.get("tau", 0.8))
    bg_floor = float(config.get("bg_floor", 0.05))
    gap_ratio = float(config.get("gap_ratio", 0.06))

    # Stage C params
    num_inference_steps = int(config.get("num_inference_steps", 40))
    cfg_pos = float(config.get("cfg_pos", 7.5))
    cfg_neg = float(config.get("cfg_neg", 3.0))
    kappa = float(config.get("kappa", 1.0))
    iter_clip = int(config.get("iter_clip", 2))
    clip_threshold = float(config.get("clip_threshold", 0.32))
    probe_interval = int(config.get("probe_interval", 5))

    # Stage D params
    sde_strength = float(config.get("sde_strength", 0.3))
    sde_steps = int(config.get("sde_steps", 20))
    sde_guidance = float(config.get("sde_guidance", 5.0))

    # Build pipeline
    print("[runner] Loading ZSRAGPipeline...")
    pipe = ZSRAGPipeline(fp32_unet=True)

    # Stage A: bootstrap geometry
    print("[runner] Stage A: Bootstrap geometry...")
    boot = pipe.bootstrap_geometry(
        global_prompt=global_prompt, 
        width=width, height=height, seed=seed, 
        guidance_scale=bootstrap_guidance, num_inference_steps=bootstrap_steps, 
        extract_lineart=True, extract_depth=True
    )
    boot_img = boot["image"]
    img_boot_path = os.path.join(out_dir, "stageA_bootstrap.png")
    boot_img.save(img_boot_path)
    
    if boot.get("lineart"):
        boot["lineart"].save(os.path.join(out_dir, "stageA_lineart.png"))
    if boot.get("depth"):
        boot["depth"].save(os.path.join(out_dir, "stageA_depth.png"))

    # Stage B: masks
    print("[runner] Stage B: Build masks...")
    subjects = attach_ref_images(subjects)
    
    if use_sam:
        masks_img = build_masks_with_sam_and_dino(
            image_pil=boot_img,
            subjects=subjects,
            image_size=(width, height),
            use_grounded_dino=use_grounded_dino,
            dino_box_threshold=float(config.get("dino_box_threshold", 0.3)),
            dino_text_threshold=float(config.get("dino_text_threshold", 0.25)),
            dino_nms_iou=float(config.get("dino_nms_iou", 0.5)),
            topk_per_class=int(config.get("topk_per_class", 1)),
            sam_model_type=str(config.get("sam_model_type", "vit_h")),
            sam_checkpoint=config.get("sam_checkpoint", None),
            sam_box_expansion=float(config.get("sam_box_expansion", 0.05)),
            postprocess=config.get("sam_postprocess", {"erode": 5, "blur_ksize": 15, "blur_sigma": 2.5}),
        )
        # Build mask manager from external masks
        mask_mgr = pipe.build_masks(
            subjects=subjects,
            image_size=(width, height),
            latent_downsample=8,
            init_mode="external",
            external_masks_img=masks_img,
            safety_expand=safety_expand,
            tau=tau,
            bg_floor=bg_floor,
            gap_ratio=gap_ratio,
        )
    else:
        # Fallback to positional
        mask_mgr = pipe.build_masks(
            subjects=subjects,
            image_size=(width, height),
            latent_downsample=8,
            init_mode="positions",
            external_masks_img=None,
            safety_expand=safety_expand,
            tau=tau,
            bg_floor=bg_floor,
            gap_ratio=gap_ratio,
        )

    # Stage C: bind attributes
    print("[runner] Stage C: Bind attributes (RACA + Contrastive CFG + IP-Adapter + CLIP iteration)...")
    bind = pipe.bind_attributes(
        global_prompt=global_prompt,
        subjects=subjects,
        width=width,
        height=height,
        mask_mgr=mask_mgr,
        num_inference_steps=num_inference_steps,
        cfg_pos=cfg_pos,
        cfg_neg=cfg_neg,
        kappa=kappa,
        seed=seed,
        iter_clip=iter_clip,
        clip_threshold=clip_threshold,
        probe_interval=probe_interval,
    )
    save_image_tensor(bind["image"], os.path.join(out_dir, "stageC_bind.png"))

    # Stage D: global harmonization (SDEdit)
    print("[runner] Stage D: Global harmonization (SDEdit)...")
    img_harmonized = pipe.harmonize_global(
        init_image=bind["image"][0],
        global_prompt=global_prompt,
        width=width,
        height=height,
        control_lineart=boot.get("lineart"),
        control_depth=boot.get("depth"),
        strength=sde_strength,
        num_inference_steps=sde_steps,
        guidance_scale=sde_guidance,
        seed=seed,
    )
    if img_harmonized is not None:
        img_harmonized.save(os.path.join(out_dir, "stageD_harmonized.png"))

    print(f"[runner] Done. Outputs saved in: {out_dir}")


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser():
    p = argparse.ArgumentParser("ZS-RAG Runner (zero-training, multi-subject region-aware guidance)")
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file (optional).")
    p.add_argument("--save_dir", type=str, default="outputs_zsrag", help="Output directory.")
    
    p.add_argument("--global_prompt", type=str, default=None, help="Global prompt.")
    p.add_argument("--subjects_json", type=str, default=None, help="Path to subjects JSON.")
    p.add_argument("--subjects", type=str, default=None, help="Inline subjects spec string. See parse_subjects_cli doc.")
    
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # Stage toggles
    p.add_argument("--use_sam", action="store_true", help="Use SAM for mask building.")
    p.add_argument("--use_grounded_dino", action="store_true", help="Use Grounded-DINO to get boxes (with SAM).")

    # Stage C
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--cfg_pos", type=float, default=7.5)
    p.add_argument("--cfg_neg", type=float, default=3.0)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--iter_clip", type=int, default=2)
    p.add_argument("--clip_threshold", type=float, default=0.32)
    p.add_argument("--probe_interval", type=int, default=5)

    # Stage D
    p.add_argument("--sde_strength", type=float, default=0.3)
    p.add_argument("--sde_steps", type=int, default=20)
    p.add_argument("--sde_guidance", type=float, default=5.0)

    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load base config
    cfg_file = load_yaml_config(args.config)
    cfg_cli: Dict[str, Any] = {}

    # CLI overrides
    if args.global_prompt:
        cfg_cli["global_prompt"] = args.global_prompt
    if args.width:
        cfg_cli["width"] = args.width
    if args.height:
        cfg_cli["height"] = args.height
    if args.seed:
        cfg_cli["seed"] = args.seed

    if args.use_sam:
        cfg_cli["use_sam"] = bool(args.use_sam)
    if args.use_grounded_dino:
        cfg_cli["use_grounded_dino"] = bool(args.use_grounded_dino)

    if args.num_inference_steps:
        cfg_cli["num_inference_steps"] = args.num_inference_steps
    if args.cfg_pos:
        cfg_cli["cfg_pos"] = args.cfg_pos
    if args.cfg_neg:
        cfg_cli["cfg_neg"] = args.cfg_neg
    if args.kappa:
        cfg_cli["kappa"] = args.kappa
    if args.iter_clip:
        cfg_cli["iter_clip"] = args.iter_clip
    if args.clip_threshold:
        cfg_cli["clip_threshold"] = args.clip_threshold
    if args.probe_interval:
        cfg_cli["probe_interval"] = args.probe_interval

    if args.sde_strength:
        cfg_cli["sde_strength"] = args.sde_strength
    if args.sde_steps:
        cfg_cli["sde_steps"] = args.sde_steps
    if args.sde_guidance:
        cfg_cli["sde_guidance"] = args.sde_guidance

    # Subjects from JSON or CLI
    subjects_list: List[Dict[str, Any]] = []
    if args.subjects_json:
        subjects_list = load_subjects_json(args.subjects_json)
    elif args.subjects:
        subjects_list = parse_subjects_cli(args.subjects)
        
    if len(subjects_list) > 0:
        cfg_cli["subjects"] = subjects_list

    # Merge configs
    cfg = merge_configs(cfg_file, cfg_cli)

    # Run
    run(cfg, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
