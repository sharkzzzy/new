"""
zsrag/pipelines/pipeline_zsrag_inpaint.py
Main pipeline for Zero-Shot Region-aware Attribute Guidance (ZS-RAG).

Stages:
A. Geometry Bootstrap: Generate initial image & extract structural signals (Lineart/Depth).
B. Mask Building: Initialize region masks via SAM or positions.
C. Attribute Binding: Multi-subject generation loop with RACA, IP-Adapter, and CLIP feedback.
D. Global Harmonization: SDEdit refinement with ControlNet for unified lighting/texture.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Core Dependencies
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


from zsrag.core.guidance import (
    prepare_timesteps, compute_cfg_raca, compute_contrastive_cfg_raca, forward_with_gate,
)
from zsrag.core.mask_manager import DynamicMaskManager
from zsrag.core.clip_eval import CLIPRegionEvaluator, evaluate_and_suggest
from zsrag.core.ip_adapter_wrapper import IPAdapterXL, IPAdapterManager
from zsrag.core.controlnet_utils import (
    load_controlnets, SDXLControlNetBuilder, ControlImagePack, 
    run_controlnet_inpaint, sdxl_sde_edit,
)
from zsrag.core.depth_lineart_extract import ControlSignalExtractor

from zsrag.core.attention_processor_raca import create_weaver_processor, ConceptWeaverProcessor
from zsrag.core.utils import attach_attention_processor, load_sdxl_base, force_fp32

class ZSRAGPipeline:
    def __init__(self, device=None, fp32_unet=True, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        self.device = device or torch.device("cuda")
        self.dtype = torch.float32 if fp32_unet else torch.float16
        
        self.pipe = load_sdxl_base(model_id, device=self.device, torch_dtype=self.dtype)
        if fp32_unet: force_fp32(self.pipe)
        
        # Weaver Processor
        self.processor = create_weaver_processor()
        attach_attention_processor(self.pipe.unet, self.processor)

    @torch.no_grad()
    def generate_asset(self, name: str, prompt: str, width: int, height: int, seed: int):
        """ Generates a single asset and records its features. """
        print(f"[Weaver] Generating Asset: {name}...")
        
        # Set Processor to Record Mode
        self.processor.set_mode("record", subject=name)
        
        # Generator
        gen = torch.Generator(self.device).manual_seed(seed)
        
        # Run Standard Generation
        # Note: We need only 1 image.
        out = self.pipe(
            prompt=prompt, 
            height=height, width=width, 
            num_inference_steps=30, 
            generator=gen
        )
        return out.images[0]

    @torch.no_grad()
    def weave(
        self, 
        global_prompt: str, 
        layout: Dict[str, Tuple[int, int, int, int]], # {name: dst_box}
        src_boxes: Dict[str, Tuple[int, int, int, int]], # {name: src_box}
        width: int, height: int, seed: int
    ):
        """ Generates the final image by weaving features. """
        print("[Weaver] Weaving Final Image...")
        
        # Configure Processor
        # We need to pass the layout map: name -> (src_box, dst_box)
        layout_map = {}
        for name, dst_box in layout.items():
            if name in src_boxes:
                layout_map[name] = (src_boxes[name], dst_box)
        
        self.processor.set_mode("weave")
        self.processor.set_layout(layout_map, canvas_size=(width, height))
        
        # Generator
        gen = torch.Generator(self.device).manual_seed(seed)
        
        # Run Generation
        out = self.pipe(
            prompt=global_prompt, 
            height=height, width=width, 
            num_inference_steps=30, 
            generator=gen
        )
        return out.images[0]

# ... run_zsrag wrapper ...
def run_zsrag(
    pipeline: ZSRAGPipeline,
    global_prompt: str,
    subjects: List[Dict[str, str]],
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    save_dir: str = None
):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Define Layouts (Hardcoded for demo, or calculate dynamically)
    # Background fills the screen
    bg_src_box = (0, 0, width, height)
    
    # Subjects (assume centered in asset gen)
    # Asset gen usually puts subject in center. Box: (256, 256, 768, 768) for 1024x1024
    subj_src_box = (int(width*0.25), int(height*0.25), int(width*0.75), int(height*0.75))
    
    # Target Layout (Left / Right)
    layout_dst = {
        "cat": (0, int(height*0.3), int(width*0.5), height), # Left bottom
        "dog": (int(width*0.5), int(height*0.3), width, height) # Right bottom
    }
    src_boxes_map = {
        "cat": subj_src_box,
        "dog": subj_src_box
    }

    # 2. Generate Assets (Phase 1)
    # Background
    bg_img = pipeline.generate_asset("background", global_prompt, width, height, seed)
    bg_img.save(os.path.join(save_dir, "asset_bg.png"))
    
    # Subjects
    for subj in subjects:
        name = subj["name"]
        # Add "white background" to ensure clean asset
        prompt = f"{subj['prompt']}, white background, simple background"
        img = pipeline.generate_asset(name, prompt, width, height, seed)
        img.save(os.path.join(save_dir, f"asset_{name}.png"))

    # 3. Weave (Phase 2 & 3)
    final_img = pipeline.weave(
        global_prompt=global_prompt,
        layout=layout_dst,
        src_boxes=src_boxes_map,
        width=width, height=height, seed=seed
    )
    final_img.save(os.path.join(save_dir, "final_weaved.png"))
    
    return {"final": final_img}
