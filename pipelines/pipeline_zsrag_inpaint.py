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

from zsrag.core.attention_processor_raca import (
    RACAAttentionProcessor, ContextBank, AlphaScheduler, LayerSelector
)
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
from zsrag.core.utils import (
    seed_everything, load_sdxl_base, force_fp32, encode_prompt_sdxl, 
    build_added_kwargs_sdxl, attach_attention_processor, prepare_latents, 
    decode_vae, get_base_latent_size, save_image_tensor,
)


# -----------------------------
# Helper: make synthetic color patch reference (zero-training)
# -----------------------------
def make_color_patch(color_rgb: Tuple[int, int, int], size: Tuple[int, int] = (256, 256)) -> Image.Image:
    if not _HAS_PIL:
        raise ImportError("Pillow is required for make_color_patch")
    img = Image.new("RGB", (int(size[0]), int(size[1])), color=color_rgb)
    return img

def parse_color_from_text(text: str) -> Optional[Tuple[int, int, int]]:
    """ 
    Simple heuristic to parse color from text for synthetic IP-Adapter reference.
    """
    t = text.lower()
    if "red" in t: return (220, 30, 30)
    if "blue" in t: return (50, 80, 220)
    if "green" in t: return (30, 160, 60)
    if "yellow" in t: return (220, 210, 50)
    if "black" in t: return (15, 15, 15)
    if "white" in t: return (235, 235, 235)
    if "brown" in t: return (150, 90, 40)
    if "purple" in t: return (140, 60, 180)
    if "orange" in t: return (240, 120, 30)
    if "grey" in t or "gray" in t: return (128, 128, 128)
    return None


class ZSRAGPipeline:
    """
    Zero-Shot Region-aware Attribute Guidance (ZS-RAG) Pipeline.
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        fp32_unet: bool = True,
        raca_layer_patterns: Optional[List[str]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or torch.float16

        # 1. Load SDXL base
        # Enable VAE tiling/slicing for memory efficiency
        self.pipe = load_sdxl_base(
            model_id, 
            device=self.device, 
            torch_dtype=self.dtype, 
            vae_tiling=True, 
            vae_slicing=True
        )
        if fp32_unet:
            force_fp32(self.pipe)

        # 2. Attach RACA Attention Processor
        self.context_bank = ContextBank()
        # AlphaScheduler is mainly for optional Self-Attn injection (default off)
        alpha_scheduler = AlphaScheduler()  
        # LayerSelector controls where RACA gating is applied
        layer_selector = LayerSelector(
            patterns=raca_layer_patterns or ["down_blocks.1", "down_blocks.2", "mid_block", "up_blocks.0", "up_blocks.1"]
        )
        self.attn_proc = RACAAttentionProcessor(self.context_bank, alpha_scheduler, layer_selector)
        attach_attention_processor(self.pipe.unet, self.attn_proc)

        # 3. CLIP Evaluator (for feedback loop)
        self.clip_eval = CLIPRegionEvaluator(device=self.device)

        # 4. IP-Adapter (for attribute binding via image prompt)
        self.ip_adapter = IPAdapterXL(device=self.device, dtype=self.pipe.unet.dtype)
        self.ip_mgr = IPAdapterManager(self.ip_adapter)

        # 5. ControlNet Builder (Lazy loaded for Stage D)
        self.cnets: Optional[Dict[str, Any]] = None
        self.cn_builder: Optional[SDXLControlNetBuilder] = None

    # ---------------------------------
    # Stage A: Geometry Bootstrap
    # ---------------------------------
    @torch.no_grad()
    def bootstrap_geometry(
        self,
        global_prompt: str,
        width: int,
        height: int,
        seed: Optional[int] = None,
        guidance_scale: float = 5.5,
        num_inference_steps: int = 20,
        extract_lineart: bool = True,
        extract_depth: bool = True,
        depth_method: str = "zoe",
    ) -> Dict[str, Any]:
        """
        Generates initial global image I0 and extracts control signals.
        """
        seed_everything(seed or 42)
        
        # Standard generation using base pipe
        out = self.pipe(
            prompt=global_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        img = out.images[0]
        
        # Extract control signals
        extractor = ControlSignalExtractor(target_size=(width, height))
        extractor.set_image(img)
        
        line = extractor.compute_lineart(coarse=False) if extract_lineart else None
        dep = extractor.compute_depth(method=depth_method, normalize=True) if extract_depth else None
        
        return {"image": img, "lineart": line, "depth": dep}

    # ---------------------------------
    # Stage B: Mask Building
    # ---------------------------------
    def build_masks(
        self,
        subjects: List[Dict[str, str]],
        image_size: Tuple[int, int],
        latent_downsample: int = 8,
        init_mode: str = "positions",  # "positions" or "external"
        external_masks_img: Optional[Dict[str, torch.Tensor]] = None,
        safety_expand: float = 0.05,
        tau: float = 0.8,
        bg_floor: float = 0.05,
        gap_ratio: float = 0.06,
        feather_radius_img: int = 15,
        feather_radius_lat: int = 3,
    ) -> DynamicMaskManager:
        """
        Initializes Mask Manager.
        """
        H, W = image_size[1], image_size[0]
        H_lat, W_lat = get_base_latent_size(width=W, height=H, downsample=latent_downsample)

        if init_mode == "external" and external_masks_img is not None:
            mgr = DynamicMaskManager.init_from_external_masks(
                image_size=(H, W),
                latent_size=(H_lat, W_lat),
                masks_img=external_masks_img,
                safety_boxes=None,
                tau=tau,
                bg_floor=bg_floor,
                feather_radius_img=feather_radius_img,
                feather_radius_lat=feather_radius_lat,
                device=self.device,
                dtype=self.pipe.unet.dtype,
                z_order=[s["name"] for s in subjects],
            )
        else:
            mgr = DynamicMaskManager.init_from_positions(
                image_size=(H, W),
                latent_size=(H_lat, W_lat),
                subjects=subjects,
                safety_expand=safety_expand,
                tau=tau,
                bg_floor=bg_floor,
                gap_ratio=gap_ratio,
                device=self.device,
                dtype=self.pipe.unet.dtype,
            )
            mgr.set_z_order([s["name"] for s in subjects])
            
        return mgr

    # ---------------------------------
    # Stage C: Attribute Binding (Loop)
    # ---------------------------------
    def _prepare_subject_embeddings(
        self,
        width: int,
        height: int,
        subjects: List[Dict[str, Any]],
        add_ip_refs: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Prepares embeddings and IP-Adapter tokens for each subject.
        """
        # Placeholder time_ids
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float32, device=self.device)

        embs: Dict[str, Dict[str, Any]] = {}
        for subj in subjects:
            name = subj["name"]
            pos_prompt = subj["prompt"]
            neg_target_text = subj.get("neg_target", "")
            
            # Encode Text
            pe, ne, pp, np = encode_prompt_sdxl(self.pipe, pos_prompt, "")
            
            # Encode Negative Target (for contrastive guidance)
            if neg_target_text:
                nt_pe, _, nt_pp, _ = encode_prompt_sdxl(self.pipe, neg_target_text, "")
            else:
                nt_pe, nt_pp = None, None

            # IP-Adapter Reference
            pos_embs = pe
            if add_ip_refs:
                ref_image = subj.get("ref_image", None)
                # Auto-generate color patch if no ref image provided
                if ref_image is None:
                    color_rgb = parse_color_from_text(pos_prompt)
                    if color_rgb is not None:
                        ref_image = make_color_patch(color_rgb, size=(256, 256))
                
                if ref_image is not None:
                    self.ip_mgr.clear() # Clear prev subject refs
                    self.ip_mgr.add_reference(name, ref_image, weight=float(subj.get("ip_weight", 1.0)))
                    
                    # Merge image tokens into text embeddings
                    # Note: this modifies the 'pos_embs' tensor shape to [B, N+M, D]
                    pos_embs = self.ip_mgr.merge_subject_refs_into_text(name, pe, agg="mean")

            # Pack
            embs[name] = {
                "pos": pos_embs,
                "uncond": ne,
                "neg": nt_pe,
                "added_pos": {"text_embeds": pp, "time_ids": add_time_ids},
                "added_uncond": {"text_embeds": np, "time_ids": add_time_ids},
                "added_neg": ({"text_embeds": nt_pp, "time_ids": add_time_ids} if nt_pp is not None else None),
            }
        return embs

    @torch.no_grad()
    def bind_attributes(
        self,
        global_prompt: str,
        subjects: List[Dict[str, Any]],
        width: int,
        height: int,
        mask_mgr: DynamicMaskManager,
        num_inference_steps: int = 40,
        cfg_pos: float = 7.5,
        cfg_neg: float = 3.0,
        kappa: float = 1.0,
        seed: int = 42,
        iter_clip: int = 2,
        clip_threshold: float = 0.32,
        probe_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        Stage C: Attribute Binding Loop.
        """
        seed_everything(seed)
        H_lat, W_lat = get_base_latent_size(width, height, downsample=8)
        self.attn_proc.set_base_latent_hw(H_lat, W_lat)

        scheduler = self.pipe.scheduler
        
        # Initial Latents (Created once, reused as starting point)
        # Note: timesteps will be set inside the loop
        latents = prepare_latents(1, 4, H_lat, W_lat, self.pipe.unet.dtype, self.device, scheduler, seed=seed)

        # Global Background Embeds
        ge_pos, ge_uncond, ge_pp, ge_np = encode_prompt_sdxl(self.pipe, global_prompt, "")
        added_global_pos = build_added_kwargs_sdxl(ge_pp, torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float32, device=self.device))
        added_global_uncond = build_added_kwargs_sdxl(ge_np, torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float32, device=self.device))

        # Subject Embeds (Initial)
        embs_sub = self._prepare_subject_embeddings(width, height, subjects, add_ip_refs=True)
        self.attn_proc.set_key_agg_mode("mean")

        # Per-Subject Config (for iteration)
        per_subject_cfg = {s["name"]: {"cfg_pos": float(cfg_pos), "cfg_neg": float(cfg_neg), "ip_weight": float(s.get("ip_weight", 1.0))} for s in subjects}

        image_out = None
        
        # Iteration Loop (Feedback-Driven)
        for iter_idx in range(max(1, int(iter_clip) + 1)):
            
            # 【CRITICAL FIX】Reset scheduler state for each iteration
            # This resets 'step_index' to 0 and re-calculates sigmas
            timesteps = prepare_timesteps(scheduler, num_inference_steps=num_inference_steps, device=self.device)
            
            # CLIP Feedback Adjustment
            if iter_idx > 0 and image_out is not None:
                masks_img = mask_mgr.get_masks_img()
                eval_result, suggestions = evaluate_and_suggest(
                    image_out[0], 
                    subjects=[{"name": s["name"], "text": s["prompt"]} for s in subjects], 
                    masks_img=masks_img, 
                    evaluator=self.clip_eval, 
                    threshold=clip_threshold, 
                    base_cfg_pos=cfg_pos, 
                    base_cfg_neg=cfg_neg, 
                    base_ip_weight=1.0
                )
                
                # Apply suggestions
                for name, sug in suggestions.items():
                    per_subject_cfg[name].update(sug)
                
                print(f"[ZS-RAG] Iter {iter_idx} CLIP suggestions: {per_subject_cfg}")

            # Diffusion Loop
            # Always start from the original noisy latents for Fair Comparison between iterations
            latents_iter = latents.clone() 
            
            for step_idx, t in enumerate(timesteps):
                self.attn_proc.set_step_index(step_idx, num_inference_steps)
                latent_model_input = scheduler.scale_model_input(latents_iter, t)

                # 1. Background Recording (Phase A)
                if step_idx % 3 == 0:
                    self.context_bank.clear()
                    self.attn_proc.set_mode("record_bg")
                    self.attn_proc.set_probe_enabled(False)
                    self.attn_proc.set_cross_gate_enabled(False)
                    self.attn_proc.set_region_mask(None)
                    
                    forward_with_gate(
                        self.pipe.unet, latent_model_input, t, ge_pos, added_global_pos, 
                        attn_proc=self.attn_proc, gate_enabled=False, mode="record_bg", 
                        probe=False, region_mask=None, subject_id=None
                    )

                # 2. Background Prediction
                eps_bg = compute_cfg_raca(
                    self.pipe.unet, latent_model_input, t,
                    emb_pos=ge_pos, emb_uncond=ge_uncond,
                    added_cond_kwargs_pos=added_global_pos,
                    added_cond_kwargs_uncond=added_global_uncond,
                    attn_proc=self.attn_proc,
                    cfg_pos=float(cfg_pos),
                    gate_uncond=False, gate_pos=False, 
                    mode_uncond="off", mode_pos="off",
                )

                # 3. Subject Predictions
                eps_subjects = {}
                masks_latent_gate, _ = mask_mgr.get_masks_latent()
                do_probe = (step_idx % max(1, int(probe_interval)) == 0)
                self.attn_proc.set_probe_enabled(do_probe)

                for subj_idx, s in enumerate(subjects):
                    name = s["name"]
                    emb = embs_sub[name]
                    
                    self.attn_proc.set_region_mask(masks_latent_gate[name])
                    self.attn_proc.set_subject_id(subj_idx)

                    # Dynamic IP-Adapter Weight Update
                    if "ip_weight" in per_subject_cfg[name]:
                        # Optimized: Re-merge only if weight changed significant logic needed here
                        # For now, simplistic re-calc (performance hit is minimal compared to UNet)
                        ref_image = s.get("ref_image", None)
                        if ref_image is None:
                            color_rgb = parse_color_from_text(s["prompt"])
                            if color_rgb is not None:
                                ref_image = make_color_patch(color_rgb, size=(256, 256))
                                
                        if ref_image is not None:
                            self.ip_mgr.clear()
                            self.ip_mgr.add_reference(name, ref_image, weight=float(per_subject_cfg[name]["ip_weight"]))
                            emb["pos"] = self.ip_mgr.merge_subject_refs_into_text(name, emb["pos"], agg="mean")

                    # CFG Calculation
                    if emb["neg"] is not None and emb["added_neg"] is not None:
                        eps_sub = compute_contrastive_cfg_raca(
                            self.pipe.unet, latent_model_input, t,
                            emb_pos=emb["pos"], emb_uncond=emb["uncond"], emb_neg_target=emb["neg"],
                            added_cond_kwargs_pos=emb["added_pos"], added_cond_kwargs_uncond=emb["added_uncond"],
                            added_cond_kwargs_neg=emb["added_neg"],
                            attn_proc=self.attn_proc,
                            cfg_pos=float(per_subject_cfg[name]["cfg_pos"]),
                            cfg_neg=float(per_subject_cfg[name]["cfg_neg"]),
                            region_mask=masks_latent_gate[name],
                            subject_id=subj_idx,
                            probe_pos=do_probe,
                            gate_uncond=False, gate_pos=True, gate_neg=True,
                            mode_uncond="off", mode_pos="inject_subject", mode_neg="inject_subject",
                        )
                    else:
                        eps_sub = compute_cfg_raca(
                            self.pipe.unet, latent_model_input, t,
                            emb_pos=emb["pos"], emb_uncond=emb["uncond"],
                            added_cond_kwargs_pos=emb["added_pos"], added_cond_kwargs_uncond=emb["added_uncond"],
                            attn_proc=self.attn_proc,
                            cfg_pos=float(per_subject_cfg[name]["cfg_pos"]),
                            region_mask=masks_latent_gate[name],
                            subject_id=subj_idx,
                            probe_pos=do_probe,
                            gate_uncond=False, gate_pos=True,
                            mode_uncond="off", mode_pos="inject_subject",
                        )
                    eps_subjects[name] = eps_sub

                self.attn_proc.set_probe_enabled(False)
                self.attn_proc.set_region_mask(None)

                # 4. Latent Blending
                eps_final = mask_mgr.energy_minimize_blend(eps_bg, eps_subjects, kappa=float(kappa))

                # 5. Step
                latents_iter = scheduler.step(eps_final, t, latents_iter).prev_sample

                # 6. Mask Update (Probing)
                if do_probe:
                    attn_maps = self.attn_proc.pop_attn_maps()
                    maps_img = {}
                    for sid, amap in attn_maps.items():
                        name = subjects[sid]["name"]
                        maps_img[name] = F.interpolate(amap, size=(height, width), mode="bilinear", align_corners=False)
                    if maps_img:
                        mask_mgr.update_from_attn(maps_img, beta=0.6, thresh=0.5, erode_k=5, blur_ksize=15, blur_sigma=2.5)

            # Decode Final Image for this iteration
            # We must decode to evaluate CLIP scores for the next loop
            image_out = decode_vae(self.pipe.vae, latents_iter)
            
            # Update latents for final return (after loop ends)
            final_latents = latents_iter.clone()

        return {"image": image_out, "latents": final_latents, "masks_img": mask_mgr.get_masks_img()}


    # ---------------------------------
    # Stage D: Global Harmonization
    # ---------------------------------
    def harmonize_global(
        self,
        init_image: Union[Image.Image, torch.Tensor],
        global_prompt: str,
        width: int,
        height: int,
        control_lineart: Optional[Image.Image],
        control_depth: Optional[Image.Image],
        strength: float = 0.3,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Stage D: SDEdit with ControlNet.
        """
        # =========================================================
        # 显存大清洗 (针对 FP32 用户必须执行)
        # =========================================================
        print("[ZS-RAG] Moving Stage C models to CPU to free VRAM for ControlNet...")
        
        # 1. 卸载 Base Pipeline 组件
        self.pipe.unet.to("cpu")
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        
        # 2. 卸载 IP-Adapter
        self.ip_adapter.clip_image_encoder.to("cpu")
        self.ip_adapter.resampler.to("cpu") # 如果有的话
        
        # 3. 卸载 CLIP Evaluator
        self.clip_eval.model.to("cpu")
        
        # 4. 清理缓存
        torch.cuda.empty_cache()
        # =========================================================

        # Lazy Load ControlNets (以下代码保持不变)
        if self.cn_builder is None:
            controlnets = {}
            # Using SDXL compatible ControlNets (e.g., Canny/Depth)
            if control_lineart is not None:
                controlnets.update(load_controlnets(
                    canny_model_id="diffusers/controlnet-canny-sdxl-1.0", # Used for lineart structure
                    depth_model_id=None, 
                    device=self.device, torch_dtype=self.pipe.unet.dtype
                ))
            if control_depth is not None:
                controlnets.update(load_controlnets(
                    canny_model_id=None,
                    depth_model_id="diffusers/controlnet-depth-sdxl-1.0", 
                    device=self.device, torch_dtype=self.pipe.unet.dtype
                ))
            
            if len(controlnets) == 0:
                # Fallback: No ControlNet, just return input (or run naive img2img)
                return init_image if isinstance(init_image, Image.Image) else None
                
            self.cn_builder = SDXLControlNetBuilder(
                base_model_id="stabilityai/stable-diffusion-xl-base-1.0", 
                device=self.device, 
                torch_dtype=self.pipe.unet.dtype, 
                controlnets=controlnets
            )

        pipe_inpaint = self.cn_builder.get_inpaint_pipeline()

        # Pack Control Images
        pack = ControlImagePack(target_size=(width, height))
        # Note: Order must match keys in cn_builder. 
        # Since we use 'canny' and 'depth' keys, we add them accordingly.
        # Check builder keys order
        for key in self.cn_builder.controlnet_keys:
            if key == "canny" and control_lineart is not None:
                pack.add("canny", control_lineart)
            elif key == "depth" and control_depth is not None:
                pack.add("depth", control_depth)

        img_sde = sdxl_sde_edit(
            pipe_inpaint,
            image=init_image,
            prompt=global_prompt,
            control_images=pack.get(),
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return img_sde


# ---------------------------------
# Convenience Runner
# ---------------------------------
@torch.no_grad()
def run_zsrag(
    pipeline: ZSRAGPipeline,  # Pass initialized pipeline
    global_prompt: str,
    subjects: List[Dict[str, str]],
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    # Stage A
    bootstrap_steps: int = 20,
    bootstrap_guidance: float = 5.5,
    # Stage B
    safety_expand: float = 0.05,
    tau: float = 0.8,
    bg_floor: float = 0.05,
    gap_ratio: float = 0.06,
    # Stage C
    num_inference_steps: int = 40,
    cfg_pos: float = 7.5,
    cfg_neg: float = 3.0,
    kappa: float = 1.0,
    iter_clip: int = 2,
    clip_threshold: float = 0.32,
    probe_interval: int = 5,
    # Stage D
    sde_strength: float = 0.3,
    sde_steps: int = 20,
    sde_guidance: float = 5.0,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the full ZS-RAG pipeline.
    """
    # Stage A
    boot = pipeline.bootstrap_geometry(
        global_prompt=global_prompt, width=width, height=height, seed=seed, 
        guidance_scale=bootstrap_guidance, num_inference_steps=bootstrap_steps, 
        extract_lineart=True, extract_depth=True
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        boot["image"].save(os.path.join(save_dir, "stageA_bootstrap.png"))
        if boot["lineart"]: boot["lineart"].save(os.path.join(save_dir, "stageA_lineart.png"))
        if boot["depth"]: boot["depth"].save(os.path.join(save_dir, "stageA_depth.png"))

    # Stage B
    mask_mgr = pipeline.build_masks(
        subjects, image_size=(width, height), latent_downsample=8, 
        init_mode="positions", external_masks_img=None, 
        safety_expand=safety_expand, tau=tau, bg_floor=bg_floor, gap_ratio=gap_ratio,
        feather_radius_img=15, feather_radius_lat=3
    )

    # Stage C
    bind = pipeline.bind_attributes(
        global_prompt=global_prompt, subjects=subjects, width=width, height=height, 
        mask_mgr=mask_mgr, num_inference_steps=num_inference_steps, 
        cfg_pos=cfg_pos, cfg_neg=cfg_neg, kappa=kappa, seed=seed, 
        iter_clip=iter_clip, clip_threshold=clip_threshold, probe_interval=probe_interval
    )
    if save_dir:
        save_image_tensor(bind["image"], os.path.join(save_dir, "stageC_bind.png"))

    # Stage D
    img_harmonized = pipeline.harmonize_global(
        init_image=bind["image"][0], global_prompt=global_prompt, width=width, height=height, 
        control_lineart=boot["lineart"], control_depth=boot["depth"], 
        strength=sde_strength, num_inference_steps=sde_steps, guidance_scale=sde_guidance, seed=seed
    )
    if save_dir and img_harmonized:
        img_harmonized.save(os.path.join(save_dir, "stageD_harmonized.png"))

    return {"bootstrap": boot, "bind": bind, "harmonized": img_harmonized}
