"""
Core modules init
"""
from .attention_processor_raca import RACAAttentionProcessor, ContextBank, AlphaScheduler, LayerSelector
from .guidance import (
    prepare_timesteps, compute_cfg_raca, compute_contrastive_cfg_raca, forward_with_gate,
)
from .mask_manager import DynamicMaskManager
from .clip_eval import CLIPRegionEvaluator, evaluate_and_suggest
from .ip_adapter_wrapper import IPAdapterXL, IPAdapterManager
from .controlnet_utils import (
    load_controlnets, SDXLControlNetBuilder, ControlImagePack, 
    run_controlnet_inpaint, sdxl_sde_edit,
)
from .depth_lineart_extract import ControlSignalExtractor
from .sam_seg import SAMSegmenter
from .grounded_dino_utils import GroundedDINODetector, boxes_from_positions
from .utils import (
    seed_everything, load_sdxl_base, force_fp32, encode_prompt_sdxl, 
    build_added_kwargs_sdxl, attach_attention_processor, prepare_latents, 
    decode_vae, get_base_latent_size, save_image_tensor,
)
