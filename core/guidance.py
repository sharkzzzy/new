"""
zsrag/core/guidance.py
Explicit guidance control for ZS-RAG.

Features:
1. Split forward passes (Uncond/Pos/Neg) with independent gating controls.
2. Supports Contrastive Guidance with mask-aware gating.
3. Ensures Uncond pass is always global (ungated) to maintain background consistency.
"""

import torch
from typing import Optional, Dict, Any

@torch.no_grad()
def _unet_forward(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Single UNet forward pass wrapper. 
    Compatible with diffusers return structure, disables gradient for inference VRAM saving.
    """
    out = unet(
        latents,
        t,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    )
    if isinstance(out, dict):
        return out.get("sample", out.get("out"))
    return getattr(out, "sample", out)

@torch.no_grad()
def forward_with_gate(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]],
    attn_proc=None,
    *,
    gate_enabled: bool = False,
    mode: str = "off",
    probe: bool = False,
    region_mask: Optional[torch.Tensor] = None,
    subject_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Single forward pass with RACA gating control.
    
    Args:
        gate_enabled: Whether to enable Spatial Gating for Cross-Attn.
        mode: AttentionProcessor mode ("off" | "record_bg" | "inject_subject").
        probe: Whether to enable Cross-Attn probe accumulation.
        region_mask: Spatial mask for gating (base latent scale). None = No gating.
        subject_id: Subject ID for probing and optional token indexing.
    """
    # Set Processor State
    if attn_proc is not None:
        if subject_id is not None:
            attn_proc.set_subject_id(subject_id)
        
        attn_proc.set_mode(mode)
        attn_proc.set_cross_gate_enabled(gate_enabled)
        attn_proc.set_probe_enabled(probe)
        attn_proc.set_region_mask(region_mask)

    # Forward
    eps = _unet_forward(unet, latents, t, encoder_hidden_states, added_cond_kwargs)

    # Optional: We don't forcefully clear state here; caller manages state transitions.
    return eps

@torch.no_grad()
def compute_cfg_raca(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    emb_pos: torch.Tensor,
    emb_uncond: torch.Tensor,
    added_cond_kwargs_pos: Optional[Dict[str, torch.Tensor]],
    added_cond_kwargs_uncond: Optional[Dict[str, torch.Tensor]],
    attn_proc=None,
    *,
    cfg_pos: float = 7.5,
    region_mask: Optional[torch.Tensor] = None,
    subject_id: Optional[int] = None,
    probe_pos: bool = False,
    # Strategy Switches: Uncond ungated, Pos gated (Default)
    gate_uncond: bool = False,
    gate_pos: bool = True,
    mode_uncond: str = "off",
    mode_pos: str = "inject_subject",
) -> torch.Tensor:
    """
    RACA version of Standard CFG (Two Passes).
    
    - Uncond Pass: Default ungated (gate_uncond=False), mode="off".
    - Pos Pass: Default gated (gate_pos=True), mode="inject_subject".
    """
    # 1. Uncond Pass (Global, Ungated)
    eps_uncond = forward_with_gate(
        unet,
        latents,
        t,
        emb_uncond,
        added_cond_kwargs_uncond,
        attn_proc=attn_proc,
        gate_enabled=gate_uncond,
        mode=mode_uncond,
        probe=False,            # Uncond never probes
        region_mask=None,       # No mask = Global context
        subject_id=subject_id,
    )

    # 2. Pos Pass (Local, Gated)
    eps_pos = forward_with_gate(
        unet,
        latents,
        t,
        emb_pos,
        added_cond_kwargs_pos,
        attn_proc=attn_proc,
        gate_enabled=gate_pos,
        mode=mode_pos,
        probe=probe_pos,        # Enable probe if requested
        region_mask=region_mask,
        subject_id=subject_id,
    )

    # Combine
    return eps_uncond + cfg_pos * (eps_pos - eps_uncond)

@torch.no_grad()
def compute_contrastive_cfg_raca(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    emb_pos: torch.Tensor,
    emb_uncond: torch.Tensor,
    emb_neg_target: torch.Tensor,
    added_cond_kwargs_pos: Optional[Dict[str, torch.Tensor]],
    added_cond_kwargs_uncond: Optional[Dict[str, torch.Tensor]],
    added_cond_kwargs_neg: Optional[Dict[str, torch.Tensor]],
    attn_proc=None,
    *,
    cfg_pos: float = 7.5,
    cfg_neg: float = 3.0,
    region_mask: Optional[torch.Tensor] = None,
    subject_id: Optional[int] = None,
    probe_pos: bool = False,
    probe_neg: bool = False,
    # Strategy Switches: Uncond ungated, Pos/Neg gated (Default)
    gate_uncond: bool = False,
    gate_pos: bool = True,
    gate_neg: bool = True,
    mode_uncond: str = "off",
    mode_pos: str = "inject_subject",
    mode_neg: str = "inject_subject",
) -> torch.Tensor:
    """
    RACA version of Contrastive CFG (Three Passes).
    eps = uncond + cfg_pos*(pos-uncond) - cfg_neg*(neg-uncond)
    
    - Uncond: Default ungated.
    - Pos: Default gated.
    - Neg: Default gated (push away from negative target ONLY in the region).
    
    Note: added_cond_kwargs_neg MUST be passed for the negative target pass.
    """
    # 1. Uncond Pass (Global)
    eps_uncond = forward_with_gate(
        unet,
        latents,
        t,
        emb_uncond,
        added_cond_kwargs_uncond,
        attn_proc=attn_proc,
        gate_enabled=gate_uncond,
        mode=mode_uncond,
        probe=False,
        region_mask=None,
        subject_id=subject_id,
    )

    # 2. Pos Pass (Local)
    eps_pos = forward_with_gate(
        unet,
        latents,
        t,
        emb_pos,
        added_cond_kwargs_pos,
        attn_proc=attn_proc,
        gate_enabled=gate_pos,
        mode=mode_pos,
        probe=probe_pos,
        region_mask=region_mask,
        subject_id=subject_id,
    )

    # 3. Neg Target Pass (Local)
    eps_neg = forward_with_gate(
        unet,
        latents,
        t,
        emb_neg_target,
        added_cond_kwargs_neg,      # Critical: Use Neg pooled embeds
        attn_proc=attn_proc,
        gate_enabled=gate_neg,
        mode=mode_neg,
        probe=probe_neg,
        region_mask=region_mask,
        subject_id=subject_id,
    )

    # Combine
    # Logic: Start from Uncond baseline.
    # Add vector towards Pos.
    # Subtract vector towards Neg Target.
    return eps_uncond + cfg_pos * (eps_pos - eps_uncond) - cfg_neg * (eps_neg - eps_uncond)

def prepare_timesteps(scheduler, num_inference_steps: int, device: torch.device):
    """
    Sets up scheduler timesteps and returns tensor.
    """
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    return timesteps
