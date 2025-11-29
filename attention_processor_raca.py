"""
zsrag/core/attention_processor_raca.py
RACA++: Region-Aware Cross-Attention Processor for ZS-RAG.

Features:
1. Controllable Cross-Attn Gating: Can be explicitly toggled (off for uncond, on for pos/neg).
2. Robust Probing: Accumulates maps only when enabled; supports mean aggregation to bypass token index issues.
3. Safe Self-Attn Injection: Optional background injection only applied outside subject masks.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextBank:
    """
    Stores Self-Attn K/V for optional background injection.
    Stores in float16 to save VRAM.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}

    def clear(self):
        self._store.clear()

    def put(self, layer_id: str, K: torch.Tensor, V: torch.Tensor):
        # Store in FP16 to save memory; convert back during retrieval
        self._store[layer_id] = {
            "K": K.detach().to(dtype=torch.float16),
            "V": V.detach().to(dtype=torch.float16),
        }

    def has(self, layer_id: str) -> bool:
        return layer_id in self._store

    def get(self, layer_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._store[layer_id]["K"], self._store[layer_id]["V"]

class AlphaScheduler:
    """
    Scheduler for injection strength.
    Note: Cross-Attn gating strength is usually binary (mask) or controlled by mask weights,
    this scheduler is primarily for Self-Attn injection if enabled.
    """
    def __init__(
        self, 
        construct_phase=(0.0, 0.35, 0.15), 
        texture_phase=(0.35, 0.75, 0.6), 
        refine_phase=(0.75, 1.0, 0.45), 
        per_layer_scaling=None
    ):
        self.phases = [construct_phase, texture_phase, refine_phase]
        self.per_layer_scaling = per_layer_scaling or {"lowres": 1.0, "midres": 0.6, "hires": 0.35}

    def alpha(self, step_idx: int, total_steps: int, layer_id: str) -> float:
        frac = step_idx / max(total_steps - 1, 1)
        val = self.phases[-1][2]
        is_refine = False
        for start, end, a in self.phases:
            if start <= frac <= end:
                val = a
                if start >= 0.8:
                    is_refine = True
                break
        
        lid = layer_id or ""
        if "down_blocks.2" in lid or "mid_block" in lid or "up_blocks.0" in lid:
            scale = self.per_layer_scaling.get("midres", 0.6)
        elif "down_blocks.0" in lid or "down_blocks.1" in lid:
            scale = self.per_layer_scaling.get("lowres", 1.0)
        else:
            scale = self.per_layer_scaling.get("hires", 0.35) if is_refine else 0.0
        return float(val * scale)

class LayerSelector:
    """
    Selects which layers to apply injection or probing.
    """
    def __init__(self, patterns=None, probe_patterns=None):
        # Default covers low/mid layers and the first upsample layer.
        # Avoiding finest layers (up.2) prevents rigid texture locking.
        self.patterns = patterns or ["down_blocks.1", "down_blocks.2", "mid_block", "up_blocks.0", "up_blocks.1"]
        self.probe_patterns = probe_patterns or ["down_blocks.2", "mid_block"]

    def for_inject(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.patterns)

    def for_probe(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.probe_patterns)

def _infer_scale_from_layer_id(layer_id: str) -> int:
    """
    Infers spatial downsampling scale from SDXL layer name.
    """
    if not layer_id: return 1
    if "mid_block" in layer_id: return 8
    
    m = re.search(r"down_blocks.(\d+)", layer_id)
    if m: return 2 ** int(m.group(1))
    
    m2 = re.search(r"up_blocks.(\d+)", layer_id)
    if m2: 
        # SDXL up blocks mapping might need adjustment based on specific config,
        # but standard is: 0->4x, 1->2x, 2->1x relative to base latent?
        # Actually SDXL base latent is 1/8 of image.
        # mid is 1/8 (scale 8 relative to image, 1 relative to latent).
        # Let's assume input layer_id structure matches standard Diffusers SDXL.
        return {0: 4, 1: 2, 2: 1}.get(int(m2.group(1)), 1)
    return 1

class RACAAttentionProcessor(nn.Module):
    """
    Region-Aware Cross-Attention Processor (RACA++).
    """
    def __init__(
        self,
        context_bank: Optional[ContextBank] = None,
        alpha_scheduler: Optional[AlphaScheduler] = None,
        layer_selector: Optional[LayerSelector] = None,
    ):
        super().__init__()
        self.context_bank = context_bank or ContextBank()
        self.alpha_scheduler = alpha_scheduler or AlphaScheduler()
        self.layer_selector = layer_selector or LayerSelector()

        # Runtime State
        self.mode: str = "off"  # "off", "record_bg", "inject_subject"
        self.step_idx: int = 0
        self.total_steps: int = 50
        self.subject_id: Optional[int] = None
        self.base_latent_hw: Optional[Tuple[int, int]] = None

        # Control Flags
        self._probe_enabled: bool = False
        self._cross_gate_enabled: bool = False  # Explicit toggle for Cross-Attn gating
        self._key_agg_mode: str = "mean"        # "mean" or "indices"
        self.subject_token_ids: Dict[int, List[int]] = {}
        self._attn_accumulator: Dict[int, torch.Tensor] = {}

        # Masks
        self._region_mask: Optional[torch.Tensor] = None  # [1,1,H0,W0]

        # Options
        self.enable_self_inject: bool = False  # Default False

    # -------------------------
    # Setters
    # -------------------------
    def set_mode(self, mode: str):
        self.mode = mode

    def set_step_index(self, idx: int, total: int):
        self.step_idx, self.total_steps = int(idx), int(total)

    def set_subject_id(self, i: Optional[int]):
        self.subject_id = i

    def set_subject_token_ids(self, mapping: Dict[int, List[int]]):
        self.subject_token_ids = mapping or {}

    def set_base_latent_hw(self, h: int, w: int):
        self.base_latent_hw = (int(h), int(w))

    def set_region_mask(self, mask: Optional[torch.Tensor]):
        """
        Sets the binary mask for the current subject. 
        Shape: [1, 1, H_latent, W_latent].
        """
        self._region_mask = mask

    def set_probe_enabled(self, flag: bool):
        """
        Enables attention probing. Clears accumulator only on False->True transition.
        """
        flag = bool(flag)
        if flag and not self._probe_enabled:
            self._attn_accumulator.clear()
        self._probe_enabled = flag

    def set_cross_gate_enabled(self, flag: bool):
        """
        Explicitly toggles Spatial Gating for Cross-Attention.
        Enable for Pos/Neg pass, Disable for Uncond pass.
        """
        self._cross_gate_enabled = bool(flag)

    def set_key_agg_mode(self, mode: str):
        """
        "indices": Use specific token IDs (hard for SDXL).
        "mean": Average over all text tokens (robust).
        """
        if mode in ("indices", "mean"):
            self._key_agg_mode = mode

    def set_self_inject_enabled(self, flag: bool):
        """
        If True, injects background Self-Attn features outside the subject mask.
        """
        self.enable_self_inject = bool(flag)

    @torch.no_grad()
    def pop_attn_maps(self) -> Dict[int, torch.Tensor]:
        """
        Returns normalized attention maps and clears accumulator.
        """
        out = {}
        for sid, amap in self._attn_accumulator.items():
            mi, ma = amap.min(), amap.max()
            out[sid] = (amap - mi) / (ma - mi + 1e-8)
        self._attn_accumulator.clear()
        return out

    # -------------------------
    # Helpers
    # -------------------------
    def _align_bg_batch(self, key, K_bg, V_bg):
        if K_bg.shape[0] != key.shape[0]:
            r = key.shape[0] // K_bg.shape[0]
            K_bg, V_bg = K_bg.repeat(r, 1, 1), V_bg.repeat(r, 1, 1)
        return K_bg, V_bg

    def _sdpa(self, q, k, v, scale=None):
        if scale:
            q = q * scale * math.sqrt(q.shape[-1])
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    def _manual_attn(self, q, k, v, scale):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
        attn_probs = attn_scores.softmax(dim=-1)
        context = torch.bmm(attn_probs, v)
        return context, attn_probs

    def _make_query_mask(self, layer_id: str, q: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Downsamples the region mask to match the spatial resolution of query 'q'.
        Returns [BxH, Lq] or None.
        """
        if self._region_mask is None or self.base_latent_hw is None:
            return None
        H0, W0 = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, H0 // s), max(1, W0 // s)
        Lq = q.shape[1]
        
        # Only apply if dimensions match (spatial attention)
        if Lq != h * w:
            return None
            
        m = F.interpolate(self._region_mask, size=(h, w), mode="bilinear", align_corners=False)
        m1d = m.reshape(1, 1, -1)
        BxH = q.shape[0]
        m1d = m1d.expand(BxH, -1, -1).squeeze(1)  # [BxH, Lq]
        return m1d.clamp(0.0, 1.0)

    def _probe_cross_attn(self, attn_probs, layer_id, batch_size):
        """
        Accumulates attention maps for the current subject.
        """
        if not (self._probe_enabled and self.layer_selector.for_probe(layer_id) and self.subject_id is not None):
            return

        if self._key_agg_mode == "indices":
            token_ids = self.subject_token_ids.get(self.subject_id, [])
            if not token_ids:
                sel = attn_probs.mean(dim=-1)
            else:
                idx = torch.tensor(token_ids, dtype=torch.long, device=attn_probs.device)
                sel = attn_probs.index_select(dim=-1, index=idx).mean(dim=-1)
        else:
            # Robust aggregation: mean over all keys
            sel = attn_probs.mean(dim=-1)

        bxh, Lq = sel.shape
        heads = bxh // max(1, batch_size)
        
        # Average over heads and batch
        sel = sel.reshape(batch_size, heads, Lq).mean(dim=1)  # [B, Lq]
        amap_1d = sel[0:1]  # Take first sample in batch

        if self.base_latent_hw is None:
            return
            
        h0, w0 = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, h0 // s), max(1, w0 // s)
        
        # Verify shape matches
        if amap_1d.shape[-1] != h * w:
            # Fallback for non-square or mismatched shapes if needed
            sq = int(math.sqrt(amap_1d.shape[-1]))
            if sq * sq == amap_1d.shape[-1]:
                h, w = sq, sq
            else:
                return
                
        amap_2d = amap_1d.reshape(1, 1, h, w)
        amap_up = F.interpolate(amap_2d, size=(h0, w0), mode="bilinear", align_corners=False)
        
        prev = self._attn_accumulator.get(self.subject_id, None)
        self._attn_accumulator[self.subject_id] = amap_up if prev is None else (prev + amap_up)

    def _apply_self_attn_injection(self, layer_id: str, q, k, v, attention_mask):
        """
        Optional: Injects background features ONLY outside the subject mask.
        """
        if (self.mode == "inject_subject"
            and self.enable_self_inject
            and self.layer_selector.for_inject(layer_id)
            and self.context_bank.has(layer_id)):
            
            K_bg, V_bg = self.context_bank.get(layer_id)
            K_bg = K_bg.to(k.dtype).to(k.device)
            V_bg = V_bg.to(v.dtype).to(v.device)
            
            K_bg, V_bg = self._align_bg_batch(k, K_bg, V_bg)
            alpha = self.alpha_scheduler.alpha(self.step_idx, self.total_steps, layer_id)
            
            if alpha > 0.0:
                qm = self._make_query_mask(layer_id, q)
                if qm is not None:
                    # Inverted mask: 1.0 where we want to inject background (outside subject)
                    inv_qm = (1.0 - qm).unsqueeze(-1) # [BxH, Lq, 1]
                    
                    # This logic appends BG tokens. 
                    # To strictly enforce spatial structure, simple concat might not be enough 
                    # without modifying attention bias, but for "style/texture" injection 
                    # this mimics standard ControlNet/Adapter behavior.
                    # We modulate injection by alpha * inverse_mask.
                    
                    # Note: Concatenating changes sequence length. 
                    # If we want replacement, we need geometric blending. 
                    # Here we stick to concat injection (standard practice).
                    
                    k = torch.cat([k, (alpha * K_bg)], dim=1)
                    v = torch.cat([v, (alpha * V_bg)], dim=1)
                    attention_mask = None
        return k, v, attention_mask

    # -------------------------
    # Main Forward
    # -------------------------
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs
    ):
        layer_id = getattr(attn, "layer_id", None)
        is_self_attn = encoder_hidden_states is None

        # Linear Projections
        q = attn.to_q(hidden_states)
        if is_self_attn:
            k, v = attn.to_k(hidden_states), attn.to_v(hidden_states)
        else:
            k, v = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

        q, k, v = attn.head_to_batch_dim(q), attn.head_to_batch_dim(k), attn.head_to_batch_dim(v)

        # 1. Self-Attention Path: Optional Injection
        if is_self_attn and self.layer_selector.for_inject(layer_id):
            if self.mode == "record_bg":
                self.context_bank.put(layer_id, k, v)
            elif self.mode == "inject_subject":
                k, v, attention_mask = self._apply_self_attn_injection(layer_id, q, k, v, attention_mask)

        scale = getattr(attn, "scale", 1.0 / math.sqrt(q.shape[-1]))

        # 2. Attention Calculation
        if is_self_attn:
            # Standard SDPA for Self-Attn (unless we implement sophisticated layout masking here)
            context = self._sdpa(q, k, v, scale)
        else:
            # Cross-Attention Path
            if self.layer_selector.for_inject(layer_id):
                # Manual calculation to allow gating/probing
                context, attn_probs = self._manual_attn(q, k, v, scale)

                # A. Probe
                if self._probe_enabled:
                    self._probe_cross_attn(attn_probs.detach(), layer_id, hidden_states.shape[0])

                # B. Spatial Gating (RACA)
                # Only apply if explicitly enabled (e.g. for Pos/Neg pass, but not Uncond)
                if self._cross_gate_enabled:
                    qm = self._make_query_mask(layer_id, q)
                    if qm is not None:
                        # Mask out attention probabilities outside the region
                        # [BxH, Lq, Lk] * [BxH, Lq, 1]
                        attn_probs = attn_probs * qm.unsqueeze(-1)
                        
                        # Re-normalize slightly to prevent total signal loss?
                        # Actually for masking, we WANT signal loss outside the mask 
                        # so that the region doesn't attend to this text prompt.
                        # But we need to avoid div by zero.
                        denom = attn_probs.sum(dim=-1, keepdim=True) + 1e-6
                        attn_probs = attn_probs / denom
                        
                        context = torch.bmm(attn_probs, v)
            else:
                # Optimized path for layers we don't care about
                context = self._sdpa(q, k, v, scale)

        # Output Projection
        return attn.to_out[0](attn.batch_to_head_dim(context))

def create_raca_attention_processor(context_bank=None, alpha_scheduler=None, layer_selector=None):
    return RACAAttentionProcessor(context_bank, alpha_scheduler, layer_selector)
