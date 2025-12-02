"""
zsrag/core/attention_processor_raca.py
Concept-Weaver Attention Processor (Feature Injection).
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import shift_tensor

class FeatureBank:
    """
    Stores K/V features for multiple subjects/background.
    Structure: { layer_id: { "bg": (K,V), "cat": (K,V), ... } }
    """
    def __init__(self):
        self.store = {} 

    def clear(self):
        self.store.clear()

    def record(self, layer_id: str, name: str, key: torch.Tensor, value: torch.Tensor):
        if layer_id not in self.store:
            self.store[layer_id] = {}
        # Store in FP16 to save memory
        self.store[layer_id][name] = (key.detach().cpu().half(), value.detach().cpu().half())

    def get(self, layer_id: str, name: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v = self.store[layer_id][name]
        return k.to(device), v.to(device)

    def has(self, layer_id: str, name: str) -> bool:
        return layer_id in self.store and name in self.store[layer_id]

class ConceptWeaverProcessor(nn.Module):
    def __init__(self, feature_bank: FeatureBank, layer_patterns: List[str] = None):
        super().__init__()
        self.bank = feature_bank
        self.layer_patterns = layer_patterns or ["up_blocks", "mid_block"] # Inject in deeper layers
        
        # Mode: "record" (save features) or "weave" (inject features)
        self.mode = "off" 
        self.current_subject_name = None # For recording
        
        # For Weaving
        self.layout_map = {} # { "cat": (src_box, dst_box), ... }
        self.canvas_size = (1024, 1024)
        self.bg_name = "background"

    def set_mode(self, mode: str, subject: str = None):
        self.mode = mode
        self.current_subject_name = subject

    def set_layout(self, layout: Dict[str, Tuple], canvas_size: Tuple[int, int]):
        self.layout_map = layout
        self.canvas_size = canvas_size

    def should_process(self, layer_id: str) -> bool:
        return any(p in layer_id for p in self.layer_patterns)

    def _get_spatial_dim(self, L: int) -> Tuple[int, int]:
        s = int(math.sqrt(L))
        if s * s == L: return s, s
        # Handle non-square (aspect ratio) if needed, for now assume square or standard aspect
        return s, s 

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
        layer_id = getattr(attn, "layer_id", "")
        is_self_attn = encoder_hidden_states is None

        # 1. Standard Projections
        q = attn.to_q(hidden_states)
        if is_self_attn:
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
        else:
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # 2. Logic Branch
        if is_self_attn and self.should_process(layer_id):
            
            # --- RECORD MODE ---
            if self.mode == "record" and self.current_subject_name:
                self.bank.record(layer_id, self.current_subject_name, k, v)
                
            # --- WEAVE MODE ---
            elif self.mode == "weave":
                # Only inject if we have features for background
                if self.bank.has(layer_id, self.bg_name):
                    
                    # 2.1 Get Base (Background) Features
                    k_out, v_out = self.bank.get(layer_id, self.bg_name, k.device)
                    # Align dtype
                    k_out = k_out.to(k.dtype)
                    v_out = v_out.to(v.dtype)
                    
                    # 2.2 Inject Subjects
                    # Iterate over layout to paste subjects
                    # We need to reshape [B*H, L, D] -> [B*H, C, h, w] for spatial shift
                    B_H, L, D = k_out.shape
                    h, w = self._get_spatial_dim(L)
                    
                    if h*w == L: # Only proceed if spatial dimensions match
                        k_spatial = k_out.permute(0, 2, 1).view(B_H, D, h, w)
                        v_spatial = v_out.permute(0, 2, 1).view(B_H, D, h, w)
                        
                        # Start with Background as canvas
                        k_final = k_spatial.clone()
                        v_final = v_spatial.clone()
                        
                        for name, (src_box, dst_box) in self.layout_map.items():
                            if self.bank.has(layer_id, name):
                                k_subj, v_subj = self.bank.get(layer_id, name, k.device)
                                k_subj = k_subj.to(k.dtype).permute(0, 2, 1).view(B_H, D, h, w)
                                v_subj = v_subj.to(v.dtype).permute(0, 2, 1).view(B_H, D, h, w)
                                
                                # Shift features
                                k_shifted = shift_tensor(k_subj, src_box, dst_box, self.canvas_size)
                                v_shifted = shift_tensor(v_subj, src_box, dst_box, self.canvas_size)
                                
                                # Soft Blend Mask
                                # Create a mask for the destination box in feature space
                                mask = torch.zeros((1, 1, h, w), device=k.device, dtype=k.dtype)
                                # Map dst_box to feature scale
                                scale_x = w / self.canvas_size[0]
                                scale_y = h / self.canvas_size[1]
                                x1 = int(dst_box[0] * scale_x)
                                y1 = int(dst_box[1] * scale_y)
                                x2 = int(dst_box[2] * scale_x)
                                y2 = int(dst_box[3] * scale_y)
                                # Clip
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    mask[..., y1:y2, x1:x2] = 1.0
                                    # Optional: Feather mask
                                    # mask = _feather(mask) 
                                
                                # Blend: Canvas = Canvas * (1-M) + Subject * M
                                k_final = k_final * (1 - mask) + k_shifted * mask
                                v_final = v_final * (1 - mask) + v_shifted * mask
                        
                        # Reshape back to [B*H, L, D]
                        k = k_final.view(B_H, D, L).permute(0, 2, 1)
                        v = v_final.view(B_H, D, L).permute(0, 2, 1)

        # 3. Attention Calculation
        scale = getattr(attn, "scale", 1.0 / math.sqrt(q.shape[-1]))
        
        # Standard SDPA with potentially modified K/V
        # If we injected, K/V now contain combined features
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        return attn.to_out[0](attn.batch_to_head_dim(out))

def create_weaver_processor():
    return ConceptWeaverProcessor(FeatureBank())
