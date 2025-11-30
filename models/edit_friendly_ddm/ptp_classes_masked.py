"""
Extended P2P classes with localized attention blending support

This adds controllers that use attention-based masking to separate
background (preserved) from foreground (edited) regions.
"""

from typing import Optional, Union, Tuple, List, Dict
import models.edit_friendly_ddm.ptp_utils as ptp_utils
import models.edit_friendly_ddm.seq_aligner as seq_aligner
import torch
import torch.nn.functional as nnf
import abc
import numpy as np

# Import base classes
from models.edit_friendly_ddm.ptp_classes import (
    AttentionControl, AttentionStore, AttentionControlEdit,
    AttentionReplace, AttentionRefine, LocalBlend, MAX_NUM_WORDS
)


class AttentionMask:
    """
    Extracts and applies attention-based masks to separate foreground/background
    """

    def __init__(self, target_words: List[str], prompts: List[str], threshold=0.3, device=None, tokenizer=None):
        """
        Args:
            target_words: Words identifying the subject to edit (e.g., ["cat"])
            prompts: [source_prompt, target_prompt]
            threshold: Attention threshold for mask binarization
            device: Torch device
            tokenizer: Model tokenizer
        """
        self.target_words = target_words
        self.prompts = prompts
        self.threshold = threshold
        self.device = device
        self.tokenizer = tokenizer
        self.attention_maps = []

    def extract_mask_from_attention(self, attention_store, res=16):
        """
        Extract binary mask from cross-attention maps for target words

        Args:
            attention_store: AttentionStore with collected attention maps
            res: Resolution of attention maps (default 16 for 64x64 latent)

        Returns:
            Binary mask [1, 1, 64, 64] indicating subject region
        """
        # Get attention maps
        attention_maps = attention_store.get_average_attention()

        # Focus on cross-attention from mid and up blocks (most semantic)
        locations = ["up_cross"]
        maps_list = []

        for location in locations:
            for attn_map in attention_maps[location]:
                if attn_map.shape[1] == res ** 2:
                    # Shape: [batch, pixels, tokens]
                    maps_list.append(attn_map)

        if not maps_list:
            # Fallback: return uniform mask if no suitable attention maps
            return torch.ones(1, 1, 64, 64, device=self.device)

        # Average across layers
        avg_map = torch.stack(maps_list).mean(0)  # [batch, pixels, tokens]

        # Get word indices for target words in source prompt (index 0)
        word_indices = []
        for word in self.target_words:
            inds = ptp_utils.get_word_inds(self.prompts[0], word, self.tokenizer)
            word_indices.extend(inds.tolist())

        if not word_indices:
            # Fallback: no target words found
            return torch.ones(1, 1, 64, 64, device=self.device)

        # Extract attention for target words from source prompt (batch idx 0)
        source_attn = avg_map[0:1]  # [1, pixels, tokens]
        target_attn = source_attn[:, :, word_indices].mean(-1)  # [1, pixels]

        # Reshape to spatial dimensions
        mask = target_attn.reshape(1, res, res)  # [1, 16, 16]

        # Upsample to latent resolution (64x64)
        mask = nnf.interpolate(mask.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
        mask = mask.squeeze(0)  # [1, 64, 64]

        # Normalize and threshold
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = (mask > self.threshold).float()

        # Apply morphological operations to clean up mask
        kernel_size = 5
        mask = nnf.max_pool2d(mask.unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2)

        return mask  # [1, 1, 64, 64]


class AttentionReplaceWithMask(AttentionReplace):
    """
    AttentionReplace with localized masking for foreground/background separation
    """

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, model=None,
                 target_words: Optional[List[str]] = None, mask_threshold: float = 0.3):
        super(AttentionReplaceWithMask, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, model
        )
        self.target_words = target_words if target_words else []
        self.mask_threshold = mask_threshold
        self.attention_mask_extractor = AttentionMask(
            self.target_words, prompts, mask_threshold, model.device, model.tokenizer
        )
        self.current_mask = None
        self.reconstruction_latents = None

    def step_callback(self, x_t):
        """
        Override to apply masked blending between reconstruction and edited latents

        Args:
            x_t: Current latents [batch, C, H, W] where batch = [source, target]

        Returns:
            Blended latents
        """
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)

        # Apply attention-based masking
        if self.current_mask is not None and x_t.shape[0] >= 2:
            # x_t[0] = reconstruction (background)
            # x_t[1] = edited (foreground)
            # Blend: background * (1 - mask) + foreground * mask
            mask = self.current_mask.to(x_t.device)
            x_t[1:] = x_t[0:1] * (1 - mask) + x_t[1:] * mask

        return x_t

    def set_mask(self, mask):
        """Set the current attention mask"""
        self.current_mask = mask


class AttentionRefineWithMask(AttentionRefine):
    """
    AttentionRefine with localized masking for foreground/background separation
    """

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, model=None,
                 target_words: Optional[List[str]] = None, mask_threshold: float = 0.3):
        super(AttentionRefineWithMask, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, model
        )
        self.target_words = target_words if target_words else []
        self.mask_threshold = mask_threshold
        self.attention_mask_extractor = AttentionMask(
            self.target_words, prompts, mask_threshold, model.device, model.tokenizer
        )
        self.current_mask = None

    def step_callback(self, x_t):
        """
        Override to apply masked blending between reconstruction and edited latents
        """
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)

        # Apply attention-based masking
        if self.current_mask is not None and x_t.shape[0] >= 2:
            mask = self.current_mask.to(x_t.device)
            x_t[1:] = x_t[0:1] * (1 - mask) + x_t[1:] * mask

        return x_t

    def set_mask(self, mask):
        """Set the current attention mask"""
        self.current_mask = mask
