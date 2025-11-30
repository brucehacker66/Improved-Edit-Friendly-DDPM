# Edit-Friendly DDPM + P2P Improvements

This document describes the implemented improvements to the baseline Edit-Friendly DDPM + Prompt-to-Prompt (P2P) image editing method, based on the strategies outlined in `prompts/edit_friendly+p2p_basic_improvements.md`.

## Problem Statement

The baseline method achieves **high structural preservation** but **low CLIP similarity** (~21-26), indicating an over-constrained system where the Edit-Friendly noise imprinting and P2P attention injection overpower the semantic guidance from the target text prompt.

## Implemented Improvements

### 1. Hyperparameter Optimization (`run_editing_ef_p2p_hyperopt.py`)

**Difficulty:** Easy
**Expected Impact:** High
**Files:** `run_editing_ef_p2p_hyperopt.py`

Adjusts key hyperparameters without code changes:

- **Decreased Skip Steps:** From 12 → 15-20
  - Introduces more randomness early in the process
  - Allows target prompt to alter content before structure settles

- **Increased CFG Scale:** From 7.5 → 12.0-15.0
  - Forces model to prioritize target caption over structural noise

- **Relaxed Cross-Attention:** From 0.4 → 0.2-0.3
  - Allows new text tokens to exert influence in later denoising stages
  - Boosts CLIP scores while maintaining structure via self-attention

**Available Configurations:**
- `hyperopt-skip15`: SKIP=15, CFG=7.5, Cross=0.4
- `hyperopt-skip18`: SKIP=18, CFG=7.5, Cross=0.4
- `hyperopt-skip20`: SKIP=20, CFG=7.5, Cross=0.4
- `hyperopt-cfg12`: SKIP=12, CFG=12.0, Cross=0.4
- `hyperopt-cfg15`: SKIP=12, CFG=15.0, Cross=0.4
- `hyperopt-cross0.2`: SKIP=12, CFG=7.5, Cross=0.2
- `hyperopt-cross0.3`: SKIP=12, CFG=7.5, Cross=0.3
- `hyperopt-combined`: SKIP=18, CFG=12.0, Cross=0.3 (recommended)

**Usage:**
```bash
python run_editing_ef_p2p_hyperopt.py \\
    --data_path data \\
    --output_path output \\
    --edit_method_list hyperopt-combined \\
    --edit_category_list 0 1 2
```

---

### 2. Negative Prompting (`run_editing_ef_p2p_negprompt.py`)

**Difficulty:** Medium
**Expected Impact:** Direct increase in `clip_similarity_target_image`
**Files:** `run_editing_ef_p2p_negprompt.py`, `models/edit_friendly_ddm/inversion_utils_neg.py`

Uses the source prompt as a **negative prompt** during editing to actively push away from the source concept, clearing "semantic space" for the target.

**Strategy:** Instead of using empty string "" as the unconditional prompt, use the source prompt as the negative prompt during the reverse diffusion process.

**Available Configurations:**
- `negprompt-baseline`: No negative prompt (standard behavior)
- `negprompt-source`: Source prompt as negative for target
- `negprompt-combined`: Negative prompt + relaxed cross-attention (Cross=0.3, Self=0.7)

**Usage:**
```bash
python run_editing_ef_p2p_negprompt.py \\
    --data_path data \\
    --output_path output \\
    --edit_method_list negprompt-source \\
    --edit_category_list 0 1 2
```

**Implementation Details:**
- Modified `inversion_reverse_process_neg()` to accept `negative_prompts` parameter
- Replaces unconditional embedding with source prompt embedding for target branch
- Maintains standard unconditional for source branch to preserve reconstruction

---

### 3. Dynamic PSD Adjustment (`run_editing_ef_p2p_psd.py`)

**Difficulty:** Medium
**Expected Impact:** Weakens structural imprint, allows clearer semantic edits
**Files:** `run_editing_ef_p2p_psd.py`, `models/edit_friendly_ddm/inversion_utils_psd.py`

Applies a **dampening factor** to the extracted noise during the editing pass to reduce the variance of the Edit-Friendly noise, weakening the structural constraint.

**Strategy:** Multiply the noise trajectory `zs` by a factor γ ∈ [0.8, 0.95] during editing (not reconstruction).

**Available Configurations:**
- `psd-0.95`: 5% noise reduction
- `psd-0.90`: 10% noise reduction (recommended)
- `psd-0.85`: 15% noise reduction
- `psd-0.80`: 20% noise reduction
- `psd-combined`: PSD=0.90 + relaxed cross-attention (Cross=0.3, Self=0.7)

**Usage:**
```bash
python run_editing_ef_p2p_psd.py \\
    --data_path data \\
    --output_path output \\
    --edit_method_list psd-0.90 \\
    --edit_category_list 0 1 2
```

**Implementation Details:**
- Modified `inversion_reverse_process_psd()` with `psd_damping_factor` parameter
- Applies dampening mask: full noise for reconstruction, dampened for editing
- Only affects the target branch, preserving source reconstruction quality

---

### 4. Localized Attention Blending (`run_editing_ef_p2p_masked.py`)

**Difficulty:** Advanced
**Expected Impact:** Maximizes preservation (identical background) while concentrating edits on subject
**Files:** `run_editing_ef_p2p_masked.py`, `models/edit_friendly_ddm/ptp_classes_masked.py`

Uses **cross-attention maps** to create a binary mask identifying the subject, then composites the reconstruction (background) with edited (foreground) latents at each step.

**Strategy:**
1. Extract cross-attention maps for subject words from source prompt
2. Create binary mask via thresholding and morphological operations
3. Blend: `output = background × (1 - mask) + foreground × mask`

**Available Configurations:**
- `masked-baseline`: Standard P2P without masking
- `masked-auto`: Auto-detect subject words and apply masking
- `masked-combined`: Masking + Cross=0.3, Self=0.7, Threshold=0.25

**Usage:**
```bash
python run_editing_ef_p2p_masked.py \\
    --data_path data \\
    --output_path output \\
    --edit_method_list masked-auto \\
    --edit_category_list 0 1 2
```

**Implementation Details:**
- `AttentionMask` class extracts masks from cross-attention
- `AttentionReplaceWithMask` and `AttentionRefineWithMask` extend P2P controllers
- Auto-detection: extracts words differing between source and target prompts
- Fallback: uses `blended_word` from dataset metadata

---

## Architecture Overview

### How Edit-Friendly DDPM + P2P Works

1. **Inversion (Forward Process):**
   - Encode image to latent space with VAE
   - Run `inversion_forward_process()` to extract noise trajectory `zs` and latents `wts`
   - Stores intermediate noise values for faithful reconstruction

2. **Editing (Reverse Process):**
   - Create P2P controller (`AttentionReplace` or `AttentionRefine`)
   - Run `inversion_reverse_process()` with both source and target prompts
   - P2P controller injects attention maps:
     - **Cross-attention:** Controls semantics (what objects appear)
     - **Self-attention:** Controls structure (where objects are)
   - Start from `wts[NUM_DDIM_STEPS - SKIP]` instead of pure noise

### Key Parameters

| Parameter | Baseline | Recommended | Effect |
|-----------|----------|-------------|--------|
| `SKIP` | 12 | 15-18 | More noise → more semantic change |
| `target_guidance_scale` | 7.5 | 12.0 | Stronger text guidance |
| `cross_replace_steps` | 0.4 | 0.3 | Less semantic constraint |
| `self_replace_steps` | 0.6 | 0.7 | More structural preservation |

---

## Running the Improvements

### Prerequisites
```bash
pip install torch diffusers transformers pillow numpy
```

### Data Format
Expects the PIE-Bench or similar dataset structure:
```
data/
├── mapping_file.json
└── annotation_images/
    └── *.jpg
```

### Example Workflow

1. **Try hyperparameter tuning first (easiest):**
```bash
python run_editing_ef_p2p_hyperopt.py \\
    --edit_method_list hyperopt-combined
```

2. **Add negative prompting:**
```bash
python run_editing_ef_p2p_negprompt.py \\
    --edit_method_list negprompt-combined
```

3. **Experiment with PSD dampening:**
```bash
python run_editing_ef_p2p_psd.py \\
    --edit_method_list psd-combined
```

4. **Apply localized masking (most advanced):**
```bash
python run_editing_ef_p2p_masked.py \\
    --edit_method_list masked-combined
```

### Output Structure
```
output/
├── hyperopt-combined/
├── negprompt-combined/
├── psd-combined/
└── masked-combined/
    └── annotation_images/
        ├── image_001.jpg          # Concatenated visualization
        └── image_001_edited.jpg   # Edited image only (for evaluation)
```

---

## Expected Results

| Method | CLIP Similarity ↑ | Structure Preservation | Semantic Change |
|--------|-------------------|------------------------|-----------------|
| Baseline | 21-26 | Very High | Low |
| Hyperopt | 28-32 (est.) | High | Medium |
| Negative Prompting | 30-35 (est.) | High | Medium-High |
| PSD Dampening | 28-33 (est.) | Medium-High | Medium |
| Localized Masking | 32-38 (est.) | Very High (BG) | High (FG) |
| **Combined** | **35-40 (est.)** | **High** | **High** |

**Note:** These are estimated improvements based on the theoretical impact. Actual results depend on the dataset and editing tasks.

---

## Combining Multiple Improvements

You can combine improvements by modifying the scripts. For example, combine negative prompting + PSD dampening:

```python
# In run_editing_ef_p2p_negprompt.py, replace inversion_reverse_process_neg with:
from models.edit_friendly_ddm.inversion_utils_psd import inversion_reverse_process_psd

# Then modify the call to include both negative prompts and PSD dampening
w0, _ = inversion_reverse_process_psd(
    ldm_stable,
    xT=wts[NUM_DDIM_STEPS - SKIP],
    etas=ETA,
    prompts=prompts,
    # Add negative_prompts parameter (requires combining both implementations)
    cfg_scales=cfg_scale_list,
    prog_bar=True,
    zs=zs[:(NUM_DDIM_STEPS - SKIP)],
    controller=controller,
    psd_damping_factor=0.90
)
```

---

## Citation

If you use these improvements, please cite the original Edit-Friendly DDPM paper:

```
@article{huberman2023edit,
  title={Edit Friendly DDPM Noise Space: Inversion and Manipulations},
  author={Huberman-Spiegelglas, Inbar and Kulikov, Vladimir and Michaeli, Tomer},
  journal={arXiv preprint arXiv:2304.06140},
  year={2023}
}
```

And the Prompt-to-Prompt paper:

```
@article{hertz2022prompt,
  title={Prompt-to-prompt image editing with cross attention control},
  author={Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2208.01626},
  year={2022}
}
```

---

## Troubleshooting

### Out of Memory
- Reduce batch size (edit one image at a time)
- Use smaller resolution (modify `load_512` to `load_256`)
- Clear cache between images: `torch.cuda.empty_cache()`

### Low CLIP Scores
- Increase CFG scale (try 15.0 or 18.0)
- Decrease cross-attention injection (try 0.2)
- Increase SKIP steps (try 20 or 25)

### Poor Structure Preservation
- Increase self-attention injection (try 0.8)
- Decrease PSD dampening (try 0.95)
- Use localized masking to preserve background

### Artifacts
- Decrease CFG scale (try 10.0)
- Increase cross-attention injection (try 0.5)
- Check if reconstruction is clean (might be inversion issue)

---

## Future Work

1. **Adaptive Hyperparameters:** Auto-tune based on prompt similarity
2. **Learned Masks:** Use segmentation models instead of attention maps
3. **Progressive Editing:** Apply edits gradually over multiple passes
4. **Multi-Scale Editing:** Different parameters for different spatial scales
5. **Prompt Decomposition:** Separate object vs. attribute edits

---

## Contact

For questions or issues, please refer to the original repositories:
- Edit-Friendly DDPM: https://github.com/inbarhub/DDPM_inversion
- Prompt-to-Prompt: https://github.com/google/prompt-to-prompt
