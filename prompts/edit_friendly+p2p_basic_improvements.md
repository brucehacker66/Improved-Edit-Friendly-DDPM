# Strategies for Improving Editing Fidelity in Edit-Friendly DDPM + P2P

## Problem Diagnosis
The current baseline achieves high structural preservation but low CLIP similarity scores (approx. 21-26). This indicates an **over-constrained system** where the "Edit-Friendly" noise imprinting and Prompt-to-Prompt (P2P) attention injection are overpowering the semantic guidance of the target text prompt.

To balance the trade-off between **Structure (Preservation)** and **Semantics (Editing)**, we propose the following improvements, ordered from hyperparameter tuning to algorithmic interventions.

---

## 1. Hyperparameter Optimization (Low Effort, High Impact)

These adjustments require no code changes, only configuration updates during inference.

### A. Decrease Skip Steps ($T_{skip}$)
The $T_{skip}$ parameter defines the starting point of the reverse diffusion process.
* **Current State:** Likely set too high (e.g., $T_{skip}=36$), initializing generation with a "too clean" image that resists change.
* **Recommendation:** Lower $T_{skip}$ to **15-20**.
* **Effect:** Introduces more randomness early in the process, allowing the new text prompt to alter the image content more significantly before the structure settles.

### B. Increase Classifier-Free Guidance (CFG)
CFG controls how strictly the model follows the text prompt versus the unconditional prior.
* **Current State:** Standard values (7.5) may be insufficient against the high-variance "Edit-Friendly" noise.
* **Recommendation:** Increase CFG scale to **12.0 - 15.0**.
* **Effect:** Forces the model to prioritize the target caption over the imprinted structural noise.

### C. Relax Cross-Attention Injection
P2P injects attention maps from the source to the target.
* **Current State:** Likely injecting cross-attention for 80%+ of steps.
* **Recommendation:** Decouple Self-Attention and Cross-Attention schedules.
    * **Self-Attention (Structure):** Keep injection high (e.g., 0.8) to preserve the layout.
    * **Cross-Attention (Semantics):** Stop injection early (e.g., **0.4**).
* **Effect:** Allows the new text tokens (e.g., "Dog") to exert influence in the later denoising stages, significantly boosting CLIP scores.

---

## 2. Algorithmic Enhancements (Medium Effort)

These methods require minor modifications to the pipeline logic but provide robust improvements.

### A. Negative Prompting Strategy
Standard inversion often neglects negative prompts. Explicitly negating the source concept clears "semantic space" for the target.
* **Implementation:**
    ```python
    # Instead of negative_prompt="", use:
    target_negative_prompt = source_prompt 
    # Example: If editing "Cat" -> "Dog", set negative_prompt="Cat"
    ```
* **Expected Metric Gain:** Direct increase in `clip_similarity_target_image`.

### B. Dynamic PSD (Power Spectral Density) Adjustment
The "Edit-Friendly" noise ($z_t$) has artificially high variance to enforce structure. This variance is what fights the edit.
* **Implementation:** Apply a dampening factor $\gamma$ to the noise injection during the **editing** pass (not the reconstruction pass).
    ```python
    # In the reverse diffusion loop:
    z_t_edited = z_t_extracted * 0.9  # Slightly reduce variance
    ```
* **Effect:** Weakens the structural "imprint" slightly, allowing the text prompt to manifest more clearly.

### C. Localized Attention Blending
To prevent the prompt from trying to "fix" the background (which lowers CLIP scores by adding noise to empty areas), strict masking can be applied via attention maps.
* **Implementation:** Use the cross-attention maps of the source object token to create a binary mask. Composite the *Reconstruction* latents (background) with the *Edited* latents (foreground) at each step.
* **Effect:** Maximizes preservation metrics (background is identical) while concentrating all editing power on the subject.