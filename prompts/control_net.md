# RFC: Integrating ControlNet with Edit-Friendly DDPM Inversion

## 1. Executive Summary
The current "Edit Friendly DDPM" method suffers from an "under-editing" problem where the generated output is too strongly biased toward the original image's structure and texture. This occurs because the inverted noise maps ($z^{inv}$) encode the original image features too rigidly.

This proposal outlines the integration of **ControlNet**, allowing us to:
1.  **Decouple Structure from Texture:** Use ControlNet to enforce the pose/shape (Structure).
2.  **Relax Noise Constraints:** Blend the rigid inverted noise with random Gaussian noise (Texture/Style).
This hybrid approach allows for drastic stylistic edits (e.g., "photo of dog" $\to$ "origami dog") while maintaining pixel-perfect structural alignment.

---

## 2. Theoretical Improvement: The "Relaxed Inversion" Equation

### 2.1 The Current Limitation
Currently, the sampling step at time $t$ uses the inverted noise $z_t^{inv}$ exclusively:
$$x_{t-1} = \mu_\theta(x_t, t, \text{prompt}) + \sigma_t \cdot z_t^{inv}$$
Because $z_t^{inv}$ is not random (it contains edges and shadows of the original image), the model ignores the text prompt if the prompt conflicts with this "structural noise."

### 2.2 The Proposed Solution
We introduce a **Fidelity Factor ($\gamma$)** and an auxiliary **Control Condition ($c$)**.

The new sampling equation becomes:
$$x_{t-1} = \mu_\theta(x_t, t, \text{prompt} | c) + \sigma_t \cdot \left( \gamma \cdot z_t^{inv} + (1-\gamma) \cdot z_t^{rand} \right)$$

* **$\mu_\theta(\dots | c)$**: The UNet prediction is now guided by ControlNet (e.g., Canny/Depth) to ensure the shape stays correct.
* **$z_t^{rand}$**: Pure Gaussian noise that encourages the model to generate *new* textures matching the text prompt.
* **$\gamma$**: A slider (0.0 to 1.0).
    * **High $\gamma$ (e.g., 0.8):** Behaves like the original algorithm (safe, similar to input).
    * **Low $\gamma$ (e.g., 0.3):** Behaves like standard Stable Diffusion with ControlNet (creative, strong edits).

---

## 3. Implementation Plan

### Phase A: dependency & Setup
**Objective:** Prepare the environment and models without breaking the existing `DDPM_inversion` code structure.

1.  **Dependency Upgrade:**
    * Upgrade `diffusers` library to $\ge 0.14.0$ (Critical for `ControlNetModel`).
    * Install `controlnet_aux` for image pre-processing (Canny/Depth detectors).
2.  **Model Loading (`main_run.py`):**
    * Initialize the `ControlNetModel` (e.g., `lllyasviel/sd-controlnet-canny`) alongside the existing Stable Diffusion pipeline.
    * Ensure the ControlNet is moved to the same device (CUDA) and dtype (float16) as the main UNet.

### Phase B: Pre-processing Logic
**Objective:** Generate the structural guidance map before the diffusion loop begins.

1.  **Input Processing:**
    * In the setup phase (before the loop in `main_run.py`), take the source image $x_0$.
    * Run the detector (e.g., Canny Edge Detector) to create the control map $c$.
    * **Important:** Resize/Crop $c$ to strictly match the resolution of $x_0$ (usually $512 \times 512$).
    * Normalize $c$ to the range $[0, 1]$ and convert to a generic tensor.

### Phase C: Core Integration (`ddm_inversion/` folder)
**Objective:** Modify the sampling loop to accept ControlNet residuals and blend noise.

*Target Location:* Look for the reverse diffusion function (likely named `p_sample_loop` or inside a class in `ddm_inversion/inversion_utils.py` or similar).

1.  **Step 1: Compute ControlNet Residuals**
    * Inside the sampling loop (looping from $T \to 0$), before the UNet prediction:
    * Pass the current noisy latent $x_t$, timestep $t$, and control map $c$ into the ControlNet model.
    * **Output:** Capture the `down_block_res_samples` and `mid_block_res_sample`.

2.  **Step 2: Inject into UNet**
    * Pass these residuals into the main UNet's forward pass.
    * *Note:* The standard `diffusers` UNet accepts `down_block_additional_residuals` and `mid_block_additional_residual` arguments.

3.  **Step 3: Modify the Step Update (The "Relaxation")**
    * Locate the line where $z_t^{inv}$ (the pre-calculated noise) is added.
    * Generate a new random noise tensor $z_t^{rand}$ of the same shape.
    * Implement the blending formula: `combined_noise` = ($\gamma \times z_t^{inv}$) + ($(1-\gamma) \times z_t^{rand}$).
    * Use `combined_noise` for the final $x_{t-1}$ update.

### Phase D: Interface Updates
**Objective:** Expose the new controls to the user.

1.  **Arguments:**
    * Add `--control_gamma` to `test.yaml` or `main_run.py` arguments. Default to `0.5`.
    * Add `--control_type` (e.g., "canny", "depth", "pose") to select the ControlNet version.