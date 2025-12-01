# Improvement Patch: Addressing Blur & "Cartoon" Artifacts

**Context:** After integrating ControlNet into the Edit-Friendly DDPM pipeline, outputs may appear blurry (low contrast) or "cartoonish" (plastic textures). This document outlines the mathematical correction and parameter adjustments required to resolve this.

## 1. The Mathematical Fix: Variance Preservation (Critical)
**The Problem:**
Simple linear blending of inverted noise and random noise (`gamma * z_inv + (1-gamma) * z_rand`) reduces the total variance of the noise tensor.
* Mathematically: If $\gamma=0.5$, the resulting variance is $0.5^2 + 0.5^2 = 0.5$.
* **Consequence:** The UNet receives a latent with $\sigma < 1.0$, causing it to "under-denoise," leading to washed-out, blurry results.

**The Solution:**
You must explicitly re-normalize the blended noise to unit variance ($\sigma \approx 1.0$) or use Spherical Linear Interpolation (SLERP). The explicit renormalization is easier to implement and robust.

**Code Implementation:**
Locate the noise blending step in your `p_sample_loop` and update it:

```python
# --- INSIDE SAMPLING LOOP ---

# 1. Blend the noise sources
# z_inv = Inverted noise from DDPM inversion
# z_rand = Fresh Gaussian noise
z_combined = (gamma * z_inv) + ((1 - gamma) * z_rand)

# 2. [CRITICAL FIX] Re-normalize to unit standard deviation
# This restores the energy of the noise map, fixing the blur.
z_combined = z_combined / z_combined.std() 

# 3. Use z_combined for the step update
x_prev = mu + sigma_t * z_combined

Parameter tunning:
Lower the ControlNet Weight or use Early Stopping.
Action: Reduce controlnet_conditioning_scale from 1.0 to 0.6 or 0.8.
Action: Set control_guidance_end to 0.8. This tells ControlNet: "Enforce the shape for the first 80% of the process, but let the model hallucinate fine details freely for the last 20%."



Resolution mismatch: 

Ensure your image_resolution in the pre-processor matches the generation size.
If generating at 512x512, ensure the Canny map is extracted at 512 resolution, not 2000.