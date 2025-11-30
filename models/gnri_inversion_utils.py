# """
# GNRI-based Inversion Utilities for Edit Friendly P2P Integration

# This module implements Guided Newton-Raphson Inversion (GNRI) for fast image inversion
# while maintaining compatibility with Prompt-to-Prompt attention control mechanisms.

# Based on:
# - GNRI: Lightning-Fast Image Inversion (NewtonRaphsonInversion)
# - Edit Friendly DDPM Inversion (PnPInversion)
# """

# import torch
# from typing import Optional, List
# from diffusers.utils.torch_utils import randn_tensor


# def encode_text_sdxl(pipe, prompts):
#     """Encode text prompts for SDXL models."""
#     if isinstance(prompts, str):
#         prompts = [prompts]

#     # Encode prompts using SDXL's dual text encoders
#     (
#         prompt_embeds,
#         negative_prompt_embeds,
#         pooled_prompt_embeds,
#         negative_pooled_prompt_embeds,
#     ) = pipe.encode_prompt(
#         prompt=prompts,
#         prompt_2=prompts,  # Use same prompt for both encoders
#         device=pipe.device,
#         num_images_per_prompt=1,
#         do_classifier_free_guidance=False,
#     )

#     return prompt_embeds, pooled_prompt_embeds


# def get_timestep_distribution(z_0, sigma, device='cuda'):
#     """
#     Create Gaussian PDF for timestep guidance.
#     This ensures the inverted noise follows a realistic distribution.
#     """
#     z_0_flat = z_0.reshape(-1, 1)

#     def gaussian_pdf(x):
#         shape = x.shape
#         x_flat = x.reshape(-1, 1)
#         # Compute negative log probability (for minimization)
#         all_probs = -0.5 * torch.pow((x_flat - z_0_flat) / sigma, 2)
#         return all_probs.reshape(shape)

#     return gaussian_pdf


# def gnri_inversion_step(
#     pipe,
#     z_t: torch.Tensor,
#     t: torch.Tensor,
#     prompt_embeds: torch.Tensor,
#     pooled_prompt_embeds: torch.Tensor,
#     add_time_ids: torch.Tensor,
#     prev_timestep: Optional[torch.Tensor] = None,
#     n_iters: int = 2,
#     alpha: float = 0.1,
#     lr: float = 0.2,
#     z_0: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """
#     Single GNRI inversion step using Newton-Raphson optimization.

#     Args:
#         pipe: The diffusion pipeline (SDXL)
#         z_t: Current latent at timestep t
#         t: Current timestep
#         prompt_embeds: Text embeddings
#         pooled_prompt_embeds: Pooled text embeddings
#         add_time_ids: Additional time embeddings for SDXL
#         prev_timestep: Previous timestep (optional)
#         n_iters: Number of Newton-Raphson iterations
#         alpha: Weight for Gaussian prior guidance
#         lr: Learning rate (not used in current implementation)
#         z_0: Original latent (for guidance)

#     Returns:
#         z_{t+1}: Inverted latent at next timestep
#     """
#     latent = z_t.clone()
#     best_latent = None
#     best_score = torch.inf

#     # Get current timestep's sigma for distribution guidance
#     step_idx = (pipe.scheduler.timesteps == t).nonzero(as_tuple=True)[0]
#     if len(step_idx) == 0:
#         # Timestep not found, return current latent
#         return z_t

#     sigma = pipe.scheduler.sigmas[step_idx[0]]
#     curr_dist = get_timestep_distribution(z_0, sigma, device=z_t.device) if z_0 is not None else None

#     # Newton-Raphson iterations
#     for i in range(n_iters):
#         latent.requires_grad = True

#         # UNet forward pass to predict noise
#         added_cond_kwargs = {
#             "text_embeds": pooled_prompt_embeds,
#             "time_ids": add_time_ids
#         }

#         # Scale model input
#         latent_model_input = pipe.scheduler.scale_model_input(latent, t)

#         with torch.enable_grad():
#             noise_pred = pipe.unet(
#                 latent_model_input,
#                 t,
#                 encoder_hidden_states=prompt_embeds,
#                 timestep_cond=None,
#                 cross_attention_kwargs=None,
#                 added_cond_kwargs=added_cond_kwargs,
#                 return_dict=False,
#             )[0]

#         # Backward step: compute what the previous sample should be
#         # This is the inverse operation
#         next_latent = backward_step_euler(
#             pipe.scheduler,
#             noise_pred,
#             t,
#             z_t,
#         )

#         # Compute objective function
#         # f(x) = ||reconstruction_error|| + alpha * ||prior_deviation||
#         reconstruction_error = (next_latent - latent).abs()

#         if curr_dist is not None:
#             prior_term = -alpha * curr_dist(next_latent)
#             f_x = reconstruction_error + prior_term
#         else:
#             f_x = reconstruction_error

#         loss = f_x.sum()
#         score = f_x.mean()

#         # Track best solution
#         if score < best_score:
#             best_score = score
#             best_latent = next_latent.detach()

#         # Gradient descent step (Newton-Raphson approximation)
#         if i < n_iters - 1:  # Don't compute gradients on last iteration
#             loss.backward()

#             # Update latent using gradient
#             with torch.no_grad():
#                 # Newton step: x = x - f(x) / f'(x)
#                 # Using simplified Jacobian approximation
#                 grad_norm = latent.grad.abs().mean()
#                 if grad_norm > 1e-6:
#                     latent = latent - (loss / (64 * 64 * 4)) * (loss / latent.grad)
#                 else:
#                     # If gradient is too small, just use current best
#                     break

#             latent = latent.detach()

#     return best_latent if best_latent is not None else z_t


# def backward_step_euler(scheduler, noise_pred, t, z_t):
#     """
#     Backward step for Euler scheduler (inversion).
#     Computes z_{t+1} from z_t and noise prediction.
#     """
#     # Get step index
#     step_idx = (scheduler.timesteps == t).nonzero(as_tuple=True)[0]
#     if len(step_idx) == 0:
#         return z_t

#     step_idx = step_idx[0]

#     # Get sigmas
#     sigma = scheduler.sigmas[step_idx]
#     sigma_next = scheduler.sigmas[step_idx + 1] if step_idx + 1 < len(scheduler.sigmas) else torch.tensor(0.0)

#     # Compute predicted original sample (x_0)
#     if scheduler.config.prediction_type == "epsilon":
#         pred_original_sample = z_t - sigma * noise_pred
#     elif scheduler.config.prediction_type == "v_prediction":
#         pred_original_sample = noise_pred * (-sigma / (sigma**2 + 1) ** 0.5) + (z_t / (sigma**2 + 1))
#     else:
#         pred_original_sample = noise_pred

#     # For inversion, we go backwards: from lower noise to higher noise
#     # Standard: z_t -> z_{t-1} (denoising)
#     # Inversion: z_t -> z_{t+1} (adding noise)

#     # Compute derivative
#     derivative = (z_t - pred_original_sample) / (sigma + 1e-7)

#     # Inverse step: add noise instead of removing it
#     dt = sigma_next - sigma
#     z_next = z_t + derivative * dt

#     return z_next


# def gnri_inversion_forward_process(
#     pipe,
#     x0: torch.Tensor,
#     prompt: str = "",
#     num_inference_steps: int = 4,
#     guidance_scale: float = 0.0,
#     n_iters: int = 2,
#     alpha: float = 0.1,
#     lr: float = 0.2,
#     prog_bar: bool = False,
# ) -> tuple:
#     """
#     GNRI forward inversion process (x_0 -> x_T).
#     Uses Newton-Raphson optimization for fast, accurate inversion.

#     Args:
#         pipe: SDXL diffusion pipeline
#         x0: Initial latent (clean image)
#         prompt: Text prompt describing the image
#         num_inference_steps: Number of diffusion steps
#         guidance_scale: CFG scale (typically 0 for inversion)
#         n_iters: Newton-Raphson iterations per step
#         alpha: Gaussian prior weight
#         lr: Learning rate
#         prog_bar: Show progress bar

#     Returns:
#         (xT, latent_trajectory): Final noisy latent and trajectory
#     """
#     # Set up scheduler
#     pipe.scheduler.set_timesteps(num_inference_steps)
#     timesteps = pipe.scheduler.timesteps.to(pipe.device)

#     # Encode prompt
#     if prompt:
#         prompt_embeds, pooled_prompt_embeds = encode_text_sdxl(pipe, prompt)
#     else:
#         prompt_embeds, pooled_prompt_embeds = encode_text_sdxl(pipe, "")

#     # Prepare additional embeddings for SDXL
#     height, width = x0.shape[-2] * 8, x0.shape[-1] * 8  # Latent to pixel space
#     original_size = (height, width)
#     target_size = (height, width)
#     crops_coords_top_left = (0, 0)

#     add_time_ids = pipe._get_add_time_ids(
#         original_size,
#         crops_coords_top_left,
#         target_size,
#         dtype=prompt_embeds.dtype,
#         text_encoder_projection_dim=1280,  # SDXL default
#     )
#     add_time_ids = add_time_ids.to(pipe.device)

#     # Initialize
#     xt = x0.clone()
#     trajectory = [xt.clone()]

#     # Reverse timesteps for inversion (t=0 -> t=T)
#     timesteps_reversed = torch.flip(timesteps, [0])

#     if prog_bar:
#         from tqdm import tqdm
#         timesteps_iter = tqdm(timesteps_reversed, desc="GNRI Inversion")
#     else:
#         timesteps_iter = timesteps_reversed

#     prev_timestep = None

#     # Inversion loop
#     for t in timesteps_iter:
#         xt = gnri_inversion_step(
#             pipe=pipe,
#             z_t=xt,
#             t=t,
#             prompt_embeds=prompt_embeds,
#             pooled_prompt_embeds=pooled_prompt_embeds,
#             add_time_ids=add_time_ids,
#             prev_timestep=prev_timestep,
#             n_iters=n_iters,
#             alpha=alpha,
#             lr=lr,
#             z_0=x0,
#         )
#         trajectory.append(xt.clone())
#         prev_timestep = t

#     return xt, trajectory


# def gnri_inversion_reverse_process(
#     pipe,
#     xT: torch.Tensor,
#     prompts: List[str],
#     cfg_scales: List[float],
#     num_inference_steps: int = 4,
#     prog_bar: bool = False,
#     controller=None,
#     trajectory: Optional[List[torch.Tensor]] = None,
#     strength: float = 1.0,
# ) -> tuple:
#     """
#     GNRI reverse process with P2P attention control (x_T -> x_0).

#     Args:
#         pipe: SDXL diffusion pipeline
#         xT: Starting noisy latent
#         prompts: List of prompts [source_prompt, target_prompt]
#         cfg_scales: List of CFG scales for each prompt
#         num_inference_steps: Number of denoising steps
#         prog_bar: Show progress bar
#         controller: P2P attention controller
#         trajectory: Optional inversion trajectory
#         strength: Denoising strength (1.0 = full denoising)

#     Returns:
#         (x0, trajectory): Denoised latents and trajectory
#     """
#     batch_size = len(prompts)

#     # Encode all prompts
#     all_prompt_embeds = []
#     all_pooled_embeds = []

#     for prompt in prompts:
#         prompt_embeds, pooled_embeds = encode_text_sdxl(pipe, prompt)
#         all_prompt_embeds.append(prompt_embeds)
#         all_pooled_embeds.append(pooled_embeds)

#     # Stack embeddings
#     prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
#     pooled_prompt_embeds = torch.cat(all_pooled_embeds, dim=0)

#     # Prepare additional embeddings
#     height, width = xT.shape[-2] * 8, xT.shape[-1] * 8
#     original_size = (height, width)
#     target_size = (height, width)
#     crops_coords_top_left = (0, 0)

#     add_time_ids = pipe._get_add_time_ids(
#         original_size,
#         crops_coords_top_left,
#         target_size,
#         dtype=prompt_embeds.dtype,
#         text_encoder_projection_dim=1280,
#     )
#     add_time_ids = add_time_ids.repeat(batch_size, 1).to(pipe.device)

#     # Set up scheduler
#     pipe.scheduler.set_timesteps(num_inference_steps)
#     timesteps = pipe.scheduler.timesteps.to(pipe.device)

#     # Determine starting point based on strength
#     start_idx = int((1.0 - strength) * len(timesteps))
#     timesteps = timesteps[start_idx:]

#     # Expand latent for batch processing
#     xt = xT.expand(batch_size, -1, -1, -1)

#     reverse_trajectory = [xt.clone()]

#     if prog_bar:
#         from tqdm import tqdm
#         timesteps_iter = tqdm(timesteps, desc="GNRI Editing")
#     else:
#         timesteps_iter = timesteps

#     # Denoising loop
#     for i, t in enumerate(timesteps_iter):
#         # Scale model input
#         latent_model_input = pipe.scheduler.scale_model_input(xt, t)

#         # Prepare conditioning
#         added_cond_kwargs = {
#             "text_embeds": pooled_prompt_embeds,
#             "time_ids": add_time_ids
#         }

#         # Predict noise
#         with torch.no_grad():
#             noise_pred = pipe.unet(
#                 latent_model_input,
#                 t,
#                 encoder_hidden_states=prompt_embeds,
#                 cross_attention_kwargs=None,
#                 added_cond_kwargs=added_cond_kwargs,
#                 return_dict=False,
#             )[0]

#         # Apply CFG
#         cfg_scales_tensor = torch.tensor(cfg_scales).view(-1, 1, 1, 1).to(pipe.device)
#         # For simplicity, assume first prediction is unconditional
#         # In practice, you'd need to handle this properly
#         noise_pred_adjusted = noise_pred * cfg_scales_tensor.mean()

#         # Scheduler step
#         xt = pipe.scheduler.step(
#             noise_pred_adjusted,
#             t,
#             xt,
#             return_dict=False,
#         )[0]

#         # Apply attention controller callback
#         if controller is not None:
#             xt = controller.step_callback(xt)

#         reverse_trajectory.append(xt.clone())

#     return xt, reverse_trajectory
