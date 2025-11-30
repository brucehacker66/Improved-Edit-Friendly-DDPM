# """
# Integrated Fast Image Editing Pipeline: GNRI + Edit Friendly P2P

# This module combines:
# 1. GNRI (Guided Newton-Raphson Inversion) for fast inversion (~0.4s)
# 2. Edit Friendly DDPM's Prompt-to-Prompt attention control for structure preservation

# The result is a real-time image editing pipeline that maintains structural fidelity
# while being orders of magnitude faster than the original Edit Friendly method.
# """

# import torch
# import numpy as np
# from PIL import Image
# from typing import Optional, List, Union
# import sys
# import os

# from diffusers import AutoPipelineForImage2Image, StableDiffusionXLPipeline
# from models.gnri_inversion_utils import (
#     gnri_inversion_forward_process,
#     gnri_inversion_reverse_process,
# )

# # Import P2P controllers from Edit Friendly codebase
# try:
#     from models.edit_friendly_ddm.ptp_classes import (
#         AttentionReplace,
#         AttentionRefine,
#         AttentionStore,
#     )
#     from models.edit_friendly_ddm.ptp_utils import register_attention_control
# except ImportError:
#     # Fallback: copy the necessary classes if import fails
#     print("Warning: Could not import P2P classes from PnPInversion. Using fallback.")
#     from PnPInversion.models.edit_friendly_ddm.ptp_classes import (
#         AttentionReplace,
#         AttentionRefine,
#         AttentionStore,
#     )
#     from PnPInversion.models.edit_friendly_ddm.ptp_utils import register_attention_control

# # Import GNRI scheduler
# try:
#     from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
# except ImportError:
#     print("Warning: Could not import GNRI scheduler. Using fallback.")
#     from NewtonRaphsonInversion.src.euler_scheduler import MyEulerAncestralDiscreteScheduler


# class IntegratedFastEditor:
#     """
#     Integrated Fast Image Editor combining GNRI inversion with P2P attention control.

#     This class provides a unified interface for:
#     1. Fast image inversion using GNRI (Newton-Raphson)
#     2. Structure-preserving editing using Prompt-to-Prompt attention control
#     """

#     def __init__(
#         self,
#         model_id: str = "stabilityai/sdxl-turbo",
#         device: str = "cuda",
#         num_inference_steps: int = 4,
#         dtype: torch.dtype = torch.float16,
#         cache_dir: Optional[str] = None,
#     ):
#         """
#         Initialize the integrated fast editor.

#         Args:
#             model_id: HuggingFace model ID (default: SDXL-Turbo for speed)
#             device: Device to run on ('cuda' or 'cpu')
#             num_inference_steps: Number of diffusion steps (4 for turbo models)
#             dtype: Data type for model weights
#             cache_dir: Optional cache directory for model weights
#         """
#         self.device = device
#         self.num_inference_steps = num_inference_steps
#         self.dtype = dtype

#         print(f"Loading model: {model_id}")

#         # Load pipeline
#         self.pipe = StableDiffusionXLPipeline.from_pretrained(
#             model_id,
#             torch_dtype=dtype,
#             use_safetensors=True,
#             variant="fp16" if dtype == torch.float16 else None,
#             cache_dir=cache_dir,
#         ).to(device)

#         # Set up scheduler (GNRI-compatible)
#         self.pipe.scheduler = MyEulerAncestralDiscreteScheduler.from_config(
#             self.pipe.scheduler.config
#         )

#         # Initialize noise list for deterministic sampling
#         self._setup_noise()

#         print("Model loaded successfully!")

#     def _setup_noise(self, seed: int = 42):
#         """Set up deterministic noise for reproducible results."""
#         g_cpu = torch.Generator().manual_seed(seed)
#         img_size = (512, 512)
#         VQAE_SCALE = 8
#         latents_size = (
#             1,
#             4,
#             img_size[0] // VQAE_SCALE,
#             img_size[1] // VQAE_SCALE,
#         )

#         noise_list = [
#             torch.randn(
#                 latents_size,
#                 dtype=self.dtype,
#                 device=self.device,
#                 generator=g_cpu,
#             )
#             for _ in range(self.num_inference_steps)
#         ]

#         self.pipe.scheduler.set_noise_list(noise_list)

#     def load_and_encode_image(
#         self, image_path: Union[str, Image.Image], size: int = 512
#     ) -> torch.Tensor:
#         """
#         Load and encode an image to latent space.

#         Args:
#             image_path: Path to image or PIL Image
#             size: Target size (default: 512)

#         Returns:
#             Encoded latent tensor
#         """
#         # Load image
#         if isinstance(image_path, str):
#             image = Image.open(image_path).convert("RGB")
#         else:
#             image = image_path

#         # Resize and center crop
#         image = self._center_crop_resize(image, size)

#         # Convert to tensor
#         image_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
#         image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
#         image_tensor = image_tensor.to(self.dtype)

#         # Encode to latent
#         with torch.no_grad():
#             latent = (
#                 self.pipe.vae.encode(image_tensor).latent_dist.mode() * 0.18215
#             )

#         return latent

#     def _center_crop_resize(self, image: Image.Image, size: int) -> Image.Image:
#         """Center crop and resize image to square."""
#         width, height = image.size
#         min_dim = min(width, height)

#         left = (width - min_dim) // 2
#         top = (height - min_dim) // 2
#         right = left + min_dim
#         bottom = top + min_dim

#         image = image.crop((left, top, right, bottom))
#         image = image.resize((size, size), Image.Resampling.LANCZOS)

#         return image

#     def decode_latent(self, latent: torch.Tensor) -> Image.Image:
#         """Decode latent to image."""
#         with torch.no_grad():
#             image_tensor = self.pipe.vae.decode(latent / 0.18215).sample

#         image = (image_tensor / 2 + 0.5).clamp(0, 1)
#         image = image.cpu().permute(0, 2, 3, 1).numpy()
#         image = (image[0] * 255).astype(np.uint8)

#         return Image.fromarray(image)

#     def invert_image(
#         self,
#         image_latent: torch.Tensor,
#         source_prompt: str,
#         guidance_scale: float = 1.0,
#         n_iters: int = 2,
#         alpha: float = 0.1,
#         show_progress: bool = True,
#     ) -> tuple:
#         """
#         Invert image using GNRI.

#         Args:
#             image_latent: Encoded image latent
#             source_prompt: Text description of the image
#             guidance_scale: CFG scale for inversion (typically 1.0)
#             n_iters: Newton-Raphson iterations per step
#             alpha: Gaussian prior weight
#             show_progress: Show progress bar

#         Returns:
#             (inverted_latent, trajectory): Inverted latent and inversion trajectory
#         """
#         print("Starting GNRI inversion...")

#         inverted_latent, trajectory = gnri_inversion_forward_process(
#             pipe=self.pipe,
#             x0=image_latent,
#             prompt=source_prompt,
#             num_inference_steps=self.num_inference_steps,
#             guidance_scale=guidance_scale,
#             n_iters=n_iters,
#             alpha=alpha,
#             prog_bar=show_progress,
#         )

#         print("Inversion complete!")
#         return inverted_latent, trajectory

#     def edit_image(
#         self,
#         inverted_latent: torch.Tensor,
#         source_prompt: str,
#         target_prompt: str,
#         source_guidance_scale: float = 1.0,
#         target_guidance_scale: float = 1.2,
#         cross_replace_steps: float = 0.4,
#         self_replace_steps: float = 0.6,
#         show_progress: bool = True,
#         trajectory: Optional[List[torch.Tensor]] = None,
#     ) -> Image.Image:
#         """
#         Edit image using P2P attention control.

#         Args:
#             inverted_latent: Inverted latent from inversion step
#             source_prompt: Original image description
#             target_prompt: Target image description
#             source_guidance_scale: CFG scale for source
#             target_guidance_scale: CFG scale for target
#             cross_replace_steps: Cross-attention replacement ratio
#             self_replace_steps: Self-attention replacement ratio
#             show_progress: Show progress bar
#             trajectory: Optional inversion trajectory

#         Returns:
#             Edited image
#         """
#         print("Starting P2P-guided editing...")

#         # Register attention control
#         prompts = [source_prompt, target_prompt]
#         cfg_scale_list = [source_guidance_scale, target_guidance_scale]

#         # Choose controller based on prompt length
#         if len(source_prompt.split(" ")) == len(target_prompt.split(" ")):
#             print("Using AttentionReplace (same token count)")
#             controller = AttentionReplace(
#                 prompts,
#                 self.num_inference_steps,
#                 cross_replace_steps=cross_replace_steps,
#                 self_replace_steps=self_replace_steps,
#                 model=self.pipe,
#             )
#         else:
#             print("Using AttentionRefine (different token count)")
#             controller = AttentionRefine(
#                 prompts,
#                 self.num_inference_steps,
#                 cross_replace_steps=cross_replace_steps,
#                 self_replace_steps=self_replace_steps,
#                 model=self.pipe,
#             )

#         # Register controller with model
#         register_attention_control(self.pipe, controller)

#         # Reverse process with P2P control
#         edited_latents, reverse_traj = gnri_inversion_reverse_process(
#             pipe=self.pipe,
#             xT=inverted_latent,
#             prompts=prompts,
#             cfg_scales=cfg_scale_list,
#             num_inference_steps=self.num_inference_steps,
#             prog_bar=show_progress,
#             controller=controller,
#             trajectory=trajectory,
#         )

#         # Decode edited latent (take the target/edited version)
#         edited_image = self.decode_latent(edited_latents[1:2])

#         print("Editing complete!")
#         return edited_image

#     def run_full_pipeline(
#         self,
#         image_path: Union[str, Image.Image],
#         source_prompt: str,
#         target_prompt: str,
#         source_guidance_scale: float = 1.0,
#         target_guidance_scale: float = 1.2,
#         cross_replace_steps: float = 0.4,
#         self_replace_steps: float = 0.6,
#         n_iters: int = 2,
#         alpha: float = 0.1,
#         show_progress: bool = True,
#     ) -> dict:
#         """
#         Run full integrated pipeline: Load -> Invert -> Edit.

#         Args:
#             image_path: Input image path or PIL Image
#             source_prompt: Description of input image
#             target_prompt: Description of desired output
#             source_guidance_scale: CFG for source
#             target_guidance_scale: CFG for target
#             cross_replace_steps: Cross-attention replacement ratio
#             self_replace_steps: Self-attention replacement ratio
#             n_iters: Newton-Raphson iterations
#             alpha: Gaussian prior weight
#             show_progress: Show progress bars

#         Returns:
#             Dictionary with 'original', 'reconstructed', and 'edited' images
#         """
#         import time

#         # Load and encode image
#         print("Loading image...")
#         start_time = time.time()
#         image_latent = self.load_and_encode_image(image_path)
#         load_time = time.time() - start_time
#         print(f"Image loaded in {load_time:.2f}s")

#         # Decode original for comparison
#         original_image = self.decode_latent(image_latent)

#         # Invert image
#         start_time = time.time()
#         inverted_latent, trajectory = self.invert_image(
#             image_latent=image_latent,
#             source_prompt=source_prompt,
#             guidance_scale=source_guidance_scale,
#             n_iters=n_iters,
#             alpha=alpha,
#             show_progress=show_progress,
#         )
#         inversion_time = time.time() - start_time
#         print(f"Inversion took {inversion_time:.2f}s")

#         # Test reconstruction
#         reconstructed_image = self.decode_latent(trajectory[-1] if trajectory else inverted_latent)

#         # Edit image
#         start_time = time.time()
#         edited_image = self.edit_image(
#             inverted_latent=inverted_latent,
#             source_prompt=source_prompt,
#             target_prompt=target_prompt,
#             source_guidance_scale=source_guidance_scale,
#             target_guidance_scale=target_guidance_scale,
#             cross_replace_steps=cross_replace_steps,
#             self_replace_steps=self_replace_steps,
#             show_progress=show_progress,
#             trajectory=trajectory,
#         )
#         editing_time = time.time() - start_time
#         print(f"Editing took {editing_time:.2f}s")

#         total_time = load_time + inversion_time + editing_time
#         print(f"\nTotal pipeline time: {total_time:.2f}s")

#         return {
#             "original": original_image,
#             "reconstructed": reconstructed_image,
#             "edited": edited_image,
#             "timings": {
#                 "load": load_time,
#                 "inversion": inversion_time,
#                 "editing": editing_time,
#                 "total": total_time,
#             },
#         }


# def setup_seed(seed: int = 42):
#     """Set random seeds for reproducibility."""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     import random
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# if __name__ == "__main__":
#     # Quick test
#     print("Integrated Fast Editor - GNRI + Edit Friendly P2P")
#     print("=" * 60)

#     setup_seed(42)

#     # Initialize editor
#     editor = IntegratedFastEditor(
#         model_id="stabilityai/sdxl-turbo",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         num_inference_steps=4,
#     )

#     print("\nEditor initialized successfully!")
#     print("Use editor.run_full_pipeline() to process images.")
