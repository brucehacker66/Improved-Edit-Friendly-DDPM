# """
# Main Script for Integrated Fast Image Editing Pipeline

# Usage:
#     python main.py --image_path <path> --source_prompt <prompt> --target_prompt <prompt> [options]

# Example:
#     python main.py \
#         --image_path data/annotation_images/sample.png \
#         --source_prompt "a photo of a cat" \
#         --target_prompt "a photo of a dog" \
#         --output_path output/integrated
# """

# import argparse
# import os
# import sys
# import torch
# from PIL import Image
# import numpy as np
# from pathlib import Path
# import time
# import json

# # Add project paths
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))
# sys.path.append(str(project_root / "PnPInversion"))
# sys.path.append(str(project_root / "NewtonRaphsonInversion"))

# from integrated_editing_pipeline import IntegratedFastEditor, setup_seed


# def create_comparison_image(original, reconstructed, edited, prompts):
#     """Create a side-by-side comparison image."""
#     # Convert PIL images to numpy arrays
#     orig_arr = np.array(original)
#     recon_arr = np.array(reconstructed)
#     edit_arr = np.array(edited)

#     # Add text labels
#     from PIL import ImageDraw, ImageFont

#     def add_label(img_arr, text, color=(255, 255, 255)):
#         img = Image.fromarray(img_arr)
#         draw = ImageDraw.Draw(img)

#         # Try to use a nice font, fall back to default if not available
#         try:
#             font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
#         except:
#             font = ImageFont.load_default()

#         # Add text with background
#         bbox = draw.textbbox((10, 10), text, font=font)
#         draw.rectangle(bbox, fill=(0, 0, 0, 180))
#         draw.text((10, 10), text, fill=color, font=font)

#         return np.array(img)

#     orig_labeled = add_label(orig_arr.copy(), "Original")
#     recon_labeled = add_label(recon_arr.copy(), "Reconstructed")
#     edit_labeled = add_label(edit_arr.copy(), "Edited")

#     # Create text banner with prompts
#     height = orig_arr.shape[0]
#     width = orig_arr.shape[1] * 3
#     banner_height = 100

#     banner = np.ones((banner_height, width, 3), dtype=np.uint8) * 240

#     banner_img = Image.fromarray(banner)
#     draw = ImageDraw.Draw(banner_img)

#     try:
#         font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
#     except:
#         font_small = ImageFont.load_default()

#     # Add prompt text
#     source_text = f"Source: {prompts['source']}"
#     target_text = f"Target: {prompts['target']}"

#     draw.text((10, 10), source_text, fill=(0, 0, 0), font=font_small)
#     draw.text((10, 40), target_text, fill=(0, 100, 0), font=font_small)

#     banner_arr = np.array(banner_img)

#     # Concatenate horizontally
#     comparison = np.concatenate([orig_labeled, recon_labeled, edit_labeled], axis=1)

#     # Add banner on top
#     final = np.concatenate([banner_arr, comparison], axis=0)

#     return Image.fromarray(final)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Integrated Fast Image Editing: GNRI + Edit Friendly P2P"
#     )

#     # Input/Output
#     parser.add_argument(
#         "--image_path",
#         type=str,
#         required=True,
#         help="Path to input image",
#     )
#     parser.add_argument(
#         "--source_prompt",
#         type=str,
#         required=True,
#         help="Description of the input image",
#     )
#     parser.add_argument(
#         "--target_prompt",
#         type=str,
#         required=True,
#         help="Description of the desired output",
#     )
#     parser.add_argument(
#         "--output_path",
#         type=str,
#         default="output/integrated",
#         help="Directory to save outputs",
#     )
#     parser.add_argument(
#         "--output_name",
#         type=str,
#         default=None,
#         help="Output filename (default: auto-generated)",
#     )

#     # Model settings
#     parser.add_argument(
#         "--model_id",
#         type=str,
#         default="stabilityai/sdxl-turbo",
#         help="HuggingFace model ID",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#         help="Device to run on (cuda/cpu)",
#     )
#     parser.add_argument(
#         "--num_steps",
#         type=int,
#         default=4,
#         help="Number of diffusion steps",
#     )
#     parser.add_argument(
#         "--cache_dir",
#         type=str,
#         default=None,
#         help="Cache directory for model weights",
#     )

#     # Editing parameters
#     parser.add_argument(
#         "--source_guidance_scale",
#         type=float,
#         default=1.0,
#         help="CFG scale for source prompt during inversion",
#     )
#     parser.add_argument(
#         "--target_guidance_scale",
#         type=float,
#         default=1.2,
#         help="CFG scale for target prompt during editing",
#     )
#     parser.add_argument(
#         "--cross_replace_steps",
#         type=float,
#         default=0.4,
#         help="Cross-attention replacement ratio (0-1)",
#     )
#     parser.add_argument(
#         "--self_replace_steps",
#         type=float,
#         default=0.6,
#         help="Self-attention replacement ratio (0-1)",
#     )

#     # GNRI parameters
#     parser.add_argument(
#         "--n_iters",
#         type=int,
#         default=2,
#         help="Newton-Raphson iterations per timestep",
#     )
#     parser.add_argument(
#         "--alpha",
#         type=float,
#         default=0.1,
#         help="Gaussian prior weight for GNRI",
#     )

#     # Other
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for reproducibility",
#     )
#     parser.add_argument(
#         "--no_progress",
#         action="store_true",
#         help="Disable progress bars",
#     )
#     parser.add_argument(
#         "--save_individual",
#         action="store_true",
#         help="Save individual images in addition to comparison",
#     )

#     args = parser.parse_args()

#     # Set random seed
#     setup_seed(args.seed)

#     # Create output directory
#     os.makedirs(args.output_path, exist_ok=True)

#     # Check if image exists
#     if not os.path.exists(args.image_path):
#         print(f"Error: Image not found at {args.image_path}")
#         sys.exit(1)

#     print("=" * 70)
#     print("Integrated Fast Image Editing Pipeline")
#     print("GNRI (Newton-Raphson Inversion) + Edit Friendly P2P")
#     print("=" * 70)
#     print(f"\nInput image: {args.image_path}")
#     print(f"Source prompt: {args.source_prompt}")
#     print(f"Target prompt: {args.target_prompt}")
#     print(f"Output path: {args.output_path}")
#     print(f"Device: {args.device}")
#     print(f"Model: {args.model_id}")
#     print(f"Steps: {args.num_steps}")
#     print()

#     # Initialize editor
#     print("Initializing editor...")
#     editor = IntegratedFastEditor(
#         model_id=args.model_id,
#         device=args.device,
#         num_inference_steps=args.num_steps,
#         dtype=torch.float16 if args.device == "cuda" else torch.float32,
#         cache_dir=args.cache_dir,
#     )

#     # Run pipeline
#     print("\nRunning integrated pipeline...\n")
#     start_time = time.time()

#     results = editor.run_full_pipeline(
#         image_path=args.image_path,
#         source_prompt=args.source_prompt,
#         target_prompt=args.target_prompt,
#         source_guidance_scale=args.source_guidance_scale,
#         target_guidance_scale=args.target_guidance_scale,
#         cross_replace_steps=args.cross_replace_steps,
#         self_replace_steps=args.self_replace_steps,
#         n_iters=args.n_iters,
#         alpha=args.alpha,
#         show_progress=not args.no_progress,
#     )

#     total_time = time.time() - start_time

#     print("\n" + "=" * 70)
#     print("Pipeline Complete!")
#     print("=" * 70)
#     print(f"Total time: {total_time:.2f}s")
#     print(f"  - Loading: {results['timings']['load']:.2f}s")
#     print(f"  - Inversion: {results['timings']['inversion']:.2f}s")
#     print(f"  - Editing: {results['timings']['editing']:.2f}s")
#     print()

#     # Generate output filename
#     if args.output_name:
#         base_name = args.output_name
#     else:
#         # Create name from prompts
#         source_short = args.source_prompt.replace(" ", "_")[:30]
#         target_short = args.target_prompt.replace(" ", "_")[:30]
#         base_name = f"{source_short}_to_{target_short}"

#     # Create comparison image
#     comparison_image = create_comparison_image(
#         original=results["original"],
#         reconstructed=results["reconstructed"],
#         edited=results["edited"],
#         prompts={
#             "source": args.source_prompt,
#             "target": args.target_prompt,
#         },
#     )

#     # Save comparison
#     comparison_path = os.path.join(args.output_path, f"{base_name}_comparison.png")
#     comparison_image.save(comparison_path)
#     print(f"Saved comparison: {comparison_path}")

#     # Save individual images if requested
#     if args.save_individual:
#         original_path = os.path.join(args.output_path, f"{base_name}_original.png")
#         reconstructed_path = os.path.join(args.output_path, f"{base_name}_reconstructed.png")
#         edited_path = os.path.join(args.output_path, f"{base_name}_edited.png")

#         results["original"].save(original_path)
#         results["reconstructed"].save(reconstructed_path)
#         results["edited"].save(edited_path)

#         print(f"Saved original: {original_path}")
#         print(f"Saved reconstructed: {reconstructed_path}")
#         print(f"Saved edited: {edited_path}")

#     # Save metadata
#     metadata = {
#         "image_path": args.image_path,
#         "source_prompt": args.source_prompt,
#         "target_prompt": args.target_prompt,
#         "model_id": args.model_id,
#         "num_steps": args.num_steps,
#         "parameters": {
#             "source_guidance_scale": args.source_guidance_scale,
#             "target_guidance_scale": args.target_guidance_scale,
#             "cross_replace_steps": args.cross_replace_steps,
#             "self_replace_steps": args.self_replace_steps,
#             "n_iters": args.n_iters,
#             "alpha": args.alpha,
#         },
#         "timings": results["timings"],
#         "seed": args.seed,
#     }

#     metadata_path = os.path.join(args.output_path, f"{base_name}_metadata.json")
#     with open(metadata_path, "w") as f:
#         json.dump(metadata, f, indent=2)

#     print(f"Saved metadata: {metadata_path}")
#     print("\nDone!")


# if __name__ == "__main__":
#     main()
