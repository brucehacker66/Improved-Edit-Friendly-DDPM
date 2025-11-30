# """
# Demo Script for Integrated Fast Image Editing Pipeline

# This script demonstrates the integrated GNRI + Edit Friendly P2P pipeline
# with several example editing tasks.
# """

# import sys
# import os
# from pathlib import Path
# import torch

# # Add project paths
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))
# sys.path.append(str(project_root / "PnPInversion"))
# sys.path.append(str(project_root / "NewtonRaphsonInversion"))

# from integrated_editing_pipeline import IntegratedFastEditor, setup_seed
# import matplotlib.pyplot as plt
# import numpy as np


# def display_results(results, title="Image Editing Results"):
#     """Display original, reconstructed, and edited images."""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     axes[0].imshow(results["original"])
#     axes[0].set_title("Original")
#     axes[0].axis("off")

#     axes[1].imshow(results["reconstructed"])
#     axes[1].set_title("Reconstructed")
#     axes[1].axis("off")

#     axes[2].imshow(results["edited"])
#     axes[2].set_title("Edited")
#     axes[2].axis("off")

#     fig.suptitle(title, fontsize=16, fontweight="bold")
#     plt.tight_layout()
#     plt.show()


# def demo_basic_editing():
#     """Basic demo: Edit a single image."""
#     print("\n" + "=" * 70)
#     print("DEMO 1: Basic Image Editing")
#     print("=" * 70)

#     # Set random seed
#     setup_seed(42)

#     # Initialize editor
#     print("\nInitializing editor with SDXL-Turbo...")
#     editor = IntegratedFastEditor(
#         model_id="stabilityai/sdxl-turbo",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         num_inference_steps=4,
#     )

#     # Example 1: Animal transformation
#     print("\nExample: Cat to Dog transformation")
#     print("-" * 70)

#     # You can replace this with an actual image path
#     # For demo purposes, we'll create a placeholder
#     try:
#         # Try to use an image from NewtonRaphsonInversion examples
#         image_path = project_root / "NewtonRaphsonInversion" / "example_images" / "lion.jpeg"

#         if not image_path.exists():
#             print(f"Note: Example image not found at {image_path}")
#             print("Please provide your own image path.")
#             return

#         results = editor.run_full_pipeline(
#             image_path=str(image_path),
#             source_prompt="a lion is sitting in the grass at sunset",
#             target_prompt="a tiger is sitting in the grass at sunset",
#             source_guidance_scale=1.0,
#             target_guidance_scale=1.2,
#             cross_replace_steps=0.4,
#             self_replace_steps=0.6,
#             n_iters=2,
#             alpha=0.1,
#             show_progress=True,
#         )

#         print("\n" + "-" * 70)
#         print("Results:")
#         print(f"  Inversion time: {results['timings']['inversion']:.2f}s")
#         print(f"  Editing time: {results['timings']['editing']:.2f}s")
#         print(f"  Total time: {results['timings']['total']:.2f}s")

#         # Display results
#         display_results(results, "Lion → Tiger Transformation")

#         # Save results
#         output_dir = project_root / "IntegratedPipeline" / "demo_output"
#         output_dir.mkdir(exist_ok=True)

#         results["edited"].save(output_dir / "demo_lion_to_tiger.png")
#         print(f"\nSaved edited image to: {output_dir / 'demo_lion_to_tiger.png'}")

#     except Exception as e:
#         print(f"Error during demo: {e}")
#         import traceback
#         traceback.print_exc()


# def demo_multiple_edits():
#     """Demo: Multiple editing operations on the same image."""
#     print("\n" + "=" * 70)
#     print("DEMO 2: Multiple Edits on Same Image")
#     print("=" * 70)

#     setup_seed(42)

#     editor = IntegratedFastEditor(
#         model_id="stabilityai/sdxl-turbo",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         num_inference_steps=4,
#     )

#     # Try to find an example image
#     image_path = project_root / "NewtonRaphsonInversion" / "example_images" / "lion.jpeg"

#     if not image_path.exists():
#         print("Example image not found. Skipping demo.")
#         return

#     # Load and encode once
#     print("\nLoading and encoding image...")
#     image_latent = editor.load_and_encode_image(str(image_path))
#     original_image = editor.decode_latent(image_latent)

#     # Source prompt
#     source_prompt = "a lion is sitting in the grass at sunset"

#     # Multiple target prompts
#     target_prompts = [
#         "a tiger is sitting in the grass at sunset",
#         "a leopard is sitting in the grass at sunset",
#         "a cheetah is sitting in the grass at sunset",
#     ]

#     # Invert once
#     print("\nInverting image (once)...")
#     inverted_latent, trajectory = editor.invert_image(
#         image_latent=image_latent,
#         source_prompt=source_prompt,
#         guidance_scale=1.0,
#         n_iters=2,
#         alpha=0.1,
#         show_progress=True,
#     )

#     # Edit multiple times
#     edited_images = []
#     for target_prompt in target_prompts:
#         print(f"\nEditing: {source_prompt} → {target_prompt}")
#         edited_image = editor.edit_image(
#             inverted_latent=inverted_latent,
#             source_prompt=source_prompt,
#             target_prompt=target_prompt,
#             source_guidance_scale=1.0,
#             target_guidance_scale=1.2,
#             cross_replace_steps=0.4,
#             self_replace_steps=0.6,
#             show_progress=False,
#             trajectory=trajectory,
#         )
#         edited_images.append(edited_image)

#     # Display all results
#     fig, axes = plt.subplots(1, len(edited_images) + 1, figsize=(20, 5))

#     axes[0].imshow(original_image)
#     axes[0].set_title("Original\n(Lion)")
#     axes[0].axis("off")

#     animals = ["Tiger", "Leopard", "Cheetah"]
#     for i, (img, animal) in enumerate(zip(edited_images, animals)):
#         axes[i + 1].imshow(img)
#         axes[i + 1].set_title(f"Edited\n({animal})")
#         axes[i + 1].axis("off")

#     fig.suptitle("Multiple Edits from Single Inversion", fontsize=16, fontweight="bold")
#     plt.tight_layout()
#     plt.show()

#     print("\nDemo complete!")


# def demo_parameter_comparison():
#     """Demo: Compare different parameter settings."""
#     print("\n" + "=" * 70)
#     print("DEMO 3: Parameter Comparison")
#     print("=" * 70)

#     setup_seed(42)

#     editor = IntegratedFastEditor(
#         model_id="stabilityai/sdxl-turbo",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         num_inference_steps=4,
#     )

#     image_path = project_root / "NewtonRaphsonInversion" / "example_images" / "lion.jpeg"

#     if not image_path.exists():
#         print("Example image not found. Skipping demo.")
#         return

#     source_prompt = "a lion is sitting in the grass at sunset"
#     target_prompt = "a tiger is sitting in the grass at sunset"

#     # Different cross_replace_steps values
#     cross_replace_values = [0.2, 0.4, 0.6, 0.8]

#     print(f"\nComparing cross_replace_steps: {cross_replace_values}")

#     results_list = []
#     for cross_replace in cross_replace_values:
#         print(f"\nTesting cross_replace_steps={cross_replace}")
#         results = editor.run_full_pipeline(
#             image_path=str(image_path),
#             source_prompt=source_prompt,
#             target_prompt=target_prompt,
#             cross_replace_steps=cross_replace,
#             self_replace_steps=0.6,
#             show_progress=False,
#         )
#         results_list.append(results["edited"])

#     # Display comparison
#     fig, axes = plt.subplots(1, len(cross_replace_values), figsize=(20, 5))

#     for i, (img, cross_val) in enumerate(zip(results_list, cross_replace_values)):
#         axes[i].imshow(img)
#         axes[i].set_title(f"cross_replace={cross_val}")
#         axes[i].axis("off")

#     fig.suptitle("Effect of cross_replace_steps Parameter", fontsize=16, fontweight="bold")
#     plt.tight_layout()
#     plt.show()

#     print("\nDemo complete!")


# def main():
#     """Run all demos."""
#     print("\n" + "=" * 70)
#     print("Integrated Fast Image Editing Pipeline - Demo")
#     print("GNRI + Edit Friendly P2P")
#     print("=" * 70)

#     demos = [
#         ("Basic Editing", demo_basic_editing),
#         ("Multiple Edits", demo_multiple_edits),
#         ("Parameter Comparison", demo_parameter_comparison),
#     ]

#     print("\nAvailable demos:")
#     for i, (name, _) in enumerate(demos, 1):
#         print(f"  {i}. {name}")
#     print("  0. Run all demos")

#     try:
#         choice = input("\nSelect demo (0-3): ").strip()
#         choice = int(choice)

#         if choice == 0:
#             # Run all demos
#             for name, demo_func in demos:
#                 try:
#                     demo_func()
#                 except Exception as e:
#                     print(f"\nError in {name} demo: {e}")
#                     import traceback
#                     traceback.print_exc()
#         elif 1 <= choice <= len(demos):
#             # Run selected demo
#             name, demo_func = demos[choice - 1]
#             demo_func()
#         else:
#             print("Invalid choice.")

#     except ValueError:
#         print("Invalid input. Please enter a number.")
#     except KeyboardInterrupt:
#         print("\n\nDemo interrupted by user.")


# if __name__ == "__main__":
#     main()
