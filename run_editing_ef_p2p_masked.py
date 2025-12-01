"""
Edit-Friendly DDPM + P2P with Localized Attention Blending

Improvement implemented:
- Uses cross-attention maps to create a binary mask identifying the subject
- Composites reconstruction (background) with edited (foreground) latents at each step
- Maximizes preservation metrics (background identical) while concentrating editing on subject

Strategy: Extract attention maps for the subject words, create a mask, and blend
background (preserved) with foreground (edited) regions separately.
"""

import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
import time
from torch import autocast, inference_mode
from pathlib import Path

from pipeline_utils.utils import load_512, txt_draw, get_image_files
from models.edit_friendly_ddm.inversion_utils import inversion_forward_process, inversion_reverse_process
from models.edit_friendly_ddm.ptp_classes import AttentionStore
from models.edit_friendly_ddm.ptp_classes_masked import AttentionReplaceWithMask, AttentionRefineWithMask
from models.edit_friendly_ddm.ptp_utils import register_attention_control


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_subject_words(prompt_src, prompt_tar):
    """
    Extract words that differ between source and target prompts
    These are likely the subject words to focus editing on
    """
    src_words = set(prompt_src.lower().split())
    tar_words = set(prompt_tar.lower().split())

    # Words in source but not target (being replaced)
    changing_words = src_words - tar_words

    # If no clear difference, try to find nouns (simple heuristic)
    if not changing_words:
        # Use last 2 words as subject (common pattern: "a photo of [subject]")
        words = prompt_src.split()
        changing_words = set(words[-2:]) if len(words) >= 2 else set(words)

    return list(changing_words)


image_save_paths = {
    "masked-baseline": "masked-baseline",
    "masked-auto": "masked-auto",
    "masked-combined": "masked-combined",
}


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_DDIM_STEPS = 50
model_id = "CompVis/stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
ETA = 1
SKIP = 12


def edit_image_masked(
    edit_method,
    image_path,
    prompt_src,
    prompt_tar,
    blended_words=None,
    source_guidance_scale=1,
    target_guidance_scale=7.5,
    cross_replace_steps=0.4,
    self_replace_steps=0.6
):
    """
    Edit image with localized attention-based masking

    Args:
        edit_method: Which masking configuration to use
            - 'masked-baseline': Standard P2P without masking
            - 'masked-auto': Auto-detect subject words and apply masking
            - 'masked-combined': Masking + other optimizations
        image_path: Path to input image
        prompt_src: Source prompt
        prompt_tar: Target prompt
        blended_words: Specific words to focus on (optional)
        source_guidance_scale: CFG for inversion
        target_guidance_scale: CFG for editing
        cross_replace_steps: Cross-attention injection fraction
        self_replace_steps: Self-attention injection fraction
    """
    # Load and encode image
    image_gt = load_512(image_path)
    image_gt = torch.from_numpy(image_gt).float() / 127.5 - 1
    image_gt = image_gt.permute(2, 0, 1).unsqueeze(0).to(device)

    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(image_gt).latent_dist.mode() * 0.18215).float()

    # Inversion with source prompt
    controller = AttentionStore()
    register_attention_control(ldm_stable, controller)

    wt, zs, wts = inversion_forward_process(
        ldm_stable, w0,
        etas=ETA,
        prompt=prompt_src,
        cfg_scale=source_guidance_scale,
        prog_bar=True,
        num_inference_steps=NUM_DDIM_STEPS
    )

    # Get reconstruction for background reference
    controller_recon = AttentionStore()
    register_attention_control(ldm_stable, controller_recon)

    x0_reconstruct, _ = inversion_reverse_process(
        ldm_stable,
        xT=wts[NUM_DDIM_STEPS - SKIP],
        etas=ETA,
        prompts=[prompt_src],
        cfg_scales=[source_guidance_scale],
        prog_bar=True,
        zs=zs[:(NUM_DDIM_STEPS - SKIP)],
        controller=controller_recon
    )

    # Configure masking based on method
    if edit_method == "masked-baseline":
        use_masking = False
        target_words = []
        cross_steps = 0.4
        self_steps = 0.6
        mask_threshold = 0.3
    elif edit_method == "masked-auto":
        use_masking = True
        # Auto-detect subject words
        if blended_words and len(blended_words) > 0:
            target_words = blended_words
        else:
            target_words = extract_subject_words(prompt_src, prompt_tar)
        cross_steps = 0.4
        self_steps = 0.6
        mask_threshold = 0.3
    elif edit_method == "masked-combined":
        use_masking = True
        if blended_words and len(blended_words) > 0:
            target_words = blended_words
        else:
            target_words = extract_subject_words(prompt_src, prompt_tar)
        cross_steps = 0.3  # More relaxed
        self_steps = 0.7  # Stronger structure preservation
        mask_threshold = 0.25  # Slightly lower threshold for broader mask
    else:
        raise NotImplementedError(f"No edit method named {edit_method}")

    # Editing with P2P and optional masking
    cfg_scale_list = [source_guidance_scale, target_guidance_scale]
    prompts = [prompt_src, prompt_tar]

    if len(prompt_src.split(" ")) == len(prompt_tar.split(" ")):
        if use_masking:
            controller = AttentionReplaceWithMask(
                prompts, NUM_DDIM_STEPS,
                cross_replace_steps=cross_steps,
                self_replace_steps=self_steps,
                model=ldm_stable,
                target_words=target_words,
                mask_threshold=mask_threshold
            )
        else:
            from models.edit_friendly_ddm.ptp_classes import AttentionReplace
            controller = AttentionReplace(
                prompts, NUM_DDIM_STEPS,
                cross_replace_steps=cross_steps,
                self_replace_steps=self_steps,
                model=ldm_stable
            )
    else:
        if use_masking:
            controller = AttentionRefineWithMask(
                prompts, NUM_DDIM_STEPS,
                cross_replace_steps=cross_steps,
                self_replace_steps=self_steps,
                model=ldm_stable,
                target_words=target_words,
                mask_threshold=mask_threshold
            )
        else:
            from models.edit_friendly_ddm.ptp_classes import AttentionRefine
            controller = AttentionRefine(
                prompts, NUM_DDIM_STEPS,
                cross_replace_steps=cross_steps,
                self_replace_steps=self_steps,
                model=ldm_stable
            )

    # Extract mask from attention if using masking
    if use_masking:
        # Use reconstruction controller's attention maps to extract mask
        mask = controller.attention_mask_extractor.extract_mask_from_attention(
            controller_recon, res=16
        )
        controller.set_mask(mask)
        print(f"  Using mask for words: {target_words}, threshold: {mask_threshold}")

    register_attention_control(ldm_stable, controller)
    w0, _ = inversion_reverse_process(
        ldm_stable,
        xT=wts[NUM_DDIM_STEPS - SKIP],
        etas=ETA,
        prompts=prompts,
        cfg_scales=cfg_scale_list,
        prog_bar=True,
        zs=zs[:(NUM_DDIM_STEPS - SKIP)],
        controller=controller
    )

    # Decode latents to images
    with autocast("cuda"), inference_mode():
        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0[1].unsqueeze(0)).sample
        x0_reconstruct_edit = ldm_stable.vae.decode(1 / 0.18215 * w0[0].unsqueeze(0)).sample
        x0_reconstruct_viz = ldm_stable.vae.decode(1 / 0.18215 * x0_reconstruct[0].unsqueeze(0)).sample

    image_instruct = txt_draw(
        f"Method: {edit_method}\\n"
        f"Masking: {'Yes' if use_masking else 'No'}, Words: {target_words if use_masking else 'N/A'}\\n"
        f"source: {prompt_src}\\ntarget: {prompt_tar}"
    )

    # Extract individual edited image
    edited_image = np.uint8((np.array(x0_dec[0].permute(1, 2, 0).cpu().detach()) / 2 + 0.5) * 255)
    edited_image_pil = Image.fromarray(edited_image)

    # Create concatenated visualization
    concatenated_image = Image.fromarray(
        np.concatenate(
            (
                image_instruct,
                np.uint8((np.array(image_gt[0].permute(1, 2, 0).cpu().detach()) / 2 + 0.5) * 255),
                np.uint8((np.array(x0_reconstruct_edit[0].permute(1, 2, 0).cpu().detach()) / 2 + 0.5) * 255),
                edited_image
            ),
            1
        )
    )

    return concatenated_image, edited_image_pil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--map_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--edit_category_list', nargs='+', type=str, default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    parser.add_argument('--edit_method_list', nargs='+', type=str, default=["masked-auto"])
    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    map_path = args.map_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list

    # Load mapping file
    print(f"\nLoading mapping file from: {map_path}")
    with open(map_path, 'r') as f:
        mapping_data = json.load(f)
    print(f"Loaded {len(mapping_data)} entries from mapping file")

    image_files = get_image_files(data_path)
    print(f"Found {len(image_files)} image files")

    if len(image_files) == 0:
        print("No images found in the directory!")
        exit(1)

    data_dir_path = Path(data_path)
    # Try to find 'annotation_images' in the path to preserve structure from there
    parts = data_dir_path.parts
    if 'annotation_images' in parts:
        idx = parts.index('annotation_images')
        relative_structure = Path(*parts[idx:])
    else:
        # If no 'annotation_images', just use the last part of the path
        relative_structure = Path(data_dir_path.name)

    # Process each image
    total_time = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print("\n" + "=" * 80)
    print("Processing Images")
    print("=" * 80)

    for i, image_file in enumerate(image_files, 1):
        # Extract image ID (filename without extension)
        image_id = Path(image_file).stem
        image_path = os.path.join(data_path, image_file)

        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")

        # Look up prompts in mapping file
        if image_id not in mapping_data:
            print(f"  WARNING: Image ID '{image_id}' not found in mapping file. Skipping.")
            skipped_count += 1
            continue

        entry = mapping_data[image_id]
        original_prompt = entry["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = entry["editing_prompt"].replace("[", "").replace("]", "")
        blended_word = entry["blended_word"].split(" ") if entry["blended_word"] != "" else []

        print(f"  Original prompt: {original_prompt}")
        print(f"  Editing prompt: {editing_prompt}")
        if blended_word:
            print(f"  Blended words: {blended_word}")

        for edit_method in edit_method_list:
            output_dir = os.path.join(output_path, image_save_paths[edit_method], relative_structure)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("output dir:", output_dir)

            concatenated_output_path = os.path.join(output_dir, image_file)
            edited_only_path = os.path.join(
                output_dir,
                Path(image_file).stem + '_edited' + Path(image_file).suffix
            )

            # Check if already processed
            if os.path.exists(concatenated_output_path) and not args.rerun_exist_images:
                print(f"  SKIP: Output already exists")
                skipped_count += 1
                continue

            # Perform editing
            try:
                setup_seed()
                torch.cuda.empty_cache()

                start_time = time.time()
                concatenated_image, edited_image = edit_image_masked(
                    edit_method=edit_method,
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    blended_words=blended_word,
                    source_guidance_scale=1,
                    target_guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6
                )

                # Save results
                concatenated_image.save(concatenated_output_path)
                edited_image.save(edited_only_path)

                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
                processed_count += 1

                print(f"  Time: {elapsed_time:.2f}s")
                print(f"  Saved concatenated: {concatenated_output_path}")
                print(f"  Saved edited only: {edited_only_path}")

                print(f"finish")
    
            except Exception as e:
                print(f"  ERROR: Failed to process image: {str(e)}")
                error_count += 1
                continue

    # Print summary
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"Total images found: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Errors: {error_count}")

    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\nTotal editing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average time per image: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
