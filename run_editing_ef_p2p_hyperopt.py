"""
Edit-Friendly DDPM + P2P with Hyperparameter Optimizations

Improvements implemented:
1. Decreased SKIP steps (15-20 instead of 12) - introduces more randomness early
2. Increased CFG scale (12-15 instead of 7.5) - stronger text guidance
3. Relaxed cross-attention injection (0.2-0.4 instead of 0.4) - allows more semantic change
4. Kept self-attention high (0.6-0.8) - preserves structure

Expected: Higher CLIP similarity while maintaining structural preservation
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
from models.edit_friendly_ddm.ptp_classes import AttentionReplace, AttentionRefine, AttentionStore
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


image_save_paths = {
    "hyperopt-skip15": "hyperopt-skip15",
    "hyperopt-skip18": "hyperopt-skip18",
    "hyperopt-skip20": "hyperopt-skip20",
    "hyperopt-cfg12": "hyperopt-cfg12",
    "hyperopt-cfg15": "hyperopt-cfg15",
    "hyperopt-cross0.2": "hyperopt-cross0.2",
    "hyperopt-cross0.3": "hyperopt-cross0.3",
    "hyperopt-combined": "hyperopt-combined",
}


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_DDIM_STEPS = 50
model_id = "CompVis/stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
ETA = 1


def edit_image_hyperopt(
    edit_method,
    image_path,
    prompt_src,
    prompt_tar,
    source_guidance_scale=1,
    target_guidance_scale=7.5,
    cross_replace_steps=0.4,
    self_replace_steps=0.6,
    skip_steps=12
):
    """
    Edit image with hyperparameter optimizations

    Args:
        edit_method: Which hyperopt configuration to use
        image_path: Path to input image
        prompt_src: Source prompt
        prompt_tar: Target prompt
        source_guidance_scale: CFG for inversion (default 1)
        target_guidance_scale: CFG for editing (tunable, default 7.5)
        cross_replace_steps: Fraction of steps to inject cross-attention (tunable)
        self_replace_steps: Fraction of steps to inject self-attention (tunable)
        skip_steps: Number of steps to skip from end (tunable)
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

    # Hyperparameter configurations
    configs = {
        "hyperopt-skip15": {"skip": 15, "cfg": 7.5, "cross": 0.4, "self": 0.6},
        "hyperopt-skip18": {"skip": 18, "cfg": 7.5, "cross": 0.4, "self": 0.6},
        "hyperopt-skip20": {"skip": 20, "cfg": 7.5, "cross": 0.4, "self": 0.6},
        "hyperopt-cfg12": {"skip": 12, "cfg": 12.0, "cross": 0.4, "self": 0.6},
        "hyperopt-cfg15": {"skip": 12, "cfg": 15.0, "cross": 0.4, "self": 0.6},
        "hyperopt-cross0.2": {"skip": 12, "cfg": 7.5, "cross": 0.2, "self": 0.6},
        "hyperopt-cross0.3": {"skip": 12, "cfg": 7.5, "cross": 0.3, "self": 0.6},
        "hyperopt-combined": {"skip": 18, "cfg": 12.0, "cross": 0.3, "self": 0.7},
    }

    if edit_method not in configs:
        raise NotImplementedError(f"No edit method named {edit_method}")

    config = configs[edit_method]
    SKIP = config["skip"]
    target_cfg = config["cfg"]
    cross_steps = config["cross"]
    self_steps = config["self"]

    # Reconstruction (optional, for visualization)
    controller = AttentionStore()
    register_attention_control(ldm_stable, controller)

    x0_reconstruct, _ = inversion_reverse_process(
        ldm_stable,
        xT=wts[NUM_DDIM_STEPS - SKIP],
        etas=ETA,
        prompts=[prompt_tar],
        cfg_scales=[target_cfg],
        prog_bar=True,
        zs=zs[:(NUM_DDIM_STEPS - SKIP)],
        controller=controller
    )

    # Editing with P2P
    cfg_scale_list = [source_guidance_scale, target_cfg]
    prompts = [prompt_src, prompt_tar]

    if len(prompt_src.split(" ")) == len(prompt_tar.split(" ")):
        controller = AttentionReplace(
            prompts, NUM_DDIM_STEPS,
            cross_replace_steps=cross_steps,
            self_replace_steps=self_steps,
            model=ldm_stable
        )
    else:
        controller = AttentionRefine(
            prompts, NUM_DDIM_STEPS,
            cross_replace_steps=cross_steps,
            self_replace_steps=self_steps,
            model=ldm_stable
        )

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
        x0_reconstruct = ldm_stable.vae.decode(1 / 0.18215 * x0_reconstruct[0].unsqueeze(0)).sample

    image_instruct = txt_draw(
        f"Method: {edit_method}\\nSkip: {SKIP}, CFG: {target_cfg}, Cross: {cross_steps}, Self: {self_steps}\\n"
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
    parser.add_argument('--edit_method_list', nargs='+', type=str, default=["hyperopt-combined"])
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

        print(f"  Original prompt: {original_prompt}")
        print(f"  Editing prompt: {editing_prompt}")

        for edit_method in edit_method_list:
            output_dir = os.path.join(output_path, image_save_paths[edit_method], relative_structure)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("output dir:", output_dir)
            
            # editing_instruction_text = entry["editing_instruction"]
            # blended_word = entry["blended_word"].split(" ") if entry["blended_word"] != "" else []
            # mask = Image.fromarray(np.uint8(mask_decode(entry["mask"])[:, :, np.newaxis].repeat(3, 2))).convert("L")

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
                concatenated_image, edited_image = edit_image_hyperopt(
                    edit_method=edit_method,
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
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

