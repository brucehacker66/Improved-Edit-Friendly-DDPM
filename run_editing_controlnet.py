import torch
from diffusers import StableDiffusionPipeline, ControlNetModel
from diffusers import DDIMScheduler
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
from torch import autocast, inference_mode
from controlnet_aux import CannyDetector

from pipeline_utils.utils import load_512,txt_draw
from models.edit_friendly_ddm.inversion_utils import inversion_forward_process, inversion_reverse_process
from models.edit_friendly_ddm.inversion_utils_controlnet import inversion_reverse_process_controlnet
from models.edit_friendly_ddm.ptp_classes import AttentionReplace,AttentionRefine,AttentionStore
from models.edit_friendly_ddm.ptp_utils import register_attention_control


def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


image_save_paths={
    "edit-friendly-inversion+p2p+controlnet":"edit-friendly-inversion+p2p+controlnet",
    }


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
NUM_DDIM_STEPS = 50
model_id="CompVis/stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(
    model_id).to(device)
ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
ETA=1
SKIP=12

# Initialize ControlNet (Lazy loading or global?)
# For now, let's load it globally or inside the function if needed.
# We'll load Canny by default for now.
controlnet_id = "lllyasviel/sd-controlnet-canny"
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32).to(device)
canny_detector = CannyDetector()


def edit_image_EF(edit_method,
                  image_path,
                    prompt_src,
                    prompt_tar,
                    source_guidance_scale=1,
                    target_guidance_scale=7.5,cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    control_gamma=0.3,
                    control_scale=0.8,
                    control_guidance_end=0.8,
                    control_type="canny"
                    ):
    if edit_method=="edit-friendly-inversion+p2p+controlnet":
        image_gt = load_512(image_path)
        
        # Preprocess for ControlNet
        image_pil = Image.fromarray(image_gt)
        if control_type == "canny":
            control_image = canny_detector(image_pil, low_threshold=100, high_threshold=200, detect_resolution=512, image_resolution=512)
        else:
            raise NotImplementedError(f"Control type {control_type} not implemented yet")
            
        # Convert control image to tensor
        # ControlNet expects [batch, channels, height, width]
        # And normalized to [0, 1] usually?
        # diffusers ControlNet pipeline handles PIL images.
        # But here we are calling controlnet model directly.
        # We need to check what controlnet model expects.
        # Usually it expects tensor [B, C, H, W] in [0, 1].
        
        control_image_tensor = torch.from_numpy(np.array(control_image)).float() / 255.0
        if len(control_image_tensor.shape) == 2:
            control_image_tensor = control_image_tensor.unsqueeze(0).repeat(3, 1, 1)
        elif len(control_image_tensor.shape) == 3:
            control_image_tensor = control_image_tensor.permute(2, 0, 1)
            
        control_image_tensor = control_image_tensor.unsqueeze(0).to(device)
        
        image_gt = torch.from_numpy(image_gt).float() / 127.5 - 1
        image_gt = image_gt.permute(2, 0, 1).unsqueeze(0).to(device)
        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable.vae.encode(image_gt).latent_dist.mode() * 0.18215).float()
            
        controller = AttentionStore()
        register_attention_control(ldm_stable, controller)
            
        wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=ETA, prompt=prompt_src, cfg_scale=source_guidance_scale, prog_bar=True, num_inference_steps=NUM_DDIM_STEPS)
        
        controller = AttentionStore()
        register_attention_control(ldm_stable, controller)
        
        # check inversion like the baseline
        
        x0_reconstruct, _ = inversion_reverse_process(ldm_stable, xT=wts[NUM_DDIM_STEPS-SKIP], etas=ETA, prompts=[prompt_tar], cfg_scales=[target_guidance_scale], prog_bar=True, zs=zs[:(NUM_DDIM_STEPS-SKIP)], controller=controller)

        cfg_scale_list = [source_guidance_scale, target_guidance_scale]
        prompts = [prompt_src, prompt_tar]
        if (len(prompt_src.split(" ")) == len(prompt_tar.split(" "))):
            controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=ldm_stable)
        else:
            # Should use Refine for target prompts with different number of tokens
            controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=ldm_stable)

        register_attention_control(ldm_stable, controller)
        
        # duplicate control_image for batch size 2 (source and target)
        control_image_batch = torch.cat([control_image_tensor] * 2)
        
        w0, _ = inversion_reverse_process_controlnet(ldm_stable, 
                                                     controlnet=controlnet,
                                                     xT=wts[NUM_DDIM_STEPS-SKIP], 
                                                     etas=ETA, 
                                                     prompts=prompts, 
                                                     cfg_scales=cfg_scale_list, 
                                                     prog_bar=True, 
                                                     zs=zs[:(NUM_DDIM_STEPS-SKIP)], 
                                                     controller=controller,
                                                     control_image=control_image_batch,
                                                     control_gamma=control_gamma,
                                                     control_scale=control_scale,
                                                     control_guidance_end=control_guidance_end)
                                                     
        with autocast("cuda"), inference_mode():
            x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0[1].unsqueeze(0)).sample
            x0_reconstruct_edit = ldm_stable.vae.decode(1 / 0.18215 * w0[0].unsqueeze(0)).sample
            x0_reconstruct = ldm_stable.vae.decode(1 / 0.18215 * x0_reconstruct[0].unsqueeze(0)).sample
            
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        # Extract individual edited image
        edited_image = np.uint8((np.array(x0_dec[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255)
        edited_image_pil = Image.fromarray(edited_image)

        # Create concatenated visualization
        concatenated_image = Image.fromarray(np.concatenate(
                                            (
                                                image_instruct,
                                                np.uint8((np.array(image_gt[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255),
                                                np.uint8((np.array(x0_reconstruct_edit[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255),
                                                edited_image
                                            ),
                                            1
                                            )
                            )

        # Return both concatenated and individual edited images
        return concatenated_image, edited_image_pil
    else:
        raise NotImplementedError(f"No edit method named {edit_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--map_path', type=str, default="data/mapping_file.json") # the mapping file path
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["edit-friendly-inversion+p2p+controlnet"]) # the editing methods that needed to run
    parser.add_argument('--control_gamma', type=float, default=0.3)
    parser.add_argument('--control_scale', type=float, default=0.6)
    parser.add_argument('--control_guidance_end', type=float, default=0.8)
    parser.add_argument('--control_type', type=str, default="canny")
    
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    map_path=args.map_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    control_gamma=args.control_gamma
    control_scale=args.control_scale
    control_guidance_end=args.control_guidance_end
    control_type=args.control_type
    
    # Load mapping file
    print(f"\nLoading mapping file from: {map_path}")
    with open(map_path, "r") as f:
        mapping_data = json.load(f)
    print(f"Loaded {len(mapping_data)} entries from mapping file")

    # Get image files from data path
    # We need to import get_image_files or implement it if it's simple
    from pipeline_utils.utils import get_image_files
    from pathlib import Path
    import time

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
        
        item = mapping_data[image_id]
        
        if item["editing_type_id"] not in edit_category_list:
            print(f"  SKIP: Editing type {item['editing_type_id']} not in requested list")
            skipped_count += 1
            continue

        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        # image_path is already defined from the file iteration
        # editing_instruction = item["editing_instruction"]
        # blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        # mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        print(f"  Original prompt: {original_prompt}")
        print(f"  Editing prompt: {editing_prompt}")

        for edit_method in edit_method_list:
            # Construct output path preserving structure
            output_dir = os.path.join(output_path, image_save_paths[edit_method], relative_structure)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            concatenated_output_path = os.path.join(output_dir, image_file)
            edited_only_path = os.path.join(
                output_dir,
                Path(image_file).stem + '_edited' + Path(image_file).suffix
            )

            if ((not os.path.exists(concatenated_output_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                try:
                    setup_seed()
                    torch.cuda.empty_cache()
                    start_time = time.time()
                    
                    concatenated_image, edited_image = edit_image_EF(
                            edit_method=edit_method,
                            image_path=image_path,
                            prompt_src=original_prompt,
                            prompt_tar=editing_prompt,
                            source_guidance_scale=1,
                            target_guidance_scale=7.5,
                            cross_replace_steps=0.4,
                            self_replace_steps=0.6,
                            control_gamma=control_gamma,
                            control_scale=control_scale,
                            control_guidance_end=control_guidance_end,
                            control_type=control_type
                            )

                    # Save concatenated visualization image
                    concatenated_image.save(concatenated_output_path)
                    print(f"  Saved concatenated: {concatenated_output_path}")

                    # Save individual edited image for evaluation
                    edited_image.save(edited_only_path)
                    print(f"  Saved edited only: {edited_only_path}")

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    total_time += elapsed_time
                    processed_count += 1
                    print(f"  Time: {elapsed_time:.2f}s")
                    print(f"finish")
                except Exception as e:
                    print(f"  ERROR: Failed to process image: {str(e)}")
                    error_count += 1
                    continue
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
                skipped_count += 1

    # Print summary
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"Total images found: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")

    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\nTotal editing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average time per image: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
