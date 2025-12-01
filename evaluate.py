import json
import argparse
import os
import numpy as np
from PIL import Image
import csv
import torch
from pathlib import Path
from pipeline_utils.utils import get_image_files
try:
    from evaluation.matrics_calculator import MetricsCalculator
except ImportError as e:
    print("Error: The 'torchmetrics' library is required but not installed.")
    print("Please install it using: pip install torchmetrics")
    print(f"Original error: {e}")
    exit(1)


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array

def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "lpips_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "mse_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "ssim_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "structure_distance_unedit_part":
        if (1-src_mask).sum() == 0 or (1-tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    return "nan"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--mapping_file', type=str, default="data/mapping_file.json")
    parser.add_argument('--output_csv_path', type=str, default="evaluation_results")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--edit_method', type=str, default="edit-friendly-inversion+p2p")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    metrics_calculator = MetricsCalculator(args.device)
    
    # Metrics to calculate
    metrics_list = [
        "psnr", "lpips", "mse", "ssim", "structure_distance",
        "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part", "structure_distance_unedit_part",
        "clip_similarity_source_image", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"
    ]
    
    # Load mapping file
    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file not found at {args.mapping_file}")
        return

    with open(args.mapping_file, "r") as f:
        mapping_data = json.load(f)
        
    # Prepare output directory and file
    output_dir = os.path.join(args.output_csv_path, args.edit_method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_csv_file = os.path.join(output_dir, "evaluation_results.csv")

    # Initialize CSV
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["image_id"] + metrics_list
        writer.writerow(header)
        
    print(f"Starting evaluation. Results will be saved to {output_csv_file}")
    
    processed_count = 0
    skipped_count = 0
    
    # Accumulate results for averaging
    all_results = []

    # Get images from data_path
    image_files = get_image_files(args.data_path)
    print(f"Found {len(image_files)} image files in {args.data_path}")

    if len(image_files) == 0:
        print("No images found in the directory!")
        return

    data_dir_path = Path(args.data_path)
    # Try to find 'annotation_images' in the path to preserve structure from there
    parts = data_dir_path.parts
    if 'annotation_images' in parts:
        idx = parts.index('annotation_images')
        relative_structure = Path(*parts[idx:])
    else:
        # If no 'annotation_images', just use the last part of the path
        relative_structure = Path(data_dir_path.name)

    for image_file in image_files:
        image_id = Path(image_file).stem
        
        if image_id not in mapping_data:
            print(f"Skipping {image_id}: Not found in mapping file")
            skipped_count += 1
            continue

        item = mapping_data[image_id]
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        
        # Construct paths
        src_image_path = os.path.join(args.data_path, image_file)
        
        # Expected output path
        edit_method_folder = args.edit_method
        
        tgt_folder = os.path.join(args.output_path, edit_method_folder, relative_structure)
        
        # Add _edited suffix
        tgt_filename = image_id + "_edited" + Path(image_file).suffix
        tgt_image_path = os.path.join(tgt_folder, tgt_filename)
            
        
        if not os.path.exists(tgt_image_path):
            print(f"Skipping {image_id}: Output file not found at {tgt_image_path}")
            skipped_count += 1
            continue
            
        if not os.path.exists(src_image_path):
             print(f"Skipping {image_id}: Source file not found at {src_image_path}")
             skipped_count += 1
             continue
             
        print(f"Processing {image_id}...")
        
        try:
            # Load images
            src_image = Image.open(src_image_path).convert("RGB")
            tgt_image = Image.open(tgt_image_path).convert("RGB")
            
            # Resize if needed (though they should match)
            if src_image.size != tgt_image.size:
                tgt_image = tgt_image.resize(src_image.size)
                
            # Decode mask
            mask = mask_decode(item["mask"])
            mask = mask[:, :, np.newaxis].repeat(3, axis=2)
            
            # Calculate metrics
            row = [image_id]
            metric_values = []
            for metric in metrics_list:
                val = calculate_metric(metrics_calculator, metric, src_image, tgt_image, mask, mask, original_prompt, editing_prompt)
                row.append(val)
                if val != "nan":
                    metric_values.append(float(val))
                else:
                    metric_values.append(np.nan)
            
            all_results.append(metric_values)

            # Append to CSV
            with open(output_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            skipped_count += 1
            
    print(f"\nEvaluation complete.")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results saved to {output_csv_file}")
    
    # Calculate and save summary
    if processed_count > 0:
        all_results = np.array(all_results)
        # Calculate mean, ignoring NaNs
        means = np.nanmean(all_results, axis=0)
        
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append("Evaluation Summary")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total Processed: {processed_count}")
        summary_lines.append(f"Total Skipped: {skipped_count}")
        summary_lines.append("-" * 50)
        
        print("\n" + "\n".join(summary_lines))
        
        for i, metric in enumerate(metrics_list):
            line = f"{metric}: {means[i]:.4f}"
            print(line)
            summary_lines.append(line)
            
        summary_file = output_csv_file.replace(".csv", "_summary.txt")
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))
            
        print(f"\nSummary saved to {summary_file}")
    else:
        print("\nNo images processed, skipping summary.")

if __name__ == "__main__":
    main()
