# Integrated Fast Image Editing Pipeline

**GNRI (Guided Newton-Raphson Inversion) + Edit Friendly P2P**

This pipeline combines the best of both worlds:
- **Speed**: GNRI's lightning-fast Newton-Raphson inversion (~0.4 seconds)
- **Quality**: Edit Friendly DDPM's structure-preserving Prompt-to-Prompt attention control

## Overview

This integrated pipeline achieves **real-time image editing** (total ~2-3 seconds per edit) while maintaining high structural fidelity. It replaces the slow stochastic inversion of Edit Friendly DDPM (~30 seconds) with GNRI's fast deterministic solver.

### Performance Comparison

| Method | Inversion Time | Total Edit Time | Quality |
|--------|---------------|-----------------|---------|
| **Edit Friendly DDPM** (Original) | ~30s | ~40s | High ✓ |
| **GNRI** (Fast but basic) | ~0.4s | ~2s | Medium |
| **This Pipeline** (Integrated) | ~0.4s | **~2-3s** | **High ✓** |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Input Image + Prompts                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              VAE Encoding (z₀)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  GNRI Inversion: z₀ → zₜ                                   │
│  • Newton-Raphson Optimization (2 iterations)               │
│  • Gaussian Prior Guidance (α=0.1)                          │
│  • 4 timesteps (SDXL-Turbo)                                 │
│  ⚡ Time: ~0.4s                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  P2P Attention Control + Denoising: zₜ → z₀'               │
│  • AttentionReplace/AttentionRefine                         │
│  • Cross-attention injection (40%)                          │
│  • Self-attention injection (60%)                           │
│  ⚡ Time: ~1-2s                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              VAE Decoding (Edited Image)                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA-capable GPU (recommended)
nvidia-smi
```

### Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision diffusers transformers accelerate
pip install numpy pillow matplotlib tqdm
```

### Download Model Weights

The pipeline will automatically download SDXL-Turbo on first run. To specify a cache directory:

```bash
export HF_HOME=/path/to/cache
```

## Usage

### Quick Start (Command Line)

```bash
# Basic usage
python main.py \
    --image_path path/to/image.jpg \
    --source_prompt "a photo of a cat" \
    --target_prompt "a photo of a dog"

# With custom parameters
python main.py \
    --image_path path/to/image.jpg \
    --source_prompt "a lion sitting in grass" \
    --target_prompt "a tiger sitting in grass" \
    --output_path output/my_edits \
    --cross_replace_steps 0.4 \
    --self_replace_steps 0.6 \
    --target_guidance_scale 1.2
```

### Python API

```python
from integrated_editing_pipeline import IntegratedFastEditor, setup_seed

# Set random seed for reproducibility
setup_seed(42)

# Initialize editor
editor = IntegratedFastEditor(
    model_id="stabilityai/sdxl-turbo",
    device="cuda",
    num_inference_steps=4,
)

# Run full pipeline
results = editor.run_full_pipeline(
    image_path="path/to/image.jpg",
    source_prompt="a photo of a cat",
    target_prompt="a photo of a dog",
    source_guidance_scale=1.0,
    target_guidance_scale=1.2,
    cross_replace_steps=0.4,
    self_replace_steps=0.6,
)

# Access results
original = results["original"]
reconstructed = results["reconstructed"]
edited = results["edited"]
timings = results["timings"]

# Save edited image
edited.save("output/edited.png")
```

### Advanced: Batch Processing

```python
# Load and encode image once
image_latent = editor.load_and_encode_image("path/to/image.jpg")

# Invert once
inverted_latent, trajectory = editor.invert_image(
    image_latent=image_latent,
    source_prompt="a photo of a cat",
)

# Edit multiple times with different prompts
target_prompts = ["a photo of a dog", "a photo of a tiger", "a photo of a lion"]

for target_prompt in target_prompts:
    edited_image = editor.edit_image(
        inverted_latent=inverted_latent,
        source_prompt="a photo of a cat",
        target_prompt=target_prompt,
        trajectory=trajectory,
    )
    edited_image.save(f"output/{target_prompt.replace(' ', '_')}.png")
```

## Parameters

### Main Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_inference_steps` | 4 | 1-50 | Number of diffusion steps (4 for Turbo models) |
| `cross_replace_steps` | 0.4 | 0.0-1.0 | Ratio of steps to replace cross-attention |
| `self_replace_steps` | 0.6 | 0.0-1.0 | Ratio of steps to replace self-attention |
| `source_guidance_scale` | 1.0 | 0.0-10.0 | CFG scale for inversion |
| `target_guidance_scale` | 1.2 | 0.0-10.0 | CFG scale for editing |

### GNRI-Specific Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_iters` | 2 | 1-10 | Newton-Raphson iterations per timestep |
| `alpha` | 0.1 | 0.0-1.0 | Gaussian prior guidance weight |

### Parameter Tuning Guide

- **For more structural preservation**: Increase `cross_replace_steps` and `self_replace_steps`
- **For more creative edits**: Decrease `cross_replace_steps` and `self_replace_steps`
- **For faster inference**: Reduce `n_iters` (may reduce quality)
- **For better inversion quality**: Increase `n_iters` or `alpha`

## Examples

### Example 1: Animal Transformation

```bash
python main.py \
    --image_path examples/lion.jpg \
    --source_prompt "a lion sitting in the grass at sunset" \
    --target_prompt "a tiger sitting in the grass at sunset" \
    --output_path output/animal_transform
```

**Result**: The lion's structure and pose are preserved while the texture changes to tiger stripes.

### Example 2: Style Transfer

```bash
python main.py \
    --image_path examples/portrait.jpg \
    --source_prompt "a photo of a person" \
    --target_prompt "an oil painting of a person" \
    --cross_replace_steps 0.6 \
    --self_replace_steps 0.8
```

**Result**: The person's identity and pose are preserved while the style changes to oil painting.

### Example 3: Object Replacement

```bash
python main.py \
    --image_path examples/scene.jpg \
    --source_prompt "a car on a street" \
    --target_prompt "a bicycle on a street" \
    --cross_replace_steps 0.3 \
    --self_replace_steps 0.5
```

**Result**: The car is replaced with a bicycle while maintaining the scene structure.

## Demo Script

Run the interactive demo to explore the pipeline:

```bash
python demo.py
```

This will present options for:
1. Basic image editing
2. Multiple edits from single inversion
3. Parameter comparison

## Project Structure

```
IntegratedPipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── models/
│   └── gnri_inversion_utils.py       # GNRI inversion implementation
├── integrated_editing_pipeline.py     # Main pipeline class
├── main.py                            # Command-line interface
└── demo.py                            # Interactive demo script
```

## Technical Details

### GNRI Inversion

The GNRI inversion step solves for the noise latent zₜ using Newton-Raphson optimization:

```
z_{t+1} = z_t - [f(z_t) / ∇f(z_t)]
```

Where the objective function includes:
- **Reconstruction error**: ||z_t - denoise(z_t)||
- **Gaussian prior**: α · ||z_t - 𝒩(0,σ)||

This ensures:
1. The inverted noise reconstructs the original image
2. The noise follows a realistic Gaussian distribution

### Prompt-to-Prompt Attention Control

During denoising, the pipeline:
1. Injects cross-attention maps from source prompt to target prompt
2. Preserves self-attention for structural consistency
3. Uses `AttentionReplace` for same-length prompts
4. Uses `AttentionRefine` for different-length prompts

## Troubleshooting

### Out of Memory

If you encounter CUDA out of memory errors:

```python
# Use lower precision
editor = IntegratedFastEditor(dtype=torch.float32)  # or torch.float16

# Use CPU (slower)
editor = IntegratedFastEditor(device="cpu")

# Clear cache between runs
torch.cuda.empty_cache()
```

### Poor Edit Quality

Try adjusting:
- Increase `cross_replace_steps` and `self_replace_steps` for more preservation
- Increase `target_guidance_scale` for stronger edits
- Increase `n_iters` for better inversion quality

### Import Errors

Make sure parent directories are in the Python path:

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "PnPInversion"))
sys.path.append(str(project_root / "NewtonRaphsonInversion"))
```

## Citation

If you use this integrated pipeline in your research, please cite both original papers:

```bibtex
@article{gnri2024,
  title={GNRI: Lightning-Fast Image Inversion and Editing for Text-to-Image Diffusion Models},
  author={Samuel, Dvir and Meiri, Barak and Maron, Haggai and Tewel, Yoad and Darshan, Nir and Chechik, Gal and Avidan, Shai and Ben-Ari, Rami},
  year={2024}
}

@article{editfriendly2023,
  title={An Edit Friendly DDPM Noise Space: Inversion and Manipulations},
  author={Huberman-Spiegelglas, Inbar and Kulikov, Vladimir and Michaeli, Tomer},
  journal={arXiv preprint arXiv:2304.06140},
  year={2023}
}
```

## License

This integrated pipeline combines code from:
- GNRI (NewtonRaphsonInversion): See original repository for license
- Edit Friendly DDPM (PnPInversion): See original repository for license

## Acknowledgments

This work integrates:
- **GNRI** by Dvir Samuel et al. (OriginAI, Tel Aviv University, Technion, Bar Ilan University, NVIDIA Research)
- **Edit Friendly DDPM** by Inbar Huberman-Spiegelglas et al. (Technion)
- **Prompt-to-Prompt** by Amir Hertz et al. (Google Research)

Special thanks to the Diffusers library by HuggingFace.

## Contact

For issues, questions, or contributions, please open an issue on the GitHub repository.
# Faster-Edit-Friendly-DDPM
