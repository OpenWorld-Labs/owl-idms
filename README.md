<div align="center">

# 🦉 OWL IDMs

<p align="center">
  This is our codebase for IDM training
</p>

---

</div>

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started is using Docker with CUDA support:

```bash
# Build the Docker image
docker build -t owl-idms .

# Run with GPU support (requires nvidia-docker)
docker run --gpus all -it -v $(pwd):/app owl-idms

# For training with data mounted
docker run --gpus all -it -v $(pwd):/app -v /path/to/your/data:/data owl-idms
```

### Option 2: Local Installation with uv

We recommend using [uv](https://github.com/astral-sh/uv) for local development:

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

2. Install dependencies using the lock file:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

**Alternative**: Install without virtual environment:
```bash
# Install PyTorch with CUDA 12.8 support
uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
uv pip install --system -r requirements.txt
```

## Training

To train an IDM model, use the training script with a configuration file:

```bash
python -m train --config_path configs/basic_adamw.yml
```

### Configuration

The training configuration is specified in YAML files. The `configs/basic_adamw.yml` file contains a complete example with:

- **Model Configuration**: 3D CNN with transformer layers
- **Training Parameters**: AdamW optimizer, batch size, learning rate, etc.
- **Data Pipeline**: Uses CoD (Call of Duty) dataset with configurable window length
- **Checkpointing**: Automatic saving and resuming from checkpoints

Key configuration sections:
- `model`: Defines the model architecture (CNN3D with ResNet + Transformer)
- `train`: Training hyperparameters and data settings
- `wandb`: Weights & Biases logging configuration

### Data Pipeline

The training uses the `CoDDataset` which expects data in the following structure:
```
/path/to/data/
├── session1/
│   └── splits/
│       ├── clip1.pt
│       ├── clip1_mouse.pt
│       ├── clip1_buttons.pt
│       └── ...
└── session2/
    └── splits/
        └── ...
```

Each clip consists of:
- Video frames (`clip.pt`): Shape `[n_frames, channels, height, width]`
- Mouse movements (`clip_mouse.pt`): Shape `[n_frames, 2]`
- Button states (`clip_buttons.pt`): Shape `[n_frames, n_buttons]`
  
Feel free to add your own data pipelines for your own use-cases, and if they are general, open a PR and we will see about merging them in!
## Inference

For inference examples, see `inference/get_sample.py` which demonstrates how to:
- Load a trained model from checkpoint
- Process video data with sliding windows
- Generate mouse and button predictions
- Visualize results

### Pre-trained Model

Download the pre-trained checkpoint (trained with `configs/basic_adamw.yml`):
- **Model**: [17k_ema_idm.pt](https://model-checkpoints.fly.storage.tigris.dev/17k_ema_idm.pt) (30MB)
- **Configuration**: `configs/basic_adamw.yml`

### Samples

Here are some samples from the pre-trained model:

**Call of Duty: Zombies**
![Zombies Sample](media/samples_zombies.gif)

**Halo Infinite**
![Halo Sample](media/samples_halo.gif)
