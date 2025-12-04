<div align="center">


# Reward Forcing: <br> Efficient Streaming Video Generation with <br> Rewarded Distribution Matching Distillation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](your-arxiv-link)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/JaydenLyh/Reward-Forcing)
[![Models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/JaydenLu666/Reward-Forcing-T2V-1.3B)

</div>

## 📢 News

- **[2025-12-05]** 🎉 Code and pretrained models released!
- **[2025-12-05]** 🌐 Project page is online!
- **[2025-12-05]** 📝 Our paper is online!


## 🎯 Overview

<div align="center">
  <img src="assets/teaser.png" width="800px">
</div>

> **TL;DR**: We propose Reward Forcing to distill a bidirectional video diffusion model into a 4-step autoregressive student model that enables real-time (23.1 FPS) streaming video generation. Instead of using vanilla distribution matching distillation (DMD), Reward Forcing adopts a novel rewarded distribution matching distillation (Re-DMD) that prioritizes matching towards high-reward regions, leading to enhanced object motion dynamics and immersive scene navigation dynamics in generated videos.



## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Pretrained Checkpoints](#-pretrained-checkpoints)
- [Inference](#-inference)
- [Training](#-training)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)


## 🔧 Hardware Requirements

- GPU: NVIDIA GPU with at least 24GB memory for inference, 80GB memory for training.
- RAM: 64GB or more recommended.
- Linux operating system.

## 🛠️ Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/JaydenLyh/Reward-Forcing.git
cd Reward-Forcing
```

### Step 2: Create conda environment
```bash
conda create -n reward_forcing python=3.10
conda activate reward_forcing
```


### Step 3: Install dependencies
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Step 4: Install the package
```bash
pip install -e .
```

## 📦 Pretrained Checkpoints
### Download Links

| Model |  Download |
|-------|----------|
| VideoReward |  [Hugging Face](https://huggingface.co/KlingTeam/VideoReward) |
| Wan2.1-T2V-1.3B |  [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B |  [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| ODE Initialization | [Hugging Face](https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/ode_init.pt) |

### File Structure
After downloading, organize the checkpoints as follows:
```
checkpoints/
├── Videoreward/
│   ├── checkpoint-11352/
│   └── model_config.json
├── Wan2.1-T2V-1.3B/
├── Wan2.1-T2V-14B/
├── Reward-Forcing/
└── ode_init.pt
```

### Quick Download Script
```bash
pip install "huggingface_hub[cli]"

# Download all checkpoints
bash download_checkpoints.sh
```


## 🚀 Inference
### Quick Start
```bash
# 5-seconds video inference
python inference.py \
    --num_output_frames 21 \
    --config_path configs/reward_forcing.yaml \
    --checkpoint_path checkpoints/Reward-Forcing/rewardforcing.pt \
    --output_folder videos/rewardforcing-5s \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema

# 30-seconds video inference
python inference.py \
    --num_output_frames 120 \
    --config_path configs/reward_forcing.yaml \
    --checkpoint_path checkpoints/Reward-Forcing/rewardforcing.pt \
    --output_folder videos/rewardforcing-30s \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

## 🏋️ Training
### Multi-GPU Training
```bash
# bash train.sh
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_PORT train.py  --config_path configs/reward_forcing.yaml \
    --logdir logs/reward_forcing \
    --disable-wandb
```

### Multi-Node Training
```bash
torchrun --nnodes=$NODE_SIZE --nproc_per_node=8 --node-rank=$NODE_RANK --rdzv_id=5235 --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT train.py  --config_path configs/reward_forcing.yaml \
    --logdir logs/reward_forcing \
    --disable-wandb
```

### Configuration Files
Training configurations are in `configs/`:
- `default_config.yaml`: Default configuration
- `reward_forcing.yaml`: Reward Forcing configuration



## 📊 Results
### Quantitative Results

#### Performance on VBench
| Method | Total Score | Quality Score | Semantic Score | Params | FPS |
|--------|----------|----------|----------|--------|-----|
| SkyReels-V2 | 82.67 | 84.70 | 74.53 | 1.3B | 0.49 |
| MAGI-1 | 79.18 | 82.04 | 67.74 | 4.5B | 0.19 |
| NOVA | 80.12 | 80.39 | 79.05 | 0.6B | 0.88 |
| Pyramid Flow | 81.72 | 84.74 | 69.62 | 2B | 6.7 |
| CausVid | 82.88 | 83.93 | 78.69 | 1.3B | 17.0 |
| Self Forcing | 83.80 | 84.59 | 80.64 | 1.3B | 17.0 |
| LongLive | 83.22 | 83.68 | **81.37** | 1.3B | 20.7 |
| **Ours** | **84.13** | **84.84** | 81.32 | 1.3B | **23.1** |


### Qualitative Results
Visualizations can be found in our [Project Page](your-project-page).




## 📄 Citation
If you find this work useful, please consider citing:

```bibtex
@inproceedings{yourname2024yourpaper,
  title={Your Paper Title},
  author={Your Name and Collaborator Names},
  booktitle={Conference/Journal Name},
  year={2024},
  pages={xxx--xxx}
}
```


## 🙏 Acknowledgements
This project is built upon several excellent works: [CausVid](https://github.com/tianweiy/CausVid), [Self Forcing](https://github.com/guandeh17/Self-Forcing), [Infinite Forcing](https://github.com/SOTAMak1r/Infinite-Forcing), [Wan2.1](https://github.com/Wan-Video/Wan2.1)

We thank the authors for their great work and open-source contribution.



## 📧 Contact
For questions and discussions, please:
- Open an issue on [GitHub Issues](link)
- Contact us at: yunhonglu@zju.edu.cn

