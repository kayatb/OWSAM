"""
Config for training classification model with SAM RPN and MoCo v2 initialized ResNet-50 as feature extractor.
"""

from configs.fully_supervised.main import *

model_type = "rpn"

feature_extractor_ckpt = "checkpoints/moco_v2_800ep_pretrain.pth.tar"

epochs = 50
batch_size = 2

lr = 1e-2
momentum = 0.9
weight_decay = 1e-4
weight_decay_norm = 0.0

# Multiply the learning rate by gamma at the milestone epochs.
gamma = 0.1
milestones = [2, 4]  # [33, 44]
# Do warmup for n steps and go from warmup_start_lr to initial lr.
warmup_steps = 10  # 1000
warmup_start_lr = 1e-3

dir = f"rpn_mocov2_resnet50_SGD_{lr}lr_{epochs}epochs_{batch_size}bs"
checkpoint_dir = f"checkpoints/{dir}"
log_dir = f"tensorboard_logs/{dir}"

# RNCDL settings:
# augmentations:
# random resize, random flips

# batch size = 16
# SGD optimizer for 180K iterations => 50 epochs
# linearly increase lr from 10^-3 to 10^-2 for the first 1k iterations and decrease it tenfold at iterations 120k and 160k.
