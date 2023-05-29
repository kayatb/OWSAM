"""
Config for training classification model with SAM RPN and MoCo v2 initialized ResNet-50 as feature extractor.
"""

from configs.fully_supervised.main import *

model_type = "rpn"

feature_extractor_ckpt = "checkpoints/moco_v2_800ep_pretrain.pth.tar"

epochs = 50
lr = 1e-4
batch_size = 16

dir = f"rpn_mocov2_resnet50_{lr}lr_{epochs}epochs_{batch_size}bs"
checkpoint_dir = f"checkpoints/{dir}"
log_dir = f"tensorboard_logs/{dir}"

# RNCDL settings:
# augmentations:
# random resize, random flips

# batch size = 16
# SGD optimizer for 180K iterations => 50 epochs
# linearly increase lr from 10^-3 to 10^-2 for the first 1k iterations and decrease it tenfold at iterations 120k and 160k.
