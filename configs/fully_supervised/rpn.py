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
