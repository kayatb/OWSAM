"""
Config for training ResNet-18 model on SAM bbox crops from images.
"""

from configs.fully_supervised.main import *

model_type = "resnet"

epochs = 50
lr = 1e-4
batch_size = 4

dir = f"resnet18_crop_{lr}lr_{epochs}epochs_{batch_size}bs"
checkpoint_dir = f"checkpoints/{dir}"
log_dir = f"tensorboard_logs/{dir}"
