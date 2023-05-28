"""
Config for training supervised MLP for classification of SAM bounding boxes with DINO v2 features.
"""
from configs.fully_supervised.main import *

model_type = "mlp"

num_layers = 5
hidden_dim = 2048

epochs = 50
lr = 1e-4
dropout = 0.1
batch_size = 4

use_mixup = True
mixup_alpha = 0.2

crop_feat_dir = "dino_features/all"

dir = f"mlp_{num_layers}layers_{hidden_dim}dim_{lr}lr_{dropout}dropout_{use_mixup}mixup_{mixup_alpha}mixup_alpha_{epochs}epochs_{batch_size}bs"
checkpoint_dir = f"checkpoints/{dir}"
log_dir = f"tensorboard_logs/{dir}"
