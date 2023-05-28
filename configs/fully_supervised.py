"""Config for the training of the fully-supervised classification models."""

model_type = "rpn"
assert model_type in ("linear", "resnet", "rpn")

num_layers = 5
hidden_dim = 2048

epochs = 50
lr = 1e-4
weight_decay = None
clip = None  # Gradient clipping
dropout = 0.1

use_mixup = False
mixup_alpha = 0.2

ann_train = "../datasets/coco/annotations/instances_train2017.json"
# ann_train = "../datasets/lvis/lvis_v1_train.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
# ann_val = "../datasets/lvis/lvis_v1_val.json"
masks_dir = "mask_features/all"
img_train = "../datasets/coco/train2017"
img_val = "../datasets/coco/val2017"
crop_feat_dir = "dino_features/all"

feature_extractor_ckpt = "checkpoints/moco_v2_800ep_pretrain.pth.tar"

num_classes = 80
batch_size = 4
num_workers = 12
pad_num = 700  # Max number of detected masks in COCO is 666.

checkpoint_dir = f"checkpoints/{num_layers}layers_{hidden_dim}dim_{lr}lr_{dropout}dropout_{mixup_alpha}mixup_alpha_{epochs}epochs_{batch_size}bs"
log_dir = f"tensorboard_logs/{num_layers}layers_{hidden_dim}dim_{lr}lr_{dropout}dropout_{mixup_alpha}mixup_alpha_{epochs}epochs_{batch_size}bs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
