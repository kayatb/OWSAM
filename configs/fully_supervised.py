"""Config for the training of the fully-supervised classification model."""

model_type = "linear"
assert model_type in ("linear", "resnet")

num_layers = 5
hidden_dim = 2048

epochs = 50
lr = 1e-4
weight_decay = None
clip = None  # Gradient clipping
dropout = 0.1

ann_train = "../datasets/coco/annotations/instances_train2017.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
masks_dir = "mask_features/all"
img_train = "../datasets/coco/train2017"
img_val = "../datasets/coco/val2017"
crop_feat_dir = "dino_features/all"

num_classes = 80
batch_size = 4
num_workers = 12
pad_num = 700  # Max number of detected masks in COCO is 666.

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
