"""Config for the training of the fully-supervised classification model."""

num_layers = 3
hidden_dim = 512

epochs = 100
lr = 1e-3
weight_decay = None
clip = None  # Gradient clipping

ann_train = "../datasets/coco/annotations/instances_train2017.json"
ann_val = "../datasets/coco/annotations/instances_train2017.json"
masks_train = "mask_features/mask_features_25"
masks_val = "mask_features/mask_features_25"
num_classes = 80
batch_size = 1
num_workers = 12
pad_num = 700  # Max number of detected masks in COCO is 666.

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
