"""Config for the training of the fully-supervised classification model."""

num_layers = 10
hidden_dim = 100

epochs = 500
lr = 1e-3
weight_decay = None
clip = None  # Gradient clipping

ann_train = "../datasets/coco/annotations/instances_val2017.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
masks_train = "mask_features_single"
masks_val = "mask_features_single"
num_classes = 80
batch_size = 1
num_workers = 12
pad_num = 138

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
