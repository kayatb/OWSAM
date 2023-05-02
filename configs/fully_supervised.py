"""Config for the training of the fully-supervised classification model."""

num_layers = 100
hidden_dim = 1000

epochs = 100
lr = 1e-4
weight_decay = None
clip = None  # Gradient clipping

ann_train = "../datasets/coco/annotations/instances_val2017.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
masks_train = "mask_features"
masks_val = "mask_features"
num_classes = 80
batch_size = 3
num_workers = 12
pad_num = 200

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
