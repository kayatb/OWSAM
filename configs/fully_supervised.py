"""Config for the training of the fully-supervised classification model."""

num_layers = 3
hidden_dim = 100

epochs = 10
lr = 1e-4
weight_decay = None
clip = None  # Gradient clipping

ann_train = "../datasets/coco/annotations/instances_train2017.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
embeds_train = "img_embeds"
embeds_val = "img_embeds"
num_classes = 80
batch_size = 2
num_workers = 12

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cpu"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
