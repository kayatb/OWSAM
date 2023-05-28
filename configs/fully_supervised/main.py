"""Config for the training of the fully-supervised classification models."""

ann_train = "../datasets/coco/annotations/instances_train2017.json"
# ann_train = "../datasets/lvis/lvis_v1_train.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
# ann_val = "../datasets/lvis/lvis_v1_val.json"
masks_dir = "mask_features/all"
img_train = "../datasets/coco/train2017"
img_val = "../datasets/coco/val2017"

num_classes = 80
num_workers = 12
pad_num = 700  # Max number of detected masks in COCO is 666.

use_mixup = False
mixup_alpha = None

device = "cpu"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
