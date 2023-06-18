"""Config for the training of the fully-supervised classification models."""

# ann_train = "../datasets/coco/annotations/instances_train2017.json"
ann_train = "../datasets/coco/annotations/coco_half_train.json"
# ann_train = "../datasets/lvis/lvis_v1_train.json"
# ann_val = "../datasets/coco/annotations/instances_val2017.json"
ann_val = "../datasets/coco/annotations/coco_half_val.json"
# ann_val = "../datasets/lvis/lvis_v1_val.json"
masks_dir = "mask_features/all"
img_dir = "../datasets/coco"

num_classes = 80
num_workers = 12
pad_num = 700  # Max number of detected masks in COCO is 666.

use_mixup = False
mixup_alpha = None

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1
