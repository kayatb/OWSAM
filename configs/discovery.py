"""Config for the training of the RNCDL discovery model."""
from configs.fully_supervised.rpn import momentum, weight_decay

epochs = 10  # FIXME: RNCDL trains for 15K iterations, which is around 2.4 epochs
lr = 1e-2  # halfed compared to supervised phase.
end_lr = 1e-3  # Minimum LR for cosine annealing scheduler.
# Do warmup for n steps and go from warmup_start_lr to initial lr.
warmup_steps = 3000
warmup_start_lr = 1e-5

ann_train_labeled = "../datasets/coco/annotations/coco_half_train.json"
ann_val_labeled = "../datasets/coco/annotations/coco_half_val.json"
ann_train_unlabeled = "../datasets/lvis/lvis_v1_train.json"
ann_val_unlabeled = "../datasets/lvis/lvis_v1_val.json"

# ann_train_labeled = "../datasets/coco/annotations/instances_val2017.json"
# ann_val_labeled = "../datasets/coco/annotations/instances_val2017.json"
# ann_train_unlabeled = "../datasets/coco/annotations/instances_val2017.json"
# ann_val_unlabeled = "../datasets/coco/annotations/instances_val2017.json"

masks_dir = "mask_features/all"
img_dir = "../datasets/coco"

feature_extractor_ckpt = "checkpoints/moco_v2_800ep_pretrain.pth.tar"
supervis_ckpt = "checkpoints/faster_rcnn_TUM/best_model_epoch=45.ckpt"

# RNCDL settings
num_labeled = 80
num_unlabeled = 3000  # 2100
feat_dim = 1024
hidden_dim = 512
proj_dim = 256
num_views = 2
memory_batches = 100
items_per_batch = 50
memory_patience = 150
num_iters_sk = 3
epsilon_sk = 0.05
temperature = 0.1
supervised_loss_lambda = 0.5  # 0.01

num_layers = 3
hidden_dim = 256

batch_size = 16  # RNCDL uses 4*4 per GPU
num_workers = 12

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1

# Settings for mapping discovered classes to GT classes.
novel_class_id_thresh = 10000
max_class_num = 20000
last_free_class_id = 10000

# Class IDs of known classes in LVIS dataset, used for evaluation.
# These IDs are as they occur in the GT annotation of LVIS.
lvis_known_class_ids = [
    3,
    12,
    34,
    35,
    36,
    41,
    45,
    58,
    60,
    76,
    77,
    80,
    90,
    94,
    99,
    118,
    127,
    133,
    139,
    154,
    173,
    183,
    207,
    217,
    225,
    230,
    232,
    271,
    296,
    344,
    367,
    378,
    387,
    421,
    422,
    445,
    469,
    474,
    496,
    534,
    569,
    611,
    615,
    631,
    687,
    703,
    705,
    716,
    735,
    739,
    766,
    793,
    816,
    837,
    881,
    912,
    923,
    943,
    961,
    962,
    964,
    976,
    982,
    1000,
    1019,
    1037,
    1071,
    1077,
    1079,
    1095,
    1097,
    1102,
    1112,
    1115,
    1123,
    1133,
    1139,
    1190,
    1202,
]
