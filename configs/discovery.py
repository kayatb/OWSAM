"""Config for the training of the RNCDL discovery model."""
from configs.fully_supervised.rpn import momentum, weight_decay

epochs = 10
# TODO: RNCDL uses a cosine warmup LR schedule
lr = 1e-2  # halfed compared to supervised phase.
end_lr = 1e-3  # Minimum LR for cosine annealing scheduler.
# Do warmup for n steps and go from warmup_start_lr to initial lr.
warmup_steps = 10  # 3000
warmup_start_lr = 1e-5

ann_train_labeled = "../datasets/coco/annotations/instances_train2017.json"
ann_val_labeled = "../datasets/coco/annotations/instances_val2017.json"
ann_train_unlabeled = "../datasets/lvis/lvis_v1_train.json"
ann_val_unlabeled = "../datasets/lvis/lvis_v1_val.json"

masks_dir = "mask_features/all"
img_train = "../datasets/coco/train2017"
img_val = "../datasets/coco/val2017"

feature_extractor_ckpt = "checkpoints/moco_v2_800ep_pretrain.pth.tar"

# RNCDL settings
num_labeled = 80  # + 1  # FIXME: +1 is also done in FullySupervisedClassifier, watch out!
num_unlabeled = 2100
feat_dim = 1024
hidden_dim = 512
proj_dim = 256
num_views = 2
memory_batches = 100
items_per_batch = 50  # TODO: take here the average number of masks predicted by SAM
memory_patience = 150
num_iters_sk = 3
epsilon_sk = 0.05
temperature = 0.1
supervised_loss_lambda = 0.01

num_layers = 3
hidden_dim = 256

batch_size = 2  # 16  # RNCDL uses 4*4 per GPU
num_workers = 12
pad_num = 700  # Max number of detected masks in COCO is 666.

checkpoint_dir = "checkpoints"
log_dir = "tensorboard_logs"

device = "cuda"
assert device in ("cpu", "cuda")
num_devices = 1
seed = 29

save_every = 1

# RNCDL optimizer
"""
SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)

# Use cosine LR schedule
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=0.1,
    ),
    warmup_length=0.5 * one_epoch_niter / train.max_iter,
    warmup_method="linear",
    warmup_factor=1e-3,
)

During the discovery training phase, we train for 15K iterations with a learning rate of 10^-2 and
follow a cosine decay schedule with linear ramp-up. The learning rate is linearly increased from
10^-5 to 10^-2 for the first 3K epochs and then decayed with a cosine decay schedule [51] to 10^-3.

We also use a supervised loss scale coefficient of 0.5, which results in an effective learning rate for
the supervised loss to be twice smaller than that of the discovery loss.

lvis known class ids: [3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 173, 183,
                         207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
                         611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
                         964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
                         1139, 1190, 1202]
"""
