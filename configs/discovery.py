"""Config for the training of the RNCDL discovery model."""
num_layers = 3
hidden_dim = 256

epochs = 10
# TODO: RNCDL uses a cosine warmup LR schedule
lr = 1e-3
weight_decay = None
clip = 1.0  # Gradient clipping
clip_type = "value"

ann_train_labeled = "../datasets/coco/annotations/instances_train2017.json"
ann_val_labeled = "../datasets/coco/annotations/instances_val2017.json"
ann_train_unlabeled = "../datasets/lvis/lvis_v1_train.json"
ann_val_unlabeled = "../datasets/lvis/lvis_v1_val.json"

masks_dir = "mask_features/all"
img_train = "../datasets/coco/train2017"
img_val = "../datasets/coco/val2017"

feature_extractor_ckpt = "checkpoints/moco_v2_800ep_pretrain.pth.tar"

# Supervised classifier settings
supervis_num_layers = 10
supervis_hidden_dim = 512

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

During the discovery training phase, we train for 15K iterations with a learning rate of 10^-2 and
follow a cosine decay schedule with linear ramp-up. The learning rate is linearly increased from
10^-5 to 10^-2 for the first 3K epochs and then decayed with a cosine decay schedule [51] to 10^-3.

We also use a supervised loss scale coefficient of 0.5, which results in an effective learning rate for
the supervised loss to be twice smaller than that of the discovery loss.
"""
