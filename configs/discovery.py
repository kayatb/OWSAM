"""Config for the training of the RNCDL discovery model."""
num_layers = 3
hidden_dim = 256

epochs = 10
# TODO: RNCDL uses a cosine warmup LR schedule
lr = 1e-3
weight_decay = None
clip = 1.0  # Gradient clipping
clip_type = "value"

ann_train = "../datasets/coco/annotations/instances_train2017.json"
ann_val = "../datasets/coco/annotations/instances_val2017.json"
masks_train = "mask_features/mask_features_25"
masks_val = "mask_features/mask_features_25"

# RNCDL settings
num_labeled = 80 + 1  # FIXME: +1 is also done in FullySupervisedClassifier, watch out!
num_unlabeled = 2100
feat_dim = 256
hidden_dim = 512
proj_dim = 256
num_views = 2
memory_batches = 100
items_per_batch = 50
memory_patience = 150
num_iters_sk = 3
epsilon_sk = 0.05
temperature = 0.1
supervised_loss_lambda = 0.01


batch_size = 16  # RNCDL uses 4*4 per GPU
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
"""
