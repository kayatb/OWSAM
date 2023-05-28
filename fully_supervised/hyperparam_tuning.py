import configs.fully_supervised.mlp as config
from fully_supervised.train import LitFullySupervisedClassifier, load_data

import lightning.pytorch as pl
from ray import tune
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler  # , PopulationBasedTraining


# The maximum training epochs per sample.
num_epochs = 5

# Number of samples from parameter space.
num_samples = 100

accelerator = "cpu"

# Hyperparameter space.
tune_config = {
    "hidden_dim": tune.choice([128, 256, 512, 1024]),
    "num_layers": tune.choice([i + 1 for i in range(10)]),
    "lr": tune.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]),
    # "batch_size": tune.choice([1, 2, 4, 8, 16]),
}

dataloader_train, dataloader_val = load_data(batch_size=8)

lightning_config = (
    LightningConfigBuilder()
    .module(cls=LitFullySupervisedClassifier, device="cpu", tune_config=tune_config)
    .trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        logger=pl.loggers.tensorboard.TensorBoardLogger(save_dir=config.log_dir),
    )
    .fit_params(train_dataloaders=dataloader_train, val_dataloader=dataloader_val)
    .checkpointing(monitor="val_map", save_top_k=2, mode="max")
    .build()
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_map",
        checkpoint_score_order="max",
    ),
)

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

scaling_config = ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 1})

# Define a base LightningTrainer without hyper-parameters for Tuner
lightning_trainer = LightningTrainer(
    scaling_config=scaling_config,
    run_config=run_config,
)


def tune_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            metric="val_map",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="tune_asha",
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
    best_result


tune_asha(num_samples=num_samples)
