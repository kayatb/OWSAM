import configs.discovery as config
from fully_supervised.model import FullySupervisedClassifier
from discovery.discovery_network import DiscoveryClassifier

import torch
import torch.nn as nn


class DiscoveryModel(nn.Module):
    def __init__(self, supervised_ckpt):
        super().__init__()

        self.supervised_classifier = self.load_supervised_classifier(supervised_ckpt)
        self.remove_background_from_supervised_model()
        self.supervised_criterion = self.load_supervised_criterion()
        self.supervised_loss_lambda = config.supervised_loss_lambda

        self.discovery_classifier = DiscoveryClassifier(
            num_labeled=config.num_labeled,
            num_unlabeled=config.num_unlabeled,
            feat_dim=config.feat_dim,
            hidden_dim=config.hidden_dim,
            proj_dim=config.proj_dim,
            num_views=config.num_views,
            memory_batches=config.memory_batches,
            items_per_batch=config.items_per_batch,
            memory_patience=config.memory_patience,
            num_iters_sk=config.num_iters_sk,
            epsilon_sk=config.epsilon_sk,
            temperature=config.temperature,
            batch_size=config.batch_size,
            num_hidden_layers=config.num_layers,
        )

    def forward(self, supervised_batch, unsupervised_batch):
        supervised_output = self.supervised_classifier(supervised_batch)
        # contains "supervised_loss" and "supervised_class_error"
        supervised_loss = self.supervised_criterion(supervised_output)
        supervised_loss = {"supervised_" + k: v for k, v in supervised_loss.items()}
        supervised_loss["supervised_ce_loss"] = supervised_loss["supervised_loss"].item()

        if not self.is_discovery_memory_filled():
            supervised_loss["supervised_loss"] *= 0

        # TODO: change discovery network forward pass to accomodate batch.
        discovery_loss = self.discovery_classifier(unsupervised_batch)
        discovery_loss = {"discovery_" + k: v for k, v in discovery_loss.items()}

        loss = supervised_loss["supervised_loss"] * self.supervised_loss_labmda + discovery_loss["discovery_loss"])

        return loss, supervised_loss, discovery_loss

    def load_supervised_classifier(self, ckpt_path):
        """Load the pre-trained supervised classification head."""
        ckpt_state_dict = torch.load(ckpt_path)["state_dict"]

        # Change the state dict to match the plain Torch module.
        del ckpt_state_dict["criterion.empty_weight"]
        model_state_dict = {}
        for key in ckpt_state_dict.keys():
            model_state_dict[key[6:]] = ckpt_state_dict[key]

        model = FullySupervisedClassifier(config.supervis_num_layers, config.supervis_hidden_dim, config.num_labeled)
        model.load_state_dict(model_state_dict)

        return model

    # TODO: test this
    def remove_background_from_supervised_model(self):
        """Remove the background class from the supervised classifier to enable discovery of new classes."""
        classifier = self.supervised_classifier.classifier
        classifier.weight = nn.Parameter(classifier.weight[:-1])
        classifier.bias = nn.Parameter(classifier.bias[:-1])

    def is_discovery_memory_filled(self):
        return self.discovery_classifier.memory_patience == 0

    def load_supervised_criterion(self):
        raise NotImplementedError


if __name__ == "__main__":
    model = DiscoveryModel("checkpoints/10_512_1e-4/best_model_epoch=23.ckpt")
    print(model.supervised_classifier)
