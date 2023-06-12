import configs.discovery as config
from fully_supervised.model import SAMRPN
from discovery.discovery_network import DiscoveryClassifier

from modelling.criterion import SetCriterion
from modelling.matcher import HungarianMatcher

import torch
import torch.nn as nn


class DiscoveryModel(nn.Module):
    def __init__(self, supervised_ckpt):
        super().__init__()

        self.supervised_model = self.load_supervised_model(supervised_ckpt)
        self.remove_background_from_supervised_model()
        # Freeze all components except for the classification head.
        # self.supervised_model.feature_extractor.freeze()
        # self.supervised_model.box_roi_pool.freeze()
        # self.supervised_model.box_head.freeze()
        self.supervised_criterion = self.load_supervised_criterion()
        self.supervised_loss_lambda = config.supervised_loss_lambda

        self.discovery_model = DiscoveryClassifier(
            self.supervised_model.classifier,
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
            sk_mode="lognormal",
        )

    def forward(self, supervised_batch, unsupervised_batch):
        supervised_output = None
        supervised_loss = {"supervised_loss": 0.0}
        discovery_output = None
        discovery_loss = {"discovery_loss": 0.0}

        if supervised_batch is not None:
            supervised_output = self.supervised_model(supervised_batch)
            # print("pred", torch.argmax(supervised_output["pred_logits"][: supervised_batch["num_masks"][0]], dim=-1))
            # print("target", supervised_batch["targets"][0]["labels"])
            # Contains "supervised_loss" and "supervised_class_error"
            supervised_loss = self.supervised_criterion(supervised_output, supervised_batch["targets"])
            supervised_loss = {"supervised_" + k: v for k, v in supervised_loss.items()}
            supervised_loss["supervised_ce_loss"] = supervised_loss["supervised_loss"].item()

            if not self.is_discovery_memory_filled():
                supervised_loss["supervised_loss"] *= 0

        if unsupervised_batch is not None:
            # Generate features for two different augmented images (so each in feats)
            # Extract features for the ROIs from both augmented images
            # Input to discovery_model should be list with the RoI features for each view.
            if self.training:
                roi_feats = []
                for i in range(len(unsupervised_batch["images"])):
                    with torch.no_grad():
                        img_feat = self.supervised_model.feature_extractor(unsupervised_batch["images"][i])
                        img_shapes = [img.shape[1:] for img in unsupervised_batch["images"][i]]
                        roi_feat = self.supervised_model.box_roi_pool(
                            img_feat, unsupervised_batch["trans_boxes"][i], img_shapes
                        )
                        roi_feat = self.supervised_model.box_head(roi_feat)
                    roi_feats.append(roi_feat)

                discovery_loss = self.discovery_model(roi_feats)
                discovery_loss = {"discovery_" + k: v for k, v in discovery_loss.items()}
            else:
                with torch.no_grad():
                    img_feat = self.supervised_model.feature_extractor(unsupervised_batch["images"])
                    img_shapes = [img.shape[1:] for img in unsupervised_batch["images"]]
                    roi_feat = self.supervised_model.box_roi_pool(
                        img_feat, unsupervised_batch["trans_boxes"], img_shapes
                    )
                    roi_feat = self.supervised_model.box_head(roi_feat)
                discovery_output = self.discovery_model.forward_heads_single_view(roi_feat)

        loss = supervised_loss["supervised_loss"] * self.supervised_loss_lambda + discovery_loss["discovery_loss"]

        return loss, supervised_loss, discovery_loss, supervised_output, discovery_output

    def load_supervised_model(self, ckpt_path):
        """Load the pre-trained supervised classification head."""
        ckpt_state_dict = torch.load(ckpt_path)["state_dict"]

        # Change the state dict to match the plain Torch module.
        del ckpt_state_dict["criterion.empty_weight"]
        model_state_dict = {}
        for key in ckpt_state_dict.keys():
            if key.startswith("model"):
                model_state_dict[key[len("model.") :]] = ckpt_state_dict[key]

        model = SAMRPN(
            config.num_labeled, config.feature_extractor_ckpt, pad_num=config.pad_num, trainable_backbone_layers=0
        )
        model.load_state_dict(model_state_dict)
        model.freeze()

        return model

    # TODO: test this
    def remove_background_from_supervised_model(self):
        """Remove the background class from the supervised classifier to enable discovery of new classes."""
        classifier = self.supervised_model.classifier
        classifier.weight = nn.Parameter(classifier.weight[:-1])
        classifier.bias = nn.Parameter(classifier.bias[:-1])
        self.supervised_model.num_classes -= 1

    def is_discovery_memory_filled(self):
        return self.discovery_model.memory_patience == 0

    def load_supervised_criterion(self):
        # Use the same criterion as during supervised training phase.
        eos_coef = 0.05  # Was 0.1
        weight_dict = {"loss_ce": 0, "loss_bbox": 5}
        weight_dict["loss_giou"] = 2

        losses = ["labels"]

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            self.supervised_model.num_classes,
            matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
        )
        criterion.empty_weight[-1] = 1  # Weight for bg class no longer necessary, because there isn't one.
        # criterion.to(device)

        return criterion


if __name__ == "__main__":
    model = DiscoveryModel("checkpoints/rpn_TUMlike/best_model_epoch=45.ckpt")
    # print(model.supervised_classifier)
    # print(model.state_dict()["supervised_classifier.classifier.bias"].shape)
