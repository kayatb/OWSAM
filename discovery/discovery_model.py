import configs.discovery as config
from fully_supervised.model import SAMFasterRCNN
from discovery.discovery_network import DiscoveryClassifier

import torch
import torch.nn as nn


class ForwardMode:
    """Different modes for the discovery model forward pass."""

    TRAIN = 0
    SUPERVISED_VAL = 1
    UNSUPERVISED_VAL = 2


class DiscoveryModel(nn.Module):
    def __init__(self, supervised_ckpt):
        super().__init__()

        self.supervised_model = self.load_supervised_model(supervised_ckpt)
        self.remove_background_from_supervised_model()
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

    def forward(self, supervised_batch, unsupervised_batch, mode):
        if mode == ForwardMode.TRAIN:
            return self.forward_train(supervised_batch, unsupervised_batch)
        elif mode == ForwardMode.SUPERVISED_VAL:
            return self.forward_supervis_val(supervised_batch)
        elif mode == ForwardMode.UNSUPERVISED_VAL:
            return self.forward_unsupervis_val(unsupervised_batch)
        else:
            raise ValueError(f"Unknown ForwardMode {mode} given.")

    def forward_train(self, supervised_batch, unsupervised_batch):
        # Contains "loss_classifier" and "loss_box_reg"
        supervised_loss = self.supervised_model(supervised_batch)
        supervised_loss = {"supervised_" + k: v for k, v in supervised_loss.items()}

        if not self.is_discovery_memory_filled():
            supervised_loss["supervised_loss_classifier"] *= 0

        # Generate features for two different augmented images (so each in feats)
        # Extract features for the ROIs from both augmented images
        # Input to discovery_model should be list with the RoI features for each view.
        box_feats = self.supervised_model.get_box_features(unsupervised_batch, num_views=config.num_views)
        # TODO: once batched processing implemented. Split the different views.
        # box_feats = torch.split(box_feats, len(unsupervised_batch["images"]))
        # assert len(box_feats[0]) == len(box_feats[1])

        discovery_loss = self.discovery_model(box_feats)
        discovery_loss = {"discovery_" + k: v for k, v in discovery_loss.items()}

        loss = (
            supervised_loss["supervised_loss_classifier"] * self.supervised_loss_lambda
            + discovery_loss["discovery_loss"]
        )

        return loss, supervised_loss, discovery_loss

    def forward_supervis_val(self, batch):
        """Validation forward pass on supervised data."""
        self.supervised_model.eval()

        output = self.supervised_model(batch)
        return output

    def forward_unsupervis_val(self, batch):
        """Validation forward pass on unsupervised data."""
        self.supervised_model.eval()
        self.discovery_model.eval()

        box_feats = self.supervised_model.get_box_features(batch, is_discovery_train=False, num_views=1)
        output = self.discovery_model.forward_heads_single_view(box_feats)

        return output

    @torch.no_grad()
    def extract_gt_preds(self, batch):
        """Extract predictions for the GT boxes and for the predicted boxes."""
        target_boxes = [t["boxes"] for t in batch["targets"]]
        img_feats = self.supervised_model.feature_extractor(batch["images"])
        img_shapes = [img.shape for img in batch["images"]]

        preds = []
        for boxes in [batch["trans_boxes"], target_boxes]:
            roi_feats = self.supervised_model.box_roi_pool(img_feats, boxes, img_shapes)
            roi_feats = self.supervised_model.box_head(roi_feats)

            logits = self.discovery_model.forward_heads_single_view(roi_feats)
            preds.append(torch.argmax(logits, dim=-1))

            assert len(preds[-1] == len(boxes))

        return preds[0], preds[1]  # predictions, GT

    def load_supervised_model(self, ckpt_path):
        """Load the pre-trained supervised classification head."""
        ckpt_state_dict = torch.load(ckpt_path)["state_dict"]

        # Change the state dict to match the plain Torch module.
        model_state_dict = {}
        for key in ckpt_state_dict.keys():
            if key.startswith("model"):
                model_state_dict[key[len("model.") :]] = ckpt_state_dict[key]

        model = SAMFasterRCNN(config.num_labeled + 1, config.feature_extractor_ckpt, trainable_backbone_layers=0)
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

    # def load_supervised_criterion(self):
    #     # Use the same criterion as during supervised training phase.
    #     eos_coef = 0.05  # Was 0.1
    #     weight_dict = {"loss_ce": 0, "loss_bbox": 5}
    #     weight_dict["loss_giou"] = 2

    #     losses = ["labels"]

    #     matcher = HungarianMatcher()
    #     criterion = SetCriterion(
    #         self.supervised_model.num_classes,
    #         matcher,
    #         weight_dict=weight_dict,
    #         eos_coef=eos_coef,
    #         losses=losses,
    #     )
    #     criterion.empty_weight[-1] = 1  # Weight for bg class no longer necessary, because there isn't one.
    #     # criterion.to(device)

    #     return criterion


if __name__ == "__main__":
    model = DiscoveryModel("checkpoints/faster_rcnn_TUM/best_model_epoch=45.ckpt")
    # print(model.supervised_classifier)
    # print(model.state_dict()["supervised_classifier.classifier.bias"].shape)
