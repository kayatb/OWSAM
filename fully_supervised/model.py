from utils.misc import add_padding, box_xywh_to_xyxy

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead


class LinearClassifier(nn.Module):
    """A simple classification head on top of the hidden mask features extracted from SAM to classify the masks."""

    def __init__(self, num_layers, hidden_dim, num_classes, dropout_prob, pad_num=700, input_dim=1024):
        super().__init__()

        self.num_classes = num_classes
        self.pad_num = pad_num
        self.input_dim = input_dim

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.ELU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.ELU())

        self.classifier = nn.Linear(hidden_dim, self.num_classes + 1)  # +1 for the no-object/background class

    def forward(self, batch):
        # x = self.layers(batch["mask_features"])
        x = self.layers(batch["crop_features"].float())
        # x = batch["crop_features"].float()
        class_logits = self.classifier(x)

        padded_class_logits = add_padding(
            class_logits, batch["num_masks"], self.num_classes, self.pad_num, batch["boxes"].device
        )

        return {
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pad_num=700):
        super().__init__()
        self.num_classes = num_classes
        self.pad_num = pad_num

        # self.resnet = resnet18(weights=None)
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(dim, num_classes + 1)  # +1 for no-object/background class

    def forward(self, batch):
        class_logits = self.resnet(batch["crops"])

        padded_class_logits = add_padding(
            class_logits, batch["num_masks"], self.num_classes, self.pad_num, batch["boxes"].device
        )

        return {
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


class SAMRPN(nn.Module):
    """Boxes extracted from SAM as ROI proposals (i.e. SAM is RPN), then ROI align on
    the feature map extracted from the whole image, and classification of those ROIs."""

    def __init__(self, num_classes, feature_extractor_ckpt, trainable_backbone_layers=5, pad_num=700):
        """Args:
        num_classes: number of classes to predict, without the background class
        feature_extractor_ckpt: location of the checkpoint for the feature extractor
        image_size: square size to which the images are resized
        trainable_backbone_layers: number of non-frozen layers starting from the final block. Valid values
        are between 0 and 5, with 5 meaning all blocks are trainable."""
        # RNCDL uses ResNet-50 FPN with MoCo v2 initialization (https://github.com/facebookresearch/moco)
        # They also add AMP and SyncBatchNorm to stabilize training
        super().__init__()
        self.num_classes = num_classes
        self.pad_num = pad_num

        self.feature_extractor = self.load_resnet50_fpn_with_moco(feature_extractor_ckpt, trainable_backbone_layers)
        self.load_roi_heads(self.num_classes + 1)

    def load_resnet50_fpn_with_moco(self, checkpoint_path, trainable_backbone_layers):
        moco_checkpoint = torch.load(checkpoint_path)

        # Rename MoCo pre-trained keys (taken from official MoCo repo):
        # https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L250
        state_dict = moco_checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # Retain only encoder_q up to before the embedding layer.
            if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                # Remove prefix.
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # Delete renamed or unused k.
            del state_dict[k]

        backbone = resnet50(weights=None)
        # Change the last ResNet block strides from (2, 2) to (1, 1)
        # to increase the output feature map from 14x14 to 28x28.
        backbone.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        backbone.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # Load the weights into ResNet-50
        backbone.load_state_dict(state_dict, strict=False)

        # Turn it into an FPN feature extractor.
        feature_extractor = _resnet_fpn_extractor(
            backbone, trainable_layers=trainable_backbone_layers, norm_layer=nn.BatchNorm2d
        )

        return feature_extractor

    def load_roi_heads(self, num_classes):
        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = self.box_roi_pool.output_size[0]
        representation_size = 1024
        out_channels = self.feature_extractor.out_channels
        self.box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        representation_size = 1024
        self.classifier = nn.Linear(representation_size, num_classes)

    def forward(self, batch):
        features = self.feature_extractor(batch["images"])

        proposals = batch["resized_boxes"]  # Boxes resized same as the image.
        # proposals = [proposal for proposal in proposals]  # RoI pooler expects a list of Tensors as format.

        image_shapes = batch["img_sizes"]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)

        x = box_features.flatten(start_dim=1)
        class_logits = self.classifier(x)

        padded_class_logits = add_padding(
            class_logits, batch["num_masks"], self.num_classes, self.pad_num, batch["boxes"].device
        )

        return {
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],  # Boxes belonging to original image.
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


if __name__ == "__main__":
    from data.datasets.mask_feature_dataset import ImageMaskData

    device = "cpu"

    dataset = ImageMaskData(
        "mask_features/all",
        "../datasets/coco/annotations/instances_val2017.json",
        "../datasets/coco/val2017",
        "cpu",
        train=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageMaskData.collate_fn)

    # model = LinearClassifier(3, 100, 80)
    model = SAMRPN(80, "checkpoints/moco_v2_800ep_pretrain.pth.tar")
    model.to(device)

    for batch in dataloader:
        output = model(batch)
        print("AAAA", output["pred_logits"].shape)
