from utils.misc import add_padding

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


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


if __name__ == "__main__":
    from data.datasets.mask_feature_dataset import CropMaskData

    device = "cpu"

    dataset = CropMaskData(
        "mask_features/val_all",
        "../datasets/coco/annotations/instances_val2017.json",
        "../datasets/coco/val2017",
        "cpu",
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=CropMaskData.collate_fn)

    # model = LinearClassifier(3, 100, 80)
    model = ResNetClassifier(80)
    model.to(device)

    for batch in dataloader:
        output = model(batch)

        print(len(output["pred_boxes"]))
        print(output["pred_boxes"].shape)

        print(output["pred_logits"].shape)
