import torch
import torch.nn as nn
import torch.nn.functional as F


class FullySupervisedClassifier(nn.Module):
    """A simple classification head on top of the hidden mask features extracted from SAM to classify the masks."""

    def __init__(self, num_layers, hidden_dim, num_classes, pad_num=500, input_dim=256):
        super().__init__()

        self.num_classes = num_classes + 1  # +1 for the backrground/no-object class.
        self.pad_num = pad_num
        self.input_dim = input_dim

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.classifier = nn.Linear(hidden_dim, self.num_classes + 1)  # +1 for the no-object class

    def forward(self, batch):
        # masks, boxes, mask_features, iou_scores = self.get_mask_features(batch)
        batch_size = batch["boxes"].shape[0]

        x = self.layers(batch["mask_features"])
        class_logits = self.classifier(x)

        # Split the class logits per image.
        class_logits = torch.split(class_logits, batch["num_masks"])
        padded_class_logits = torch.empty(
            batch_size, self.pad_num, self.num_classes + 1, device=batch["mask_features"].device
        )
        # Now batch the class logits to be shape [batch_size x pad_num x num_classes].
        # Pad each image's logits with extremely low values to make the shape uniform
        # across images.
        for i in range(batch_size):  # TODO: can you do this without a for-loop?
            padded_class_logits[i] = F.pad(
                class_logits[i], (0, 0, 0, self.pad_num - class_logits[i].shape[0]), mode="constant", value=-1000
            )

        return {
            "masks": batch["masks"],
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


if __name__ == "__main__":
    from data.mask_feature_dataset import MaskData

    device = "cpu"

    dataset = MaskData("mask_features", "../datasets/coco/annotations/instances_val2017.json", "cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=MaskData.collate_fn)

    model = FullySupervisedClassifier(3, 100, 80)
    model.to(device)

    for batch in dataloader:
        output = model(batch)

        print(len(output["pred_boxes"]))
        print(output["pred_boxes"].shape)

        print(output["pred_logits"].shape)
