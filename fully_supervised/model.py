import torch
import torch.nn as nn


class FullySupervisedClassifier(nn.Module):
    """A simple classification head on top of the hidden mask features extracted from SAM to classify the masks."""

    def __init__(self, sam_generator, num_layers, hidden_dim, num_classes, pad_num=500, input_dim=256):
        super().__init__()

        self.num_classes = num_classes + 1  # +1 for the backrground/no-object class.
        self.pad_num = pad_num
        self.input_dim = input_dim

        self.sam_generator = sam_generator

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, batch):
        # masks, boxes, mask_features, iou_scores = self.get_mask_features(batch)
        batch_size = batch["boxes"].shape[0]
        num_actual_masks = batch["num_masks"]  # Number of actual (non-padded) predicted masks per image.

        x = self.layers(batch["mask_features"])
        class_logits = self.classifier(x)

        # Change the logits for the padded features with extremely low values.
        # Necessary for batched processing since the amount of found masks is not constant across images.
        for i in range(batch_size):  # TODO: can you do this without the for-loop?
            pad_logits = torch.ones(self.pad_num - num_actual_masks[i], self.num_classes) * -1000
            class_logits[i, num_actual_masks[i] :] = pad_logits

        return {
            "masks": batch["masks"],
            "pred_logits": class_logits,
            "pred_boxes": batch["boxes"],
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


if __name__ == "__main__":
    from segment_anything import sam_model_registry
    from data.mask_feature_dataset import MaskData
    from modelling.sam_mask_generator import OWSamMaskGenerator

    device = "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    dataset = MaskData("mask_features", "../datasets/coco/annotations/instances_val2017.json", "cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=MaskData.collate_fn)

    mask_generator = OWSamMaskGenerator(sam)

    model = FullySupervisedClassifier(mask_generator, 3, 100, 80)
    model.to(device)

    for batch in dataloader:
        output = model(batch)

        print(len(output["pred_boxes"]))
        print(output["pred_boxes"].shape)

        print(output["pred_logits"].shape)
