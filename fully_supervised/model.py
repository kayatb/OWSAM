import torch
import torch.nn as nn

import torch.nn.functional as F


class FullySupervisedClassifier(nn.Module):
    def __init__(self, sam_generator, num_layers, hidden_dim, num_classes, pad_num=200, input_dim=256):
        super().__init__()

        self.num_classes = num_classes + 1  # Account for the backrground/no-object class.
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
        masks, boxes, mask_features, iou_scores = self.get_mask_features(batch)
        batch_size = batch["embed"].shape[0]
        num_actual_masks = [mask.shape[0] for mask in masks]  # Number of actual (non-padded) predicted masks per image.

        x = self.layers(mask_features)
        class_logits = self.classifier(x)

        # Change the logits for the padded features with extremely low values.
        # Necessary for batched processing since the amount of found masks is not constant across images.
        for i in range(batch_size):  # TODO: can you do this without the for-loop?
            pad_logits = torch.ones(self.pad_num - num_actual_masks[i], self.num_classes) * -1000
            class_logits[i, num_actual_masks[i] :] = pad_logits

        return {
            "masks": masks,
            "pred_logits": class_logits,
            "pred_boxes": boxes,
            "iou_scores": iou_scores,
        }

    @torch.no_grad()
    def get_mask_features(self, batch):
        batch_size = batch["embed"].shape[0]

        masks = []
        boxes = torch.zeros((batch_size, self.pad_num, 4))
        # mask_features = []
        mask_features = torch.zeros((batch_size, self.pad_num, self.input_dim))
        iou_scores = torch.zeros((batch_size, self.pad_num))

        # TODO: batch-ify this.
        for i in range(batch_size):
            # TODO: make this torch tensors from the get-go to avoid torch.stack calls
            batch_masks = []
            # batch_mask_features = []

            sam_output = self.sam_generator.generate(batch["embed"][i].unsqueeze(0), batch["original_size"][i])

            for j, mask in enumerate(sam_output):
                batch_masks.append(torch.as_tensor(mask["segmentation"]))
                boxes[i, j] = torch.as_tensor(mask["bbox"])
                mask_features[i, j] = torch.as_tensor(mask["mask_feature"])
                # batch_mask_features.append(torch.as_tensor(mask["mask_feature"]))
                iou_scores[i, j] = mask["predicted_iou"]

            masks.append(torch.stack(batch_masks))
            # mask_features.append(torch.stack(batch_mask_features))

        return masks, boxes, mask_features, iou_scores


if __name__ == "__main__":
    from segment_anything import sam_model_registry
    from data.img_embeds_dataset import ImageEmbeds
    from modelling.sam_mask_generator import OWSamMaskGenerator

    device = "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    dataset = ImageEmbeds("img_embeds", "../datasets/coco/annotations/instances_val2017.json", sam.device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    mask_generator = OWSamMaskGenerator(sam)

    model = FullySupervisedClassifier(mask_generator, 3, 100, 80)
    model.to(device)

    for batch in dataloader:
        output = model(batch)

        print(len(output["pred_boxes"]))
        print(output["pred_boxes"].shape)

        print(output["pred_logits"].shape)

    # for a in output:
    #     print(a.shape)
    # print(model(batch).shape)
