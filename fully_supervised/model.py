import torch
import torch.nn as nn

# import torch.nn.functional as F


class FullySupervisedClassifier(nn.Module):
    def __init__(self, sam_generator, num_layers, hidden_dim, num_classes, input_dim=256):
        super().__init__()

        self.num_classes = num_classes + 1  # Account for the backrground/no-object class.

        self.sam_generator = sam_generator

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    # TODO: batch-ify this
    def forward(self, batch):
        masks, boxes, mask_features = self.get_mask_features(batch)

        class_logits = []
        for feature in mask_features:
            x = self.layers(feature)
            x = self.classifier(x)
            class_logits.append(x)

        return {
            "masks": masks,
            "pred_logits": class_logits,
            "pred_boxes": boxes,
        }  # TODO: could also return mask_features here, but don't think it's necessary

    def get_mask_features(self, batch):
        masks = []
        boxes = []
        mask_features = []

        # TODO: batch-ify this.
        for i in range(batch["embed"].shape[0]):
            batch_masks = []
            batch_bbox = []
            batch_mask_features = []
            with torch.no_grad():
                sam_output = self.sam_generator.generate(batch["embed"][i].unsqueeze(0), batch["original_size"][i])

            for mask in sam_output:
                batch_masks.append(torch.as_tensor(mask["segmentation"]))
                batch_bbox.append(torch.as_tensor(mask["bbox"]))
                batch_mask_features.append(torch.as_tensor(mask["mask_feature"]))

            masks.append(torch.stack(batch_masks))
            boxes.append(torch.stack(batch_bbox))
            mask_features.append(torch.stack(batch_mask_features))

        return masks, boxes, mask_features


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
        for box in output["pred_boxes"]:
            print(box.shape)
    # for a in output:
    #     print(a.shape)
    # print(model(batch).shape)
