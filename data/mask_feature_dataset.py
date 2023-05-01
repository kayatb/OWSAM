from utils.misc import filter_empty_imgs

import torch
import os
import gzip
import io
from pycocotools.coco import COCO


class MaskData(torch.utils.data.Dataset):
    """Load the pre-extracted masks and their features into a torch Dataset."""

    def __init__(self, dir, ann_file, device, pad_num=500):
        """Load the masks and their features from `dir`."""
        self.dir = dir
        self.files = filter_empty_imgs(os.listdir(dir))
        self.coco = COCO(ann_file)
        self.device = device
        self.pad_num = pad_num

        # Create a mapping from category ID to category name.
        self.cat_id_to_name = {}
        for category in self.coco.loadCats(self.coco.getCatIds()):
            self.cat_id_to_name[category["id"]] = category["name"]

        # COCO class ids are non-continuous. Map them to a continuous range and vice versa.
        self.cat_id_to_continuous = {}
        self.continuous_to_cat_id = {}
        for i, id in enumerate(self.cat_id_to_name.keys()):
            self.cat_id_to_continuous[id] = i + 1  # Start the classes at 1, to keep 0 as the no-object class.
            self.continuous_to_cat_id[i + 1] = id

    def __getitem__(self, idx):
        """Returns the masks, boxes, mask_features, iou_scores, the image id (i.e. filename),
        and the targets (class labels and segmentation masks) from the COCO dataset."""
        file_path = os.path.join(self.dir, self.files[idx])
        mask_data = self.get_mask_data(file_path)

        img_id = int(os.path.splitext(self.files[idx])[0])
        targets = self.get_coco_targets(img_id)

        # Pad the output to a uniform number for batched processing.
        # The number of masks outputted by SAM is not constant, so pad with
        # empty values.
        boxes = torch.zeros((self.pad_num, 4))
        mask_features = torch.zeros((self.pad_num, 256))
        iou_scores = -torch.ones((self.pad_num))

        for i, mask in enumerate(mask_data):
            boxes[i] = torch.as_tensor(mask["bbox"])
            mask_features[i] = torch.as_tensor(mask["mask_feature"])
            iou_scores[i] = mask["predicted_iou"]

        return {
            "masks": [mask["segmentation"] for mask in mask_data],
            "boxes": boxes,
            "mask_features": mask_features,
            "iou_scores": iou_scores,
            "num_masks": i + 1,  # The number of actual masks (i.e. without padding)
            "img_id": img_id,
            "targets": targets,
        }

    def __len__(self):
        return len(self.files)

    def get_mask_data(self, path):
        """The pre-extracted masks are saved as a single object in a tar.gz file."""
        with gzip.open(path, "rb") as fp:
            content = fp.read()
        mask_data = torch.load(io.BytesIO(content))
        # mask_data is a list of dicts (one dict per predicted mask in the image), where each dict has the following
        # keys: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'mask_feature'
        return mask_data

    def get_coco_targets(self, img_id):
        """Get the COCO annotations belonging to the image embedding.
        Convert the annotation to the format expected by the criterion."""
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        assert len(ann_ids) > 0, f"No annotations found for image `{img_id}`. Check the annotation file."
        anns = self.coco.loadAnns(ann_ids)

        targets = {}
        targets["labels"] = torch.as_tensor(
            [self.cat_id_to_continuous[ann["category_id"]] for ann in anns], dtype=torch.long
        )
        targets["boxes"] = torch.as_tensor([ann["bbox"] for ann in anns])

        return targets

    @staticmethod
    def collate_fn(data):
        masks = [d["masks"] for d in data]
        boxes = torch.stack([d["boxes"] for d in data])
        mask_features = torch.stack([d["mask_features"] for d in data])
        iou_scores = torch.stack([d["iou_scores"] for d in data])
        num_masks = [d["num_masks"] for d in data]
        img_ids = [d["img_id"] for d in data]
        targets = [d["targets"] for d in data]

        return {
            "masks": masks,
            "boxes": boxes,
            "mask_features": mask_features,
            "iou_scores": iou_scores,
            "num_masks": num_masks,
            "img_ids": img_ids,
            "targets": targets,
        }


if __name__ == "__main__":
    dataset = MaskData("mask_features", "../datasets/coco/annotations/instances_val2017.json", "cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=MaskData.collate_fn)

    for batch in dataloader:
        print(len(batch["masks"]))
        print(batch["mask_features"].shape)
        print(batch["boxes"].shape)
        print(batch["iou_scores"].shape)
        print(batch["num_masks"])
        print(batch["img_ids"])
        print(batch["targets"])
