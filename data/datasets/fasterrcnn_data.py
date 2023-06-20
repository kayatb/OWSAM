from utils.misc import filter_empty_imgs, box_xywh_to_xyxy

import torch
from torchvision.transforms.functional import to_tensor
import os
from pycocotools.coco import COCO
from lvis import LVIS
from PIL import Image


class ImageData(torch.utils.data.Dataset):
    """Load the pre-extracted masks and their features into a torch Dataset."""

    def __init__(self, feature_dir, ann_file, img_dir, device, offset=0):
        """Load the masks and their features from `dir`."""
        self.feature_dir = feature_dir
        self.img_dir = img_dir

        self.device = device
        # self.num_masks = num_masks

        # Create a mapping from category ID to category name.
        self.cat_id_to_name = {}

        if "coco" in ann_file:
            self.target_mode = "coco"
            self.coco = COCO(ann_file)
            self.img_ids = filter_empty_imgs(self.coco.getImgIds(), dataset="coco")
            self.get_targets = self.get_coco_targets

            for category in self.coco.loadCats(self.coco.getCatIds()):
                self.cat_id_to_name[category["id"]] = category["name"]

        elif "lvis" in ann_file:
            self.target_mode = "lvis"
            self.lvis = LVIS(ann_file)
            self.img_ids = filter_empty_imgs(self.lvis.get_img_ids(), dataset="lvis")
            self.get_targets = self.get_lvis_targets

            for category in self.lvis.load_cats(self.lvis.get_cat_ids()):
                self.cat_id_to_name[category["id"]] = category["name"]
        else:
            raise ValueError(f"Can't infer target mode from annotation file location `{ann_file}`.")

        # Class ids are non-continuous. Map them to a continuous range and vice versa.
        self.cat_id_to_continuous = {}
        self.continuous_to_cat_id = {}
        # Start the classes at 1. ID 0 is reserved for no-object class.
        # Don't do this in discovery phase, since the bg class is removed.
        # offset = 0 if discovery else 1
        for i, id in enumerate(self.cat_id_to_name.keys()):
            self.cat_id_to_continuous[id] = i + offset
            self.continuous_to_cat_id[i + offset] = id

    def __getitem__(self, idx):
        """Returns the masks, boxes, mask_features, iou_scores, the image id (i.e. filename),
        and the targets (class labels and segmentation masks) from the COCO dataset."""
        if self.target_mode == "coco":
            img_dict = self.coco.imgs[self.img_ids[idx]]
        else:
            img_dict = self.lvis.imgs[self.img_ids[idx]]

        img_file = self.get_file_name(self.img_dir, img_dict)
        with Image.open(img_file) as img:
            img = img.convert("RGB")
        img = to_tensor(img)  # Convert PIL image to Tensor in range [0.0, 1.0]

        file_path = os.path.join(self.feature_dir, f"{self.img_ids[idx]}.pt")
        # mask_data is a list of dicts (one dict per predicted mask in the image), where each dict has the following
        # keys: 'area', 'bbox', 'predicted_iou', 'stability_score', 'mask_feature'
        mask_data = torch.load(file_path, map_location=self.device)

        targets = self.get_targets(self.img_ids[idx])

        # boxes = torch.zeros((self.num_masks, 4))
        boxes = []
        iou_scores = []
        # iou_scores = -torch.ones((self.num_masks))

        for mask in mask_data:
            if mask["bbox"][2] == 0 or mask["bbox"][3] == 0:  # Skip boxes with zero width or height.
                continue
            # boxes[i] = box_xywh_to_xyxy(torch.as_tensor(mask["bbox"]))
            boxes.append(box_xywh_to_xyxy(torch.as_tensor(mask["bbox"])))
            iou_scores.append(mask["predicted_iou"])
            # iou_scores[i] = mask["predicted_iou"]

        boxes = torch.stack(boxes)
        iou_scores = torch.as_tensor(iou_scores)

        return {
            "image": img,
            "sam_boxes": boxes,
            "iou_scores": iou_scores,
            "img_id": self.img_ids[idx],
            "targets": targets,
        }

    def __len__(self):
        return len(self.img_ids)

    def get_coco_targets(self, img_id):
        """Get the COCO annotations belonging to the image.
        Convert the annotations to the format expected by the criterion."""
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        assert len(ann_ids) > 0, f"No annotations found for image `{img_id}`. Check the annotation file."
        anns = self.coco.loadAnns(ann_ids)

        targets = {}
        targets["labels"] = torch.as_tensor(
            [self.cat_id_to_continuous[ann["category_id"]] for ann in anns], dtype=torch.long
        )
        # NOTE: pre-trained Faster R-CNN does not use continuous IDs, but original COCO IDs.
        # targets["labels"] = torch.as_tensor([ann["category_id"] for ann in anns], dtype=torch.long)
        # Filter out target boxes with height or width 0.0.
        targets["boxes"] = torch.as_tensor(
            [ann["bbox"] for ann in anns if ann["bbox"][2] > 0.0 and ann["bbox"][3] > 0.0]
        )

        return targets

    def get_lvis_targets(self, img_id):
        """Get the LVIS annotations belonging to the image.
        Convert the annotaitons to the format expected by the criterion."""
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        assert len(ann_ids) > 0, f"No annotations found for image `{img_id}`. Check the annotation file."
        anns = self.lvis.load_anns(ann_ids)

        targets = {}
        targets["labels"] = torch.as_tensor(
            [self.cat_id_to_continuous[ann["category_id"]] for ann in anns], dtype=torch.long
        )
        targets["boxes"] = torch.as_tensor([ann["bbox"] for ann in anns])

        return targets

    # Taken from Detectron2:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis.py#L120
    def get_file_name(self, img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root, split_folder, file_name)

    @staticmethod
    def collate_fn(data):
        images = [d["image"] for d in data]
        sam_boxes = [d["sam_boxes"] for d in data]
        iou_scores = [d["iou_scores"] for d in data]
        # boxes = torch.stack([d["boxes"] for d in data])
        # iou_scores = torch.stack([d["iou_scores"] for d in data])
        img_ids = [d["img_id"] for d in data]
        targets = [d["targets"] for d in data]

        return {
            "images": images,
            "sam_boxes": sam_boxes,
            "iou_scores": iou_scores,
            "img_ids": img_ids,
            "targets": targets,
        }


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = ImageData(
        "mask_features/", "../datasets/coco/annotations/coco_half_train.json", "../datasets/coco", "cpu"
    )
    dataset.img_ids = [200365]

    print(dataset[0]["targets"]["boxes"])
