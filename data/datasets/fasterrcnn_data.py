from utils.misc import filter_empty_imgs, box_xywh_to_xyxy

import torch
from torchvision.transforms.functional import to_tensor
import os
from pycocotools.coco import COCO
from lvis import LVIS
from PIL import Image


class ImageData(torch.utils.data.Dataset):
    """Load the pre-extracted masks and their features into a torch Dataset."""

    def __init__(self, feature_dir, img_dir, ann_file, device):
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
        for i, id in enumerate(self.cat_id_to_name.keys()):
            self.cat_id_to_continuous[id] = i + 1  # Start the classes at 1. ID 0 is reserved for no-object class.
            self.continuous_to_cat_id[i + 1] = id

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
        targets["boxes"] = torch.as_tensor([ann["bbox"] for ann in anns])

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

    # dataset = MaskData("mask_features/train_all", "../datasets/coco/annotations/instances_train2017.json", "cpu")

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1, collate_fn=MaskData.collate_fn, num_workers=12, pin_memory=True, persistent_workers=True
    # )

    # # Calculate the maximum amount of masks detected by SAM.
    # # 666 for train set, 395 for val set.
    # max_dim = 0
    # for batch in tqdm(dataloader):
    #     max_dim = max(max_dim, batch["mask_features"].shape[0])
    # print(max_dim)

    # dataset = CropMaskData(
    #     "mask_features/train_all",
    #     "../datasets/coco/annotations/instances_train2017.json",
    #     "../datasets/coco/train2017",
    #     "cpu",
    # )
    # # print(dataset[0]["crops"].shape)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=CropMaskData.collate_fn)
    # for batch in dataloader:
    #     print(batch.keys())
    #     for crop in batch["crops"]:
    #         print(crop.min())
    #         print(crop.max())
    #     break

    # dataset = CropFeatureMaskData(
    #     "mask_features/all",
    #     # "../datasets/coco/annotations/instances_val2017.json",
    #     "../datasets/lvis/lvis_v1_train.json",
    #     "dino_features/all",
    #     "cpu",
    #     # lvis_ann_file="../datasets/lvis/lvis_v1_val.json",
    # )

    # # print(dataset[0]["lvis_targets"])

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=CropFeatureMaskData.collate_fn)
    # for batch in tqdm(dataloader):
    #     # print(batch.keys())
    #     # print(batch["crop_features"].shape)
    #     # break
    #     # print(batch["targets"])
    #     # break
    #     continue

    import matplotlib.pyplot as plt
    import cv2

    # torch.manual_seed(0)

    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White

    def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        cv2.rectangle(img, (x_min, y_min), (x_min, y_min), BOX_COLOR, -1)

        return img

    def visualize(image, bboxes):
        image = image.permute(1, 2, 0).numpy()
        img = image.copy()
        for bbox in bboxes:
            img = visualize_bbox(img, bbox)
        plt.figure(figsize=(12, 12))
        plt.axis("off")
        plt.imshow(img)
        plt.show()

    dataset = ImageMaskData(
        "mask_features/all",
        "../datasets/coco/annotations/instances_val2017.json",
        "../datasets/coco/val2017",
        "cpu",
        train=True,
    )

    data = dataset[0]
    visualize(data["img"], data["resized_boxes"][:3])

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=ImageMaskData.collate_fn)
    # for batch in dataloader:
    #     # print(batch["resized_boxes"][0][0])
    #     # print(batch["boxes"][0][0])
    #     break
    # print(torch.min(dataset[0]["img"]))
