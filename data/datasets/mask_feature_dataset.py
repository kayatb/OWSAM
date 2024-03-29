from utils.misc import filter_empty_imgs, crop_bboxes_from_img, box_xywh_to_xyxy
import utils.transforms as T

import torch
import torchvision.transforms as TVT
import os
from pycocotools.coco import COCO
from lvis import LVIS
from PIL import Image


class MaskData(torch.utils.data.Dataset):
    """Load the pre-extracted masks and their features into a torch Dataset."""

    def __init__(self, dir, ann_file, device, pad_num=700):
        """Load the masks and their features from `dir`."""
        self.dir = dir

        self.device = device
        self.pad_num = pad_num

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
            self.cat_id_to_continuous[id] = i  # Start the classes at 0. The last index is reserved for no-object class.
            self.continuous_to_cat_id[i] = id

    def __getitem__(self, idx):
        """Returns the masks, boxes, mask_features, iou_scores, the image id (i.e. filename),
        and the targets (class labels and segmentation masks) from the COCO dataset."""
        file_path = os.path.join(self.dir, f"{self.img_ids[idx]}.pt")
        # mask_data is a list of dicts (one dict per predicted mask in the image), where each dict has the following
        # keys: 'area', 'bbox', 'predicted_iou', 'stability_score', 'mask_feature'
        mask_data = torch.load(file_path, map_location=self.device)

        targets = self.get_targets(self.img_ids[idx])

        # Pad the output to a uniform number for batched processing.
        # The number of masks outputted by SAM is not constant, so pad with
        # empty values.
        boxes = torch.zeros((self.pad_num, 4))
        mask_features = torch.zeros((len(mask_data), 256))
        iou_scores = -torch.ones((self.pad_num))

        num_masks = 0
        for i, mask in enumerate(mask_data):
            if mask["bbox"][2] == 0 or mask["bbox"][3] == 0:  # Skip boxes with zero width or height.
                continue
            boxes[num_masks] = torch.as_tensor(mask["bbox"])
            mask_features[num_masks] = torch.as_tensor(mask["mask_feature"])
            iou_scores[num_masks] = mask["predicted_iou"]
            num_masks += 1

        return {
            "boxes": boxes,
            "mask_features": mask_features,
            "iou_scores": iou_scores,
            "num_masks": num_masks,  # The number of actual masks (i.e. without padding)
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

    @staticmethod
    def collate_fn(data):
        boxes = torch.stack([d["boxes"] for d in data])
        mask_features = torch.cat([d["mask_features"] for d in data])
        iou_scores = torch.stack([d["iou_scores"] for d in data])
        num_masks = [d["num_masks"] for d in data]
        img_ids = [d["img_id"] for d in data]
        targets = [d["targets"] for d in data]

        return {
            "boxes": boxes,
            "mask_features": mask_features,
            "iou_scores": iou_scores,
            "num_masks": num_masks,
            "img_ids": img_ids,
            "targets": targets,
        }


class CropFeatureMaskData(MaskData):
    def __init__(self, mask_dir, ann_file, dino_dir, device, pad_num=700):
        super().__init__(mask_dir, ann_file, device, pad_num)

        self.dino_dir = dino_dir

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        dino_features = torch.load(
            os.path.join(self.dino_dir, f"{self.img_ids[idx]}.pt")
        )  # , map_location=self.device)
        data["crop_feature"] = dino_features

        return data

    @staticmethod
    def collate_fn(data):
        batch = MaskData.collate_fn(data)
        batch["crop_features"] = torch.cat([d["crop_feature"] for d in data])

        return batch


class ImageMaskData(MaskData):
    def __init__(self, mask_dir, ann_file, img_dir, device, train=False, pad_num=700):
        super().__init__(mask_dir, ann_file, device, pad_num)

        self.img_dir = img_dir
        self.train = train  # Whether this is the train set or not.

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # COCO image files have filename img_id prepended with 0's until length is 12.
        # img_file = f"{str(data['img_id']).rjust(12, '0')}.jpg"
        if self.target_mode == "coco":
            img_dict = self.coco.imgs[data["img_id"]]
        else:
            img_dict = self.lvis.imgs[data["img_id"]]

        img_file = self.get_file_name(self.img_dir, img_dict)
        with Image.open(img_file) as img:
            img = img.convert("RGB")

        data["img"] = img
        data["trans_boxes"] = box_xywh_to_xyxy(data["boxes"][: data["num_masks"]])  # Remove padding and convert format.

        return data

    # Taken from Detectron2:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis.py#L120
    def get_file_name(self, img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root, split_folder, file_name)

    # @staticmethod
    def collate_fn(self, data):
        batch = MaskData.collate_fn(data)
        # batch["images"] = torch.stack([d["img"] for d in data])
        # batch["img_sizes"] = [d["img_size"] for d in data]
        # batch["trans_boxes"] = [
        #     torch.tensor(d["trans_boxes"], dtype=torch.float) for d in data
        # ]  # Format expected by RoI pooler.

        if self.train:
            # Ensure all images in the batch are randomly resized to the same size.
            min_choices = [640, 672, 704, 736, 768, 800]
            min_size = min_choices[torch.randint(len(min_choices), (1,)).item()]
            transform = T.Compose(
                [
                    # T.RandomShortestSize(min_size=min_size, max_size=1333),
                    T.Resize((min_size, min_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            transform = T.Compose(
                [
                    # NOTE: resolution for pre-trained Faster R-CNN model evaluation.
                    # T.RandomShortestSize(min_size=800, max_size=1333),
                    T.Resize((800, 800)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        # Do the transform on all image-boxes pairs and batch them.
        images = []
        batch["trans_boxes"] = []
        for d in data:
            img = d["img"]
            boxes = d["trans_boxes"]

            img, boxes = transform(img, boxes)
            images.append(img)
            batch["trans_boxes"].append(boxes)

        batch["images"] = torch.stack(images)

        return batch


# TODO: clean up this utter mess.
class DiscoveryImageMaskData(ImageMaskData):
    def __init__(self, mask_dir, ann_file, img_dir, device, train=False, pad_num=700, num_views=2):
        super().__init__(mask_dir, ann_file, img_dir, device, train=train, pad_num=pad_num)
        self.num_views = num_views

    def collate_fn(self, data):
        batch = MaskData.collate_fn(data)

        if self.train:
            batch["images"] = []
            batch["trans_boxes"] = []
            min_choices = [640, 672, 704, 736, 768, 800]
            for _ in range(self.num_views):
                # Ensure all images in the batch are randomly resized to the same size.
                min_size = min_choices[torch.randint(len(min_choices), (1,)).item()]
                transform = T.Compose(
                    [
                        # T.RandomShortestSize(min_size=min_size, max_size=1333),
                        T.Resize((min_size, min_size)),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ColorJitter(0.8, 0.8, 0.8, 0.2, prob=0.8),
                        T.GrayScale(num_output_channels=3, prob=0.2),
                        T.GaussianBlur(sigma=[0.1, 2.0], prob=0.5),
                        T.ToTensor(),
                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )

                images = []
                trans_boxes = []
                for d in data:
                    img = d["img"]
                    boxes = d["trans_boxes"]

                    img, boxes = transform(img, boxes)
                    images.append(img)
                    trans_boxes.append(boxes)

                batch["images"].append(torch.stack(images))
                batch["trans_boxes"].append(trans_boxes)

        else:
            transform = T.Compose(
                [
                    # NOTE: resolution for pre-trained Faster R-CNN model evaluation.
                    # T.RandomShortestSize(min_size=800, max_size=1333),
                    T.Resize((800, 800)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

            # Do the transform on all image-boxes pairs and batch them.
            images = []
            batch["trans_boxes"] = []
            for d in data:
                img = d["img"]
                boxes = d["trans_boxes"]

                img, boxes = transform(img, boxes)
                images.append(img)
                batch["trans_boxes"].append(boxes)

            batch["images"] = torch.stack(images)

        return batch


class CropMaskData(MaskData):
    def __init__(self, box_dir, ann_file, img_dir, device, pad_num=700, resize=256, center_crop=224):
        super().__init__(box_dir, ann_file, device, pad_num)

        self.img_dir = img_dir
        self.resize = resize
        self.center_crop = center_crop

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        boxes = box_xywh_to_xyxy(data["boxes"][: data["num_masks"]])  # Convert format and remove padding.

        # COCO image files have filename img_id prepended with 0's until length is 12.
        img_file = f"{str(data['img_id']).rjust(12, '0')}.jpg"
        with Image.open(os.path.join(self.img_dir, img_file)) as img:
            img = img.convert("RGB")
            crops = crop_bboxes_from_img(img, boxes)

        for i, crop in enumerate(crops):
            crops[i] = self.preprocess(crop)

        data["crops"] = torch.stack(crops)

        return data

    def preprocess(self, img):
        transform = TVT.Compose(
            [
                TVT.Resize(self.resize),
                TVT.CenterCrop(self.center_crop),
                TVT.ToTensor(),
                TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img = transform(img)[:3]

        return img

    @staticmethod
    def collate_fn(data):
        batch = MaskData.collate_fn(data)
        batch["crops"] = torch.cat([d["crops"] for d in data])

        return batch


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
