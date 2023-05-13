from utils.misc import filter_empty_imgs, crop_bboxes_from_img, box_xywh_to_xyxy

import torch
import torchvision.transforms as T
import os
from pycocotools.coco import COCO
from PIL import Image


class MaskData(torch.utils.data.Dataset):
    """Load the pre-extracted masks and their features into a torch Dataset."""

    def __init__(self, dir, ann_file, device, pad_num=700):
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
            self.cat_id_to_continuous[id] = i  # Start the classes at 0. The last index is reserved for no-object class.
            self.continuous_to_cat_id[i] = id

    def __getitem__(self, idx):
        """Returns the masks, boxes, mask_features, iou_scores, the image id (i.e. filename),
        and the targets (class labels and segmentation masks) from the COCO dataset."""
        file_path = os.path.join(self.dir, self.files[idx])
        # mask_data is a list of dicts (one dict per predicted mask in the image), where each dict has the following
        # keys: 'area', 'bbox', 'predicted_iou', 'stability_score', 'mask_feature'
        mask_data = torch.load(file_path, map_location=self.device)

        img_id = int(os.path.splitext(self.files[idx])[0])
        targets = self.get_coco_targets(img_id)

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
            "img_id": img_id,
            "targets": targets,
        }

    def __len__(self):
        return len(self.files)

    # def get_mask_data(self, path):
    #     """The pre-extracted masks are saved as a single object in a tar.gz file."""
    #     with gzip.open(path, "rb") as fp:
    #         mask_data = torch.load(io.BytesIO(fp.read()), map_location=self.device)

    #     # mask_data is a list of dicts (one dict per predicted mask in the image), where each dict has the following
    #     # keys: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'mask_feature'
    #     return mask_data

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
        # masks = [d["masks"] for d in data]
        boxes = torch.stack([d["boxes"] for d in data])
        mask_features = torch.cat([d["mask_features"] for d in data])
        iou_scores = torch.stack([d["iou_scores"] for d in data])
        num_masks = [d["num_masks"] for d in data]
        img_ids = [d["img_id"] for d in data]
        targets = [d["targets"] for d in data]

        return {
            # "masks": masks,
            "boxes": boxes,
            "mask_features": mask_features,
            "iou_scores": iou_scores,
            "num_masks": num_masks,
            "img_ids": img_ids,
            "targets": targets,
        }


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
        with Image.open(os.path.join(self.img_dir, img_file)) as img:  # TODO: how to get filename of image file?
            img = img.convert("RGB")
            crops = crop_bboxes_from_img(img, boxes)

        for i, crop in enumerate(crops):
            crops[i] = self.preprocess(crop)

        data["crops"] = torch.stack(crops)

        return data

    def preprocess(self, img):
        transform = T.Compose(
            [
                T.Resize(self.resize),
                T.CenterCrop(self.center_crop),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img = transform(img)[:3]  # .unsqueeze(0)

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

    dataset = CropMaskData(
        "mask_features/train_all",
        "../datasets/coco/annotations/instances_train2017.json",
        "../datasets/coco/train2017",
        "cpu",
    )
    # print(dataset[0]["crops"].shape)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=CropMaskData.collate_fn)
    for batch in dataloader:
        print(batch.keys())
        for crop in batch["crops"]:
            print(crop.min())
            print(crop.max())
        break
