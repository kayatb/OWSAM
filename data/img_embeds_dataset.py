import utils.coco_ids_without_anns as empty_ids

from segment_anything.utils.amg import build_all_layer_point_grids
from segment_anything.utils.transforms import ResizeLongestSide

import torch
import os
from pycocotools.coco import COCO
import numpy as np


class ImageEmbeds(torch.utils.data.Dataset):
    """Load the pre-extracted image embeddings into a torch Dataset."""

    def __init__(self, dir, ann_file, device, points_per_side=32):
        """Load the image embeddings from `dir`."""
        self.dir = dir
        self.files = ImageEmbeds.filter_empty_imgs(os.listdir(dir))
        self.coco = COCO(ann_file)
        self.device = device

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

        self.point_grids = build_all_layer_point_grids(points_per_side, 0, 1)
        self.transform = ResizeLongestSide(1024)

    def __getitem__(self, idx):
        """Returns the image embedding (256 x 64 x 64), the original image size (W x H), the image file name,
        and a grid of point coordinates and labels to use as prompts for SAM."""
        file_path = os.path.join(self.dir, self.files[idx])
        img_data = torch.load(file_path, map_location=self.device)

        img_id = int(os.path.splitext(self.files[idx])[0])
        targets = self.get_coco_targets(img_id)

        point_coords, point_labels = self.make_points(img_data["orig_size"])

        return {
            "embed": img_data["embed"],
            "original_size": img_data["orig_size"],
            "targets": targets,
            "img_id": img_id,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    def __len__(self):
        return len(self.files)

    def get_coco_targets(self, img_id):
        """Get the COCO annotations belonging to the image embedding.
        Convert the annotation to the format expected by the criterion."""
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        targets = {}
        targets["labels"] = torch.as_tensor(
            [self.cat_id_to_continuous[ann["category_id"]] for ann in anns], dtype=torch.long
        )
        targets["boxes"] = torch.as_tensor([ann["bbox"] for ann in anns])

        return targets

    @staticmethod
    def filter_empty_imgs(files):
        """Filter out image IDs for images that only contain the background class and
        thus have no annotations."""
        filtered_files = []
        for file in files:
            id = int(os.path.splitext(file)[0])
            if id in empty_ids.train_ids or id in empty_ids.val_ids:
                continue
            filtered_files.append(file)
        return filtered_files

    def make_points(self, orig_size):
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        transformed_points = self.transform.apply_coords(points_for_image, orig_size)
        in_points = torch.as_tensor(transformed_points, device=self.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

        return in_points, in_labels

    @staticmethod
    def collate_fn(data):
        embeds = torch.stack([d["embed"] for d in data])
        original_sizes = [d["original_size"] for d in data]
        img_ids = [d["img_id"] for d in data]
        targets = [d["targets"] for d in data]
        point_coords = torch.stack([d["point_coords"] for d in data])
        point_labels = torch.stack([d["point_labels"] for d in data])

        return {
            "embed": embeds,
            "original_size": original_sizes,
            "img_id": img_ids,
            "targets": targets,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }


if __name__ == "__main__":
    dataset = ImageEmbeds("img_embeds", "../datasets/coco/annotations/instances_val2017.json", "cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)
    # print(dataset.cat_id_to_continuous)
    # print(dataset.continuous_to_cat_id)
    # print(dataset.cat_id_to_name)
    for batch in dataloader:
        print(batch["point_coords"].shape)
        print(batch["point_labels"].shape)
        # print(batch["original_size"])
        # print(batch["targets"])
