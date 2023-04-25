from segment_anything.utils.amg import build_all_layer_point_grids
from segment_anything.utils.transforms import ResizeLongestSide

import torch
import os

# import numpy as np


class ImageEmbeds(torch.utils.data.Dataset):
    """Load the pre-extracted image embeddings into a torch Dataset."""

    def __init__(self, dir, device, points_per_side=32):
        """Load the image embeddings from `dir`."""
        self.dir = dir
        self.files = os.listdir(dir)
        self.device = device

        self.point_grids = build_all_layer_point_grids(points_per_side, 0, 1)
        self.transform = ResizeLongestSide(1024)

    def __getitem__(self, idx):
        """Returns the image embedding (256 x 64 x 64), the original image size (W x H), the image file name,
        and a grid of point coordinates and labels to use as prompts for SAM."""
        file_path = os.path.join(self.dir, self.files[idx])
        img_data = torch.load(file_path, map_location=self.device)

        # point_coords, point_labels = self.make_points(img_data["orig_size"])

        return {
            "embed": img_data["embed"],
            "original_size": img_data["orig_size"],
            # "point_coords": point_coords,
            # "point_labels": point_labels,
            "file_name": os.path.splitext(self.files[idx])[0],
        }

    # def make_points(self, orig_size):
    #     points_scale = np.array(orig_size)[None, ::-1]
    #     points_for_image = self.point_grids[0] * points_scale

    #     transformed_points = self.transform.apply_coords(points_for_image, orig_size)
    #     in_points = torch.as_tensor(transformed_points, device=self.device)
    #     in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

    #     return in_points, in_labels

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(data):
        embeds = torch.stack([d["embed"] for d in data])
        original_sizes = [d["original_size"] for d in data]
        file_names = [d["file_name"] for d in data]

        return {"embed": embeds, "original_size": original_sizes, "file_name": file_names}


if __name__ == "__main__":
    dataset = ImageEmbeds("img_embeds", "cpu")
    # print(len(dataset))
    # print(dataset[0]["point_coords"].shape)
    # print(dataset[0]["point_labels"].shape)
    # print(dataset[0]["embed"].shape)
    # print(dataset[0]["original_size"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    for batch in dataloader:
        print(batch["embed"].shape)
        print(batch["original_size"])
        # print(batch["point_coords"].shape)
        # print(batch["point_labels"].shape)
        print(batch["file_name"])
