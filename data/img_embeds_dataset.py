from segment_anything.utils.amg import build_all_layer_point_grids

import torch
import os


class ImageEmbeds(torch.utils.data.Dataset):
    """Load the pre-extracted image embeddings into a torch Dataset."""

    def __init__(self, dir, device, points_per_side=32):
        """Load the image embeddings from `dir`."""
        self.dir = dir
        self.files = os.listdir(dir)
        self.device = device

        self.point_grids = build_all_layer_point_grids(points_per_side, 0, 1)

    def __getitem__(self, idx):
        """Returns the image embedding (256 x 64 x 64), the original image size (W x H), the image file name,
        and a grid of point coordinates and labels to use as prompts for SAM."""
        file_path = os.path.join(self.dir, self.files[idx])
        img_data = torch.load(file_path, map_location=self.device)

        return {
            "embed": img_data["embed"],
            "original_size": img_data["orig_size"],
            "file_name": os.path.splitext(self.files[idx])[0],
        }

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    for batch in dataloader:
        print(batch["embed"].shape)
        print(batch["original_size"])
        print(batch["file_name"])
