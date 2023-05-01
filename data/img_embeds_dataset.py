import utils.coco_ids_without_anns as empty_ids

import torch
import os


class ImageEmbeds(torch.utils.data.Dataset):
    """Load the pre-extracted image embeddings into a torch Dataset."""

    def __init__(self, dir, device):
        """Load the image embeddings from `dir`."""
        self.dir = dir
        self.files = ImageEmbeds.filter_empty_imgs(os.listdir(dir))
        self.device = device

    def __getitem__(self, idx):
        """Returns the image embedding (256 x 64 x 64), the original image size (W x H), the image file name,
        and the targets (class labels and segmentation masks) from the COCO dataset."""
        file_path = os.path.join(self.dir, self.files[idx])
        img_data = torch.load(file_path, map_location=self.device)

        img_id = int(os.path.splitext(self.files[idx])[0])

        return {
            "embed": img_data["embed"],
            "original_size": img_data["orig_size"],
            "img_id": img_id,
        }

    def __len__(self):
        return len(self.files)

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

    @staticmethod
    def collate_fn(data):
        embeds = torch.stack([d["embed"] for d in data])
        original_sizes = [d["original_size"] for d in data]
        img_ids = [d["img_id"] for d in data]

        return {"embed": embeds, "original_size": original_sizes, "img_id": img_ids}


if __name__ == "__main__":
    dataset = ImageEmbeds("img_embeds", "cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    for batch in dataloader:
        print(batch["embed"].shape)
        print(batch["original_size"])
