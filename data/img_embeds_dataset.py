import torch
import os
from pycocotools.coco import COCO


class ImageEmbeds(torch.utils.data.Dataset):
    """Load the pre-extracted image embeddings into a torch Dataset."""

    def __init__(self, dir, ann_file, device):
        """Load the image embeddings from `dir`."""
        self.dir = dir
        self.files = os.listdir(dir)
        self.coco = COCO(ann_file)
        self.device = device

    def __getitem__(self, idx):
        """Returns the image embedding (256 x 64 x 64), the original image size (W x H), the image file name,
        and a grid of point coordinates and labels to use as prompts for SAM."""
        file_path = os.path.join(self.dir, self.files[idx])
        img_data = torch.load(file_path, map_location=self.device)

        img_id = int(os.path.splitext(self.files[idx])[0])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        return {
            "embed": img_data["embed"],
            "original_size": img_data["orig_size"],
            "gt_anns": anns,
            "img_id": img_id,
        }

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(data):
        embeds = torch.stack([d["embed"] for d in data])
        original_sizes = [d["original_size"] for d in data]
        img_ids = [d["img_id"] for d in data]
        anns = [d["gt_anns"] for d in data]

        return {"embed": embeds, "original_size": original_sizes, "img_id": img_ids, "gt_anns": anns}


if __name__ == "__main__":
    dataset = ImageEmbeds("img_embeds", "../datasets/coco/annotations/instances_val2017.json", "cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    for batch in dataloader:
        print(batch["embed"].shape)
        print(batch["original_size"])
        print(len(batch["gt_anns"]))
