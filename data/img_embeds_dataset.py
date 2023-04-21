import torch
import os


class ImageEmbeds(torch.utils.data.Dataset):
    """Load the pre-extracted image embeddings into a torch Dataset."""

    def __init__(self, dir, device):
        """Load the image embeddings from `dir`."""
        self.dir = dir
        self.files = os.listdir(dir)
        self.device = device

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.files[idx])
        img_embed = torch.load(file_path, map_location=self.device)

        return {"embed": img_embed, "img_name": os.path.splitext(self.files[idx])[0]}

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    dataset = ImageEmbeds("img_embeds", "cpu")
    print(len(dataset))
    print(dataset[0])
    print(dataset[0]["embed"].shape)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        print(batch["embed"].shape)
        print(batch["img_name"])
