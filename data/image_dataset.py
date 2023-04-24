"""
Simple dataset that loads images in a directory and preprocesses them to be fed into SAM.
This dataset is used extract batched image embeddings from the SAM image encoder.
"""
from segment_anything.utils.transforms import ResizeLongestSide

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = [file for file in os.listdir(dir)]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.files[idx])).convert("RGB")
        input_img = self.preprocess(np.array(img))
        return {"img": input_img, "fname": os.path.splitext(self.files[idx])[0]}

    def __len__(self):
        return len(self.files)

    # Taken from SAM example notebooks.
    def preprocess(self, img):
        image_size = 1024
        transform = ResizeLongestSide(image_size)

        input_image = transform.apply_image(img)
        input_image_torch = torch.as_tensor(input_image, device="cpu")
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Normalize image
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        x = (input_image_torch - pixel_mean) / pixel_std

        # Pad to a square shape
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        x = x.squeeze().numpy()

        return x


if __name__ == "__main__":
    dataset = ImageDataset("../datasets/coco/train2017")
    # print(dataset[0].shape)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
    for batch in dataloader:
        print(batch)
        break
