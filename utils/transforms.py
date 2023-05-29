"""
Transforms that work on an image and its corresponding bounding boxes.
Copied and adapted from: https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
"""
from utils.misc import resize_bboxes

from torchvision.transforms import functional as F
import torchvision.transforms as T
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, boxes=None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if boxes is not None:
                _, _, width = F.get_dimensions(image)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                # if "masks" in target:
                #     target["masks"] = target["masks"].flip(-1)

        return image, boxes


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image, boxes=None):
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if boxes is not None:
                _, height, _ = F.get_dimensions(image)
                boxes[:, [0, 1]] = height - boxes[:, [1, 0]]

        return image, boxes


# class PILToTensor(torch.nn.Module):
#     def forward(self, image, target):
#         image = F.pil_to_tensor(image)
#         return image, target


class ToTensor(T.ToTensor):
    def __call__(self, image, boxes):
        return F.to_tensor(image), boxes


class Resize(T.Resize):
    def forward(self, image, boxes=None):
        orig_size = image.size
        image = F.resize(image, self.size, self.interpolation, self.max_size, self.antialias)

        if boxes is not None:
            boxes = resize_bboxes(boxes, orig_size, self.size[0])

        return image, boxes


class Normalize(T.Normalize):
    def forward(self, image, boxes=None):
        return super().forward(image), boxes
