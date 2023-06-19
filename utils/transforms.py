"""
Transforms that work on an image and its corresponding bounding boxes.
Partially copied and adapted from: https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
"""
from utils.misc import resize_bboxes

from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T, InterpolationMode
import torch
from PIL import ImageFilter
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ColorJitter(T.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue, prob=1.0):
        super().__init__(brightness, contrast, saturation, hue)
        self.prob = prob

    def forward(self, image, boxes=None):
        if torch.rand(1) < self.prob:
            image = super().forward(image)
        return image, boxes


class GrayScale(T.Grayscale):
    def __init__(self, num_output_channels=1, prob=1.0):
        super().__init__(num_output_channels=num_output_channels)
        self.prob = prob

    def forward(self, image, boxes=None):
        if torch.rand(1) < self.prob:
            image = super().forward(image)
        return image, boxes


class GaussianBlur(torch.nn.Module):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0], prob=1.0):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image, boxes=None):
        if torch.rand(1) < self.prob:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image, boxes


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


class RandomShortestSize(torch.nn.Module):
    def __init__(
        self,
        min_size,
        max_size,
        interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(self, image, boxes=None):
        _, orig_height, orig_width = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if boxes is not None:
            boxes[:, 0::2] *= new_width / orig_width
            boxes[:, 1::2] *= new_height / orig_height
            # if "masks" in targets:
            #     target["masks"] = F.resize(
            #         target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
            #     )

        return image, boxes
