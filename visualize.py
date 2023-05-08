import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def show_anns(anns):
    if len(anns) == 0:
        return

    for ann in anns:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        # sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        # for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.5)))
        plt.show()


def show_boxes(anns):
    if len(anns) == 0:
        return

    for ann in anns:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        # sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        # for ann in sorted_anns:
        m = ann["bbox"]
        rect = patches.Rectangle((m[0], m[1]), m[2], m[3], linewidth=1, edgecolor="r", facecolor="none")

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()


image = cv2.imread("../datasets/coco/val2017/000000459195.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# plt.axis("off")
# plt.show()

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
print(len(masks))

show_boxes(masks)
# plt.axis("off")
# plt.show()
