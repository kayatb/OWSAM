"""
Calculate the IoU overlap within the set of SAM boxes generated for a single image.
Make a histogram of the frequencies.
"""

import configs.fully_supervised.main as config

from data.datasets.fasterrcnn_data import ImageData
from utils.box_ops import box_iou
from utils.misc import box_xywh_to_xyxy, box_xyxy_to_xywh

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import boxes as box_ops
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="Calculate SAM topline mAP")
    # parser.add_argument("-m", "--mode", required=True, choices=["coco", "lvis"], help="COCO or LVIS evaluation mode")
    parser.add_argument("-a", "--ann-file", required=True, help="Annotation file location of the dataset")
    # parser.add_argument("-k", "--top-k", default=-1, type=int, help="Number of boxes to use as proposals")
    # parser.add_argument("--nms", default=1.01, type=float, help="NMS threshold to apply.")
    args = parser.parse_args()

    return args


def plot_iou_histogram(histogram):
    num_buckets = len(histogram)
    bucket_edges = np.arange(num_buckets + 1)

    # Normalize the histogram to percentages
    total = sum(histogram)
    percentages = [(count / total) for count in histogram]

    plt.bar(bucket_edges[:-1], percentages, align="edge", width=0.9)
    plt.xlabel("IoU Overlap")
    plt.ylabel("Frequency")
    plt.title("IoU Overlap Frequency Within SAM Boxes")
    plt.xticks(bucket_edges[:-1])
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    dataset = ImageData(config.masks_dir, args.ann_file, config.img_dir, config.device)
    # dataset.img_ids = dataset.img_ids[:4000]
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Has to be 1 to avoid padding.
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    for i, batch in enumerate(tqdm(dataloader)):
        assert (
            len(batch["sam_boxes"]) == 1
        ), f"Batch size has to be 1 to avoid padding. Current batch size is {batch['boxes'].shape[0]}."

        pred_boxes = box_xywh_to_xyxy(batch["sam_boxes"][0])

        # Calculate the IoU overlaps between the boxes.
        ious, _ = box_iou(pred_boxes, pred_boxes)
        # ious now is symmetrical, take only one half of it
        # (without the diagonal, since that is IoU between the same boxes, so always 1)/
        indices = torch.triu_indices(ious.shape[0], ious.shape[0], offset=1)
        ious = ious[indices[0], indices[1]].flatten()

        histogram = torch.histc(ious, bins=10, min=0.0, max=1.0).numpy()
        plot_iou_histogram(histogram)
