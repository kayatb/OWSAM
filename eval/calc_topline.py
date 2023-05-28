"""
Calculate the maximum mAP we can achieve with the extracted SAM masks.
Match the masks predicted by SAM with the COCO ground truth masks based on IoU and
then assign the matched mask the ground truth labels, i.e. we assume a perfect
classifier. Now calculate the mAP with these predictions.
"""

import configs.fully_supervised.main as config
from data.datasets.mask_feature_dataset import MaskData
from utils.box_ops import box_iou
from utils.misc import box_xywh_to_xyxy, box_xyxy_to_xywh
from eval.coco_eval import CocoEvaluator
from eval.lvis_eval import LvisEvaluator

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(prog="Calculate SAM topline mAP")
    parser.add_argument("-m", "--mode", required=True, choices=["coco", "lvis"], help="COCO or LVIS evaluation mode")
    parser.add_argument("-a", "--ann-file", required=True, help="Annotation file location of the dataset")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    dataset = MaskData(config.masks_dir, args.ann_file, config.device, pad_num=config.pad_num)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Has to be 1 to avoid padding.
        shuffle=False,
        collate_fn=MaskData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    if args.mode == "coco":
        evaluator = CocoEvaluator(args.ann_file, ["bbox"])
    else:
        evaluator = LvisEvaluator(args.ann_file, ["bbox"])

    for i, batch in enumerate(tqdm(dataloader)):
        assert (
            batch["boxes"].shape[0] == 1
        ), f"Batch size has to be 1 to avoid padding. Current batch size is {batch['boxes'].shape[0]}."

        pred_boxes = box_xywh_to_xyxy(batch["boxes"][0])  # Remove padded boxes
        pred_scores = batch["iou_scores"][0]

        # Sort the boxes according to their scores (necessary for handling duplicate IoU matches).
        # NOTE: might not be necessary, boxes and their respective score seem to be already sorted...
        boxes_ind = torch.argsort(pred_scores, descending=True)
        sorted_pred_boxes = torch.empty(pred_boxes.shape)
        for i, idx in enumerate(boxes_ind):
            sorted_pred_boxes[i] = pred_boxes[idx]

        gt_boxes = box_xywh_to_xyxy(batch["targets"][0]["boxes"])
        gt_labels = batch["targets"][0]["labels"]

        # For each GT box, calculate the IoU with all predicted boxes.
        ious, _ = box_iou(sorted_pred_boxes, gt_boxes)

        # Take the predicted box with the highest IoU and assign that the corresponding GT label
        best_idx = []
        for i in range(min(len(gt_boxes), len(pred_boxes))):
            ious_ind = torch.argsort(ious[:, i], descending=True)
            j = 0
            # Ensure no duplicate assignments.
            while ious_ind[j] in best_idx:
                j += 1
            best_idx.append(ious_ind[j].item())

        assert len(best_idx) == len(torch.unique(torch.as_tensor(best_idx))), "Duplicate best boxes!"

        # All unassigned boxes are filtered out.
        best_boxes = sorted_pred_boxes[best_idx]  # Best labels is now equal to gt_labels

        # To evaluator format.
        results = {}
        results[batch["img_ids"][0]] = {
            "boxes": box_xyxy_to_xywh(best_boxes),  # Original COCO/LVIS box format.
            "labels": gt_labels.cpu().apply_(dataset.continuous_to_cat_id.get),  # Original dataset cat IDs
            "scores": pred_scores[best_idx],
        }

        evaluator.update(results)

    if args.mode == "coco":  # Only the COCO evaluator requires this
        evaluator.synchronize_between_processes()
        evaluator.accumulate()

    evaluator.summarize()
