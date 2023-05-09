import configs.fully_supervised as config
from data.mask_feature_dataset import MaskData
from utils.box_ops import box_iou
from fully_supervised.coco_eval import CocoEvaluator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [x, y, (x + w), (y + h)]

    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    xmin, ymin, xmax, ymax = x.unbind(-1)
    b = [xmin, ymin, xmax - xmin, ymax - ymin]

    return torch.stack(b, dim=-1)


dataset = MaskData(config.masks_train, config.ann_train, config.device, pad_num=config.pad_num)
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

evaluator = CocoEvaluator(config.ann_train, ["bbox"])

for batch in tqdm(dataloader):
    assert (
        batch["boxes"].shape[0] == 1
    ), f"Batch size has to be 1 to avoid padding. Current batch size is {batch['boxes'].shape[0]}."

    pred_boxes = box_xywh_to_xyxy(batch["boxes"][0])  # Remove padded boxes
    pred_scores = batch["iou_scores"][0]

    # Sort the boxes according to their scores (necessary for handling duplicate IoU matches).
    # FIXME: might not be necessary, boxes and their respective score seem to be already sorted...
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
    for i in range(len(gt_boxes)):
        ious_ind = torch.argsort(ious[:, i], descending=True)
        j = 0
        # Ensure no duplicate assignments.
        while ious_ind[j] in best_idx:
            j += 1
        best_idx.append(ious_ind[j].item())

    assert len(best_idx) == len(torch.unique(torch.as_tensor(best_idx))), "Duplicate best boxes!"

    # All unassigned boxes are filtered out.
    best_boxes = sorted_pred_boxes[best_idx]  # Best labels is now equal to gt_labels

    # To COCO evaluator format.
    results = {}
    results[batch["img_ids"][0]] = {
        "boxes": box_xyxy_to_xywh(best_boxes),  # Original COCO box format.
        "labels": gt_labels.cpu().apply_(dataset.continuous_to_cat_id.get),  # Original COCO cat IDs
        "scores": pred_scores[best_idx],
    }

    evaluator.update(results)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
