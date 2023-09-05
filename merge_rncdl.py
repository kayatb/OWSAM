from eval.coco_eval import CocoEvaluator
from utils.misc import box_xyxy_to_xywh, box_xywh_to_xyxy
from utils.box_ops import box_iou

import os
import torch
from tqdm import tqdm

label_map = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
    24: 27,
    25: 28,
    26: 31,
    27: 32,
    28: 33,
    29: 34,
    30: 35,
    31: 36,
    32: 37,
    33: 38,
    34: 39,
    35: 40,
    36: 41,
    37: 42,
    38: 43,
    39: 44,
    40: 46,
    41: 47,
    42: 48,
    43: 49,
    44: 50,
    45: 51,
    46: 52,
    47: 53,
    48: 54,
    49: 55,
    50: 56,
    51: 57,
    52: 58,
    53: 59,
    54: 60,
    55: 61,
    56: 62,
    57: 63,
    58: 64,
    59: 65,
    60: 67,
    61: 70,
    62: 72,
    63: 73,
    64: 74,
    65: 75,
    66: 76,
    67: 77,
    68: 78,
    69: 79,
    70: 80,
    71: 81,
    72: 82,
    73: 84,
    74: 85,
    75: 86,
    76: 87,
    77: 88,
    78: 89,
    79: 90,
}

evaluator = CocoEvaluator("../datasets/coco/annotations/instances_val2017.json", ["bbox"])

for rncdl_file in tqdm(os.listdir("/scratch-shared/ktburg/rncdl_preds")):
    img_id = int(rncdl_file[:-14])
    try:
        sam_data = torch.load(f"/scratch-shared/ktburg/sam_mask_features/all_32/{img_id}.pt")
    except:
        continue

    rncdl_pred = torch.load(
        os.path.join("/scratch-shared/ktburg/rncdl_preds", rncdl_file), map_location=torch.device("cpu")
    )

    boxes = rncdl_pred["instances"].pred_boxes.tensor
    labels = torch.tensor([label_map[label.item()] for label in rncdl_pred["instances"].pred_classes])
    scores = rncdl_pred["instances"].scores

    sam_boxes = box_xywh_to_xyxy(torch.tensor([data["bbox"] for data in sam_data]))
    ious, _ = box_iou(sam_boxes, boxes)

    new_boxes = torch.empty(boxes.shape)
    for i in range(boxes.shape[0]):
        ious_ind = torch.argsort(ious[:, i], descending=True)
        new_boxes[i] = sam_boxes[ious_ind[0]]

    results = {}
    results[img_id] = {"boxes": box_xyxy_to_xywh(new_boxes), "labels": labels, "scores": scores}
    evaluator.update(results)

evaluator.synchronize_between_processes()
evaluator.accumulate()
results = evaluator.summarize()
print(results)
