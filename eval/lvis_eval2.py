from eval.coco_eval import create_common_coco_eval

from lvis import LVIS, LVISEval, LVISResults
import numpy as np
from collections import OrderedDict, defaultdict


class LvisEvaluator:
    def __init__(self, lvis_gt_ann, iou_types):
        self.lvis_gt_ann = lvis_gt_ann
        self.lvis = LVIS(lvis_gt_ann)
        self.iou_types = iou_types
        self.preds = {}
        for iou_type in self.iou_types:
            self.preds[iou_type] = []

    def reset(self):
        for iou_type in self.iou_types:
            self.preds[iou_type] = []

    def update(self, predictions):
        # for iou_type in self.iou_types:
        #     preds = self.prepare(predictions, iou_type)
        #     self.preds[iou_type].extend(preds)
        cur_results = self.prepare(predictions, iou_type="bbox")
        by_id = defaultdict(list)
        for ann in cur_results:
            by_id[ann["image_id"]].append(ann)

        for id_anns in by_id.values():
            self.preds["bbox"].extend(sorted(id_anns, key=lambda x: x["score"], reverse=True)[:300])

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_lvis_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_lvis_segmentation(predictions)
        # elif iou_type == "keypoints":
        #     return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_lvis_detection(self, predictions):
        lvis_preds = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"].tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            lvis_preds.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return lvis_preds

    def summarize(self):
        results = {}
        for iou_type, preds in self.preds.items():
            lvis_results = LVISResults(self.lvis_gt_ann, preds)
            lvis_eval = LVISEval(self.lvis, lvis_results, iou_type=iou_type)
            lvis_eval.run()
            lvis_eval.print_results()
        #     lvis_eval.accumulate()
        #     print("IoU metric: {}".format(iou_type))
        #     lvis_eval.summarize()
        #     # Obtain the summarized evaluation results in a nice dict format for logging purposes.
        #     result_keys = [
        #         "map",
        #         "map50",
        #         "map75",
        #         "map_small",
        #         "map_medium",
        #         "map_large",
        #         "mar1",
        #         "mar10",
        #         "mar100",
        #         "mar_small",
        #         "mar_medium",
        #         "mar_large",
        #     ]
        #     results[iou_type] = {k: v for k, v in zip(result_keys, self.coco_eval[iou_type].stats)}

        # return results

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    # def accumulate(self):
    #     for coco_eval in self.coco_eval.values():
    #         coco_eval.accumulate()
