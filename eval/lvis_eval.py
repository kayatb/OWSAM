"""
Copied and adapted from: https://github.com/ashkamath/mdetr/blob/main/datasets/lvis_eval.py
"""
# import copy
import datetime
from collections import OrderedDict, defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch

from .coco_eval import merge
from lvis import LVIS, LVISEval

#################################################################
# From LVIS, with following changes:
#     * fixed LVISEval constructor to accept empty dt
#     * Removed logger
#     * LVIS results supports numpy inputs
#################################################################


class Params:
    def __init__(self, iou_type):
        """Params for LVIS evaluation API."""
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.rec_thrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.max_dets = 300
        self.area_rng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.use_cats = 1
        # We bin categories in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ["r", "c", "f"]
        self.iou_type = iou_type


class LVISResults(LVIS):
    def __init__(self, lvis_gt, results, max_dets=300):
        """Constructor for LVIS results.
        Args:
            lvis_gt (LVIS class instance, or str containing path of
            annotation file)
            results (str containing path of result file or a list of dicts)
            max_dets (int):  max number of detections per image. The official
            value of max_dets for LVIS is 300.
        """
        assert isinstance(lvis_gt, LVIS)
        self.lvis_gt = lvis_gt
        self.dataset = lvis_gt.dataset

        if isinstance(results, str):
            result_anns = self._load_json(results)
        elif type(results) == np.ndarray:
            result_anns = self.loadNumpyAnnotations(results)
        else:
            result_anns = results

        if max_dets >= 0:
            result_anns = self.limit_dets_per_image(result_anns, max_dets)

        if len(result_anns) > 0 and "bbox" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann["area"] = w * h
                ann["id"] = id + 1

        elif len(result_anns) > 0 and "segmentation" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                # Only support compressed RLE format as segmentation results
                ann["area"] = mask_util.area(ann["segmentation"])

                if "bbox" not in ann:
                    ann["bbox"] = mask_util.toBbox(ann["segmentation"])

                ann["id"] = id + 1

        self.dataset["annotations"] = result_anns
        self._create_index()

    def limit_dets_per_image(self, anns, max_dets):
        img_ann = defaultdict(list)
        for ann in anns:
            img_ann[ann["image_id"]].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda ann: ann["score"], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]

    def get_top_results(self, img_id, score_thrs):
        ann_ids = self.get_ann_ids(img_ids=[img_id])
        anns = self.load_anns(ann_ids)
        return list(filter(lambda ann: ann["score"] > score_thrs, anns))

    def _create_index(self):
        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])


#################################################################
# end of straight copy from lvis, just fixing constructor
#################################################################


# Class is copy-pasted from RNCDL:
# https://github.com/vlfom/RNCDL/blob/main/discovery/evaluation/lvis_evaluation.py#L148
class LVISEvalDiscovery(LVISEval):
    """
    Extends `LVISEval` with printing results for known and novel classes only when `known_class_ids` is provided.
    """

    def __init__(self, lvis_gt, lvis_dt, iou_type="segm", known_class_ids=None):
        super().__init__(lvis_gt, lvis_dt, iou_type)

        # Remap categories list following the mapping applied to train data, - that is list all categories in a
        # consecutive order and use their indices; see: `lvis-api/lvis/eval.py` line 109:
        # https://github.com/lvis-dataset/lvis-api/blob/35f09cd7c5f313a9bf27b329ca80effe2b0c8a93/lvis/eval.py#L109
        if known_class_ids is None:
            self.known_class_ids = None
        else:
            self.known_class_ids = [self.params.cat_ids.index(c) for c in known_class_ids]

    def _summarize(self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None, subset_class_ids=None):
        """Extends the default version by supporting calculating the results only for the subset of classes."""

        if subset_class_ids is None:  # Use all classes
            subset_class_ids = list(range(len(self.params.cat_ids)))

        aidx = [idx for idx, _area_rng in enumerate(self.params.area_rng_lbl) if _area_rng == area_rng]

        if summary_type == "ap":
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                subset_class_ids = list(set(subset_class_ids).intersection(self.freq_groups[freq_group_idx]))
                s = s[:, :, subset_class_ids, aidx]
            else:
                s = s[:, :, subset_class_ids, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, subset_class_ids, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Extends the default version by supporting calculating the results only for the subset of classes."""

        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        if self.known_class_ids is None:
            eval_groups = [(self.results, None)]
        else:
            cat_ids_mapped_list = list(range(len(self.params.cat_ids)))
            novel_class_ids = list(set(cat_ids_mapped_list).difference(self.known_class_ids))
            self.results_known = OrderedDict()
            self.results_novel = OrderedDict()
            eval_groups = [
                (self.results, None),
                (self.results_known, self.known_class_ids),
                (self.results_novel, novel_class_ids),
            ]

        max_dets = self.params.max_dets

        for container, subset_class_ids in eval_groups:
            container["AP"] = self._summarize("ap", subset_class_ids=subset_class_ids)
            container["AP50"] = self._summarize("ap", iou_thr=0.50, subset_class_ids=subset_class_ids)
            container["AP75"] = self._summarize("ap", iou_thr=0.75, subset_class_ids=subset_class_ids)
            container["APs"] = self._summarize("ap", area_rng="small", subset_class_ids=subset_class_ids)
            container["APm"] = self._summarize("ap", area_rng="medium", subset_class_ids=subset_class_ids)
            container["APl"] = self._summarize("ap", area_rng="large", subset_class_ids=subset_class_ids)
            container["APr"] = self._summarize("ap", freq_group_idx=0, subset_class_ids=subset_class_ids)
            container["APc"] = self._summarize("ap", freq_group_idx=1, subset_class_ids=subset_class_ids)
            container["APf"] = self._summarize("ap", freq_group_idx=2, subset_class_ids=subset_class_ids)

            key = "AR@{}".format(max_dets)
            container[key] = self._summarize("ar", subset_class_ids=subset_class_ids)

            for area_rng in ["small", "medium", "large"]:
                key = "AR{}@{}".format(area_rng[0], max_dets)
                container[key] = self._summarize("ar", area_rng=area_rng, subset_class_ids=subset_class_ids)

    def print_results(self):
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for results, type in [(self.results, "all"), (self.results_known, "known"), (self.results_novel, "novel")]:
            print(f"========== Results for {type} classes ==========")
            for key, value in results.items():
                max_dets = self.params.max_dets
                if "AP" in key:
                    title = "Average Precision"
                    _type = "(AP)"
                else:
                    title = "Average Recall"
                    _type = "(AR)"

                if len(key) > 2 and key[2].isdigit():
                    iou_thr = float(key[2:]) / 100
                    iou = "{:0.2f}".format(iou_thr)
                else:
                    iou = "{:0.2f}:{:0.2f}".format(self.params.iou_thrs[0], self.params.iou_thrs[-1])

                if len(key) > 2 and key[2] in ["r", "c", "f"]:
                    cat_group_name = key[2]
                else:
                    cat_group_name = "all"

                if len(key) > 2 and key[2] in ["s", "m", "l"]:
                    area_rng = key[2]
                else:
                    area_rng = "all"

                print(template.format(title, _type, iou, area_rng, max_dets, cat_group_name, value))


class LvisEvaluator(object):
    def __init__(self, lvis_gt, iou_types, known_class_ids=None):
        assert isinstance(iou_types, (list, tuple))
        self.lvis_gt = LVIS(lvis_gt)

        self.iou_types = iou_types
        self.known_class_ids = known_class_ids  # Known class IDs, used for discovery.

        self.lvis_eval = {}
        for iou_type in iou_types:
            if known_class_ids is None:
                self.lvis_eval[iou_type] = LVISEval(self.lvis_gt, iou_type=iou_type)
            else:
                self.lvis_eval[iou_type] = LVISEvalDiscovery(
                    self.lvis_gt, iou_type=iou_type, known_class_ids=known_class_ids
                )

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def reset(self):
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            lvis_dt = LVISResults(self.lvis_gt, results)
            lvis_eval = self.lvis_eval[iou_type]

            lvis_eval.lvis_dt = lvis_dt
            lvis_eval.params.img_ids = list(img_ids)
            lvis_eval.evaluate()
            eval_imgs = lvis_eval.eval_imgs
            eval_imgs = np.asarray(eval_imgs).reshape(
                len(lvis_eval.params.cat_ids), len(lvis_eval.params.area_rng), len(lvis_eval.params.img_ids)
            )

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_lvis_eval(self.lvis_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for lvis_eval in self.lvis_eval.values():
            lvis_eval.accumulate()

    def summarize(self):
        for iou_type, lvis_eval in self.lvis_eval.items():
            print("IoU metric: {}".format(iou_type))
            lvis_eval.run()
            lvis_eval.print_results()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_lvis_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_lvis_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_lvis_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_lvis_detection(self, predictions):
        lvis_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"].tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            lvis_results.extend(
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
        return lvis_results

    @staticmethod
    def to_lvis_format(img_ids, outputs, label_map):
        results = {}

        for i in range(len(img_ids)):
            labels = torch.argmax(outputs["pred_logits"][i], dim=1)  # Get labels from the logits

            # Remove padded boxes and those that are predicted with no-object class.
            concat = torch.cat(
                (labels.unsqueeze(1), outputs["iou_scores"][i].unsqueeze(1), outputs["pred_boxes"][i]), dim=1
            )
            # concat = concat[concat[:, 0] != 80]

            labels = concat[:, 0]
            labels = labels.cpu().apply_(label_map.get)  # Map the labels to original ones from LVIS

            results[img_ids[i]] = {"boxes": concat[:, 2:], "scores": concat[:, 1], "labels": labels}

        return results

    def prepare_for_lvis_segmentation(self, predictions):
        lvis_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            lvis_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return lvis_results


def create_common_lvis_eval(lvis_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    lvis_eval.eval_imgs = eval_imgs
    lvis_eval.params.img_ids = img_ids
