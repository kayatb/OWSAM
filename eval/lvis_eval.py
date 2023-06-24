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
from lvis import LVIS

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


class LVISEval:
    def __init__(self, lvis_gt, lvis_dt=None, iou_type="segm"):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation file)
            lvis_dt (LVISResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
        elif lvis_dt is not None:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.stats = []
        self.ious = {}  # ious between all gts and dts

        # self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.img_ids = [139]
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())

    def _to_mask(self, anns, lvis):
        for ann in anns:
            rle = lvis.ann_to_rle(ann)
            ann["segmentation"] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids))
        dts = self.lvis_dt.load_anns(self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids))
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            if gt["image_id"] != 139:
                continue
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        print("LEN GTS:", len(self._gts))

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["neg_category_ids"] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann["image_id"]].add(ann["category_id"])
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.img_nel = {d["id"]: d["not_exhaustive_category_ids"] for d in img_data}

        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()

    def _prepare_freq_group(self):
        freq_groups = [[] for _ in self.params.img_count_lbl]
        cat_data = self.lvis_gt.load_cats(self.params.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            frequency = _cat_data["frequency"]
            freq_groups[self.params.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {
            (img_id, cat_id): self.compute_iou(img_id, cat_id) for img_id in self.params.img_ids for cat_id in cat_ids
        }

        # loop through images, area range, max detection number
        self.eval_imgs = [
            self.evaluate_img(img_id, cat_id, area_rng)
            for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def _get_gt_dt(self, img_id, cat_id):
        """Create gt, dt which are list of anns/dets. If use_cats is true
        only anns/dets corresponding to tuple (img_id, cat_id) will be
        used. Else, all anns/dets in image are used and cat_id is not used.
        """
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [_ann for _cat_id in self.params.cat_ids for _ann in self._gts[img_id, cat_id]]
            dt = [_ann for _cat_id in self.params.cat_ids for _ann in self._dts[img_id, cat_id]]
        return gt, dt

    def compute_iou(self, img_id, cat_id):
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_util.iou(dt, gt, iscrowd)
        return ious

    def evaluate_img(self, img_id, cat_id, area_rng):
        """Perform evaluation for single category and image."""
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = self.ious[img_id, cat_id][:, gt_idx] if len(self.ious[img_id, cat_id]) > 0 else self.ious[img_id, cat_id]

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d["area"] < area_rng[0] or d["area"] > area_rng[1] or d["category_id"] in self.img_nel[d["image_id"]]
            for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """

        if not self.eval_imgs:
            print("Warning: Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones((num_thrs, num_recalls, num_cats, num_area_rngs))
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [self.eval_imgs[Nk + Na + img_idx] for img_idx in range(num_imgs)]
                # Remove elements which are None
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[-1]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(rc, self.params.rec_thrs, side="left")

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except Exception:
                        pass
                    precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(pr_at_recall)

        self.eval = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }

    def _summarize(self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None):
        aidx = [idx for idx, _area_rng in enumerate(self.params.area_rng_lbl) if _area_rng == area_rng]

        if summary_type == "ap":
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"] = self._summarize("ap")
        self.results["AP50"] = self._summarize("ap", iou_thr=0.50)
        self.results["AP75"] = self._summarize("ap", iou_thr=0.75)
        self.results["APs"] = self._summarize("ap", area_rng="small")
        self.results["APm"] = self._summarize("ap", area_rng="medium")
        self.results["APl"] = self._summarize("ap", area_rng="large")
        self.results["APr"] = self._summarize("ap", freq_group_idx=0)
        self.results["APc"] = self._summarize("ap", freq_group_idx=1)
        self.results["APf"] = self._summarize("ap", freq_group_idx=2)

        self.stats = np.zeros((9,))
        self.stats[0] = self._summarize("ap")
        self.stats[1] = self._summarize("ap", iou_thr=0.50)
        self.stats[2] = self._summarize("ap", iou_thr=0.75)
        self.stats[3] = self._summarize("ap", area_rng="small")
        self.stats[4] = self._summarize("ap", area_rng="medium")
        self.stats[5] = self._summarize("ap", area_rng="large")
        self.stats[6] = self._summarize("ap", freq_group_idx=0)
        self.stats[7] = self._summarize("ap", freq_group_idx=1)
        self.stats[8] = self._summarize("ap", freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize("ar")

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results[key] = self._summarize("ar", area_rng=area_rng)
        self.print_results()

    def run(self):
        """Wrapper function which calculates the results."""
        self.evaluate()
        self.accumulate()
        self.summarize()

    def print_results(self):
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for key, value in self.results.items():
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

    def get_results(self):
        if not self.results:
            print("Warning: results is empty. Call run().")
        return self.results


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

    # def synchronize_between_processes(self):
    #     for iou_type in self.iou_types:
    #         self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
    #         create_common_lvis_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    # def accumulate(self):
    #     for lvis_eval in self.coco_eval.values():
    #         lvis_eval.accumulate()

    def summarize(self):
        for iou_type, lvis_eval in self.lvis_eval.items():
            print("IoU metric: {}".format(iou_type))
            lvis_eval.run()

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
