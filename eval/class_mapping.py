"""
Get the class mapping between discovery class IDs and ground truth class IDs by
using the Hungarian algorithm.

Partially copied and adapted from RNCDL:
https://github.com/vlfom/RNCDL/blob/main/discovery/evaluation/evaluator_discovery.py
"""

import torch
import numpy as np

# import detectron2.utils.comm as comm

# from detectron2.evaluation import DatasetEvaluator
# from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from scipy.optimize import linear_sum_assignment


class ClassMapper:
    """
    Obtains discovery_id -> GT class_id based on the provided GT annotations and their predictions via solving an
    optimal transport problem.

    NOTE: this does not apply for my implementation --> Note: GT class_id here is the class_id that D2 provides to the models during training. Usually, those are not real
    GT class ids, but rather them remapped to consecutive numbers.
    """

    def __init__(self, last_free_class_id, max_class_num, novel_class_id_thresh):
        """
        Args:
            last_free_class_id: maximum total number of classes in the dataset
            max_class_num: a minimum value that can be used as a class ID for novel category that is guaranteed to be
                           higher than the number of known classes; used to assign IDs to novel classes and to avoid
                           IDs overlapping
            novel_class_id_thresh: maximum value of the ID belonging to an actual GT class.
        """
        self.last_free_class_id = last_free_class_id
        self.max_class_num = max_class_num
        self.novel_class_id_thresh = novel_class_id_thresh

        self.targets_all = None
        self.predictions_all = None
        self.class_mapping = None

    def reset(self):
        self.targets_all = []
        self.predictions_all = []
        self.class_mapping = None

    def update(self, inputs, outputs):
        targets = torch.cat([target["labels"] for target in inputs["targets"]])

        assert targets.shape[0] == outputs.shape[0]

        self.targets_all.append(targets)
        self.predictions_all.append(outputs)

    def get_mapping(self):
        # Concat all targets and predictions per current GPU
        self.targets_all = np.array(torch.cat(self.targets_all).cpu())
        self.predictions_all = np.array(torch.cat(self.predictions_all).cpu())

        # Collect all data on the main GPU
        # comm.synchronize()
        # targets_all = comm.gather(self.targets_all, dst=0)
        # predictions_all = comm.gather(self.predictions_all, dst=0)

        # if not comm.is_main_process():
        #     return None

        # targets_all = np.concatenate(targets_all)
        # predictions_all = np.concatenate(predictions_all)

        assert self.targets_all.shape[0] == self.predictions_all.shape[0]

        class_mapping = cluster_map(
            self.targets_all,
            self.predictions_all,
            last_free_class_id=self.last_free_class_id,
            max_class_num=self.max_class_num,
        )
        class_mapping = np.array(class_mapping).astype(np.int)
        self.class_mapping = dict(torch.from_numpy(class_mapping).cuda())

    def map_predictions(self, predictions):
        # Remap predicted classes and filter out predictions of novel categories
        predictions_mapped = []
        for pred in predictions:
            pred["category_id"] = self.class_mapping[pred["category_id"]]
            if pred["category_id"] < self.novel_class_id_thresh:
                predictions_mapped.append(pred)

        return predictions_mapped


def cluster_map(y_true, y_pred, last_free_class_id, max_class_num):
    """
    Args:
        last_free_class_id: least possible free ID to use for novel classes, defaults to 1203 as LVIS class IDs
                             range up to 1203
    """

    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    # Generate weight matrix
    max_y_pred = y_pred.max()
    max_y_true = y_true.max()

    w = np.zeros((max_y_pred + 1, max_y_true + 1), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Match
    x, y = linear_sum_assignment(w, maximize=True)

    mapping = list(zip(x, y))

    # Create fake extra classes for unmapped categories
    for i in range(0, max_class_num):
        if i not in x:
            mapping.append((i, last_free_class_id))
            last_free_class_id += 1

    return mapping


# class DiscoveryEvaluator(DatasetEvaluator):
#     """
#     Wrapper around existing D2 Evaluators that supports category re-mapping.
#     Modifies the `process` method only.

#     Note: currently the support has been checked only COCOEvaluator and LVISEvaluator, which both similarly process
#     cache the inputs/outputs inside the `process()`. Other evaluators may be supported as is, but it is not guaranteed.
#     """

#     def __init__(self, evaluator, novel_class_id_thresh):
#         self.evaluator = evaluator
#         self.novel_class_id_thresh = novel_class_id_thresh
#         self.class_mapping = None

#         self._debug_dumped = 0

#     def set_class_mapping(self, class_mapping):
#         class_mapping = dict(np.array(class_mapping.cpu()))  # Convert to dict mapping
#         self.class_mapping = class_mapping

#     def reset(self):
#         self.class_mapping = None
#         return self.evaluator.reset()

#     def process(self, inputs, outputs):
#         for input, output in zip(inputs, outputs):
#             prediction = {"image_id": input["image_id"]}

#             instances = output["instances"].to("cpu")
#             prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

#             # Remap predicted classes and filter out predictions of novel categories
#             predictions_mapped = []
#             for pred in prediction["instances"]:
#                 pred["category_id"] = self.class_mapping[pred["category_id"]]
#                 if pred["category_id"] < self.novel_class_id_thresh:
#                     predictions_mapped.append(pred)

#             prediction["instances"] = predictions_mapped

#             self.evaluator._predictions.append(prediction)

#     def evaluate(self):
#         return self.evaluator.evaluate()
