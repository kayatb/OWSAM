"""
Get the class mapping between discovery class IDs and ground truth class IDs by
using the Hungarian algorithm.

Partially copied and adapted from RNCDL:
https://github.com/vlfom/RNCDL/blob/main/discovery/evaluation/evaluator_discovery.py
"""
from eval.lvis_eval import LvisEvaluator, LVISEvalDiscovery, LVISResults
import configs.discovery as config
from discovery.discovery_model import DiscoveryModel, ForwardMode
from data.datasets.fasterrcnn_data import ImageData
from utils.misc import box_xyxy_to_xywh

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import json
import pickle


class ClassMapper:
    """
    Obtains discovery_id -> GT class_id based on the provided GT annotations and their predictions via solving an
    optimal transport problem.

    NOTE: GT class_id here is the class_id that is provided to the models during training. Usually, those are not real
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

        self.targets_all = []
        self.predictions_all = []
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
        class_mapping = np.array(class_mapping).astype(int)
        # self.class_mapping = dict(torch.from_numpy(class_mapping).cuda())
        self.class_mapping = dict(class_mapping)

    # def map_predictions(self, predictions):
    #     # Remap predicted classes and filter out predictions of novel categories
    #     predictions_mapped = []
    #     for pred in predictions:
    #         pred["category_id"] = self.class_mapping[pred["category_id"]]
    #         if pred["category_id"] < self.novel_class_id_thresh:
    #             predictions_mapped.append(pred)

    #     return predictions_mapped


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


class DiscoveryEvaluator:
    """First calculate the discovery ID to GT ID class mapping.
    Also save the predictions to later map to the correct category and evaluate on the LVIS dataset."""

    def __init__(self, model, label_maping):
        self.model = model
        model.eval()

        self.label_mapping = label_mapping  # Map from dataset labels to GT annotation labels

        # self.evaluator = LvisEvaluator(config.ann_val_unlabeled, ["bbox"], known_class_ids=config.lvis_known_class_ids)
        self.class_mapper = ClassMapper(config.last_free_class_id, config.max_class_num, config.novel_class_id_thresh)
        self.unsupervis_preds = {}

    def reset(self):
        # self.evaluator.reset()
        self.class_mapper.reset()
        self.unsupervis_preds = {}

    def update(self, batch, is_supervis):
        """Update the class mapping with the predictions from the batch.
        Also save the predictions from the batch for later evaluation if it comes from the unsupervised batch.
        Args:
            batch: the batch with the data to run the model on.
            is_supervis: boolean to signify whether the batch comes from the supervised or unsupervised dataset."""
        # Move data to device
        # batch["images"] = batch["images"].to(config.device)
        batch["sam_boxes"] = [box.to(config.device) for box in batch["sam_boxes"]]
        batch["targets"] = [{"labels": t["labels"], "boxes": t["boxes"].to(config.device)} for t in batch["targets"]]

        # Get outputs from the model.
        sam_outputs, gt_preds = self.model.extract_gt_preds(batch, is_supervis=is_supervis)

        # Save predictions from the unsupervised dataset.
        if not is_supervis:
            for i, id in enumerate(batch["img_ids"]):
                self.unsupervis_preds[id] = sam_outputs[i]
            # self.unsupervis_preds.extend(self.pred_to_lvis_format(batch, sam_outputs))

        self.class_mapper.update(batch, gt_preds)

    def evaluate(self, data):
        """After the class mapping has been calculated, map all predicted IDs to their assigned GT ID and
        calculate the measurements."""
        self.class_mapper.get_mapping()
        evaluator = LvisEvaluator(config.ann_val_unlabeled, ["bbox"], known_class_ids=config.lvis_known_class_ids)

        # For saving of the predictions and found class mapping:
        # print("Saving the class_mapping at owsam_eval_lvis_class_mapping.json")
        # with open("owsam_eval_lvis_class_mapping.p", "wb") as fp:
        #     pickle.dump(self.class_mapper.class_mapping, fp)
        # print("Saved class mapping")

        # print("Saving predictions at owsam_eval_lvis_preds.json")
        # with open("owsam_eval_lvis_preds.p", "wb") as fp:
        #     pickle.dump(self.unsupervis_preds, fp)
        # print("Saved predictions")

        # mapped_preds = []
        # for pred in tqdm(self.unsupervis_preds):
        #     # Map from the predicted label to the label found with the Hungarian algorithm.
        #     pred["category_id"] = self.class_mapper.class_mapping[pred["category_id"]]
        #     # Filter out predictions of novel categories (i.e. not mapped to a GT category).
        #     if pred["category_id"] < config.novel_class_id_thresh:
        #         # Now map from that label back to the labels as in the original annotations.
        #         pred["category_id"] = self.label_mapping[pred["category_id"]]
        #         mapped_preds.append(pred)
        # for batch in tqdm(data):
        #     batch["sam_boxes"] = [box.to(config.device) for box in batch["sam_boxes"]]
        #     batch["targets"] = [
        #         {"labels": t["labels"], "boxes": t["boxes"].to(config.device)} for t in batch["targets"]
        #     ]
        #     with torch.no_grad():
        #         preds = self.model(supervised_batch=None, unsupervised_batch=batch, mode=ForwardMode.UNSUPERVISED_VAL)

        #     results = {}
        #     for i in range(len(batch["img_ids"])):
        #         labels = torch.argmax(preds[i], dim=-1).cpu().apply_(self.class_mapper.class_mapping.get)
        #         print("before:", labels.shape)
        #         if labels.dim() < 2:
        #             labels = labels.unsqueeze(0)
        #         print("after:", labels.shape)
        #         results[batch["img_ids"][i]] = {
        #             "boxes": box_xyxy_to_xywh(batch["sam_boxes"][i]),  # Original COCO/LVIS box format.
        #             "labels": labels,
        #             "scores": batch["iou_scores"][i],
        #         }
        evaluator.update(self.unsupervis_preds)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()

        evaluator.summarize()
        # for pred in mapped_preds:
        #     self.evaluator.update(pred)

        # results = self.discovery_evaluator.summarize()
        # return results
        # lvis_eval = LVISEvalDiscovery(
        #     config.ann_val_unlabeled, mapped_preds, "bbox", known_class_ids=config.lvis_known_class_ids
        # )
        # lvis_eval.run()
        # lvis_eval.print_results()

    def pred_to_lvis_format(self, batch, outputs):
        """Get a model prediction in the format for LVIS evaluation. Returns a list of dicts with keys:
        image_id, category_id, bbox, score"""
        formatted = []
        for i in range(len(batch["img_ids"])):
            img_id = batch["img_ids"][i]
            boxes = np.array(box_xyxy_to_xywh(outputs[i]["boxes"]).cpu())
            scores = outputs[i]["scores"].cpu()
            labels = outputs[i]["labels"].cpu()

            formatted.extend(
                [
                    {
                        "image_id": img_id,
                        # During prediction extraction, we add a fake background class, which moves every label one to
                        # the right (i.e. background is set as ID 0). To match with the label mappers, reverse this.
                        "category_id": labels[k].item() - 1,
                        "bbox": box,
                        "score": scores[k].item(),
                    }
                    for k, box in enumerate(boxes)
                ]
            )
            return formatted


if __name__ == "__main__":
    # Load the pre-trained model.
    discovery_ckpt_path = "checkpoints/discovery_TUM_3epochs_2gpus_0.7nms/epoch=2-step=18636.ckpt"
    batch_size = 1

    model = DiscoveryModel(config.supervis_ckpt)
    # NOTE: if this throws errors for `discovery_model.memory_feat`, change the batch size to value it was trained with.
    model.from_checkpoint(discovery_ckpt_path)
    model.to(config.device)

    # Load the data
    dataset_val_labeled = ImageData(
        config.masks_dir,
        config.ann_val_labeled,
        config.img_dir,
        config.device,
    )
    # dataset_val_labeled.img_ids = dataset_val_labeled.img_ids[:1]

    dataloader_val_labeled = DataLoader(
        dataset_val_labeled,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ImageData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    dataset_val_unlabeled = ImageData(
        config.masks_dir,
        config.ann_val_unlabeled,
        config.img_dir,
        config.device,
    )

    # dataset_val_unlabeled.img_ids = dataset_val_unlabeled.img_ids[:4]

    dataloader_val_unlabeled = DataLoader(
        dataset_val_unlabeled,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ImageData.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )

    label_mapping = dataset_val_unlabeled.continuous_to_cat_id
    evaluator = DiscoveryEvaluator(model, label_mapping)

    print("Processing supervised data...")
    for batch in tqdm(dataloader_val_labeled):
        evaluator.update(batch, is_supervis=True)

    print("Processing unsupervised data...")
    for batch in tqdm(dataloader_val_unlabeled):
        evaluator.update(batch, is_supervis=False)

    print("Evaluating the predictions...")
    evaluator.evaluate(dataloader_val_unlabeled)

    # for id in config.lvis_known_class_ids:
    #     # i = dataset_val_unlabeled.cat_id_to_continuous[id]
    #     print(dataset_val_unlabeled.cat_id_to_name[id])

    # with open("owsam_eval_lvis_class_mapping.p", "rb") as fp:
    #     class_mapping = pickle.load(fp)

    # for key, item in class_mapping.items():
    #     if label_mapping[item] < config.novel_class_id_thresh:
    #         print(f"{key}: {dataset_val_unlabeled.cat_id_to_name[label_mapping[item]]}")
    # print("=======================================================")
    # with open("owsam_eval_lvis_preds.p", "rb") as fp:
    #     print(pickle.load(fp))
