from utils.misc import box_xywh_to_xyxy
import utils.transforms as T
from modelling.sam_roi_heads import SAMRoIHeads

import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import _resize_image_and_masks, resize_boxes
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional


class GeneralizedRCNNTransformSAM(GeneralizedRCNNTransform):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / SAM boxes / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    # def __init__(
    #     self,
    #     min_size: int,
    #     max_size: int,
    #     image_mean: List[float],
    #     image_std: List[float],
    #     size_divisible: int = 32,
    #     fixed_size: Optional[Tuple[int, int]] = None,
    #     **kwargs: Any,
    # ):
    #     super().__init__()
    #     if not isinstance(min_size, (list, tuple)):
    #         min_size = (min_size,)
    #     self.min_size = min_size
    #     self.max_size = max_size
    #     self.image_mean = image_mean
    #     self.image_std = image_std
    #     self.size_divisible = size_divisible
    #     self.fixed_size = fixed_size
    #     self._skip_resize = kwargs.pop("_skip_resize", False)

    def forward(self, images, sam_boxes, targets=None, is_discovery_train=False):
        if is_discovery_train:
            return self.discovery_transform(images, sam_boxes)
        else:
            return self._forward(images, sam_boxes, targets)

    def _forward(
        self, images: List[Tensor], sam_boxes: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        """Compared to original torchvision forward: added the pre-extracted SAM boxes."""
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            # TODO: do I also need to copy the SAM boxes?
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                # for k, v in t.items():
                #     data[k] = v
                data["labels"] = t["labels"]
                data["boxes"] = box_xywh_to_xyxy(t["boxes"])  # Convert boxes to format expected by Faster R-CNN.
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            sam_index = sam_boxes[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            image = self.normalize(image)
            image, sam_index, target_index = self.resize(image, sam_index, target_index)
            image, sam_index, target_index = self.horizontal_flip(image, sam_index, target_index)
            images[i] = image
            sam_boxes[i] = sam_index
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_list = self.get_image_list(images)

        return image_list, sam_boxes, targets

    def discovery_transform(self, images, sam_boxes):
        """Transform the images and boxes correspondingly, where we augment the
        images twice each, to obtain multiple views for the discovery training.
        Augmentations used: random color distortion (color jitter, grayscale), random Gaussian blur, random resizing,
        and random horizontal flip.
        TODO: The different views are returned as one batch as [all_imgs_view1, all_imgs_view2]."""
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            sam_index = sam_boxes[i]

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")

            image = self.discovery_augmentations(image)

            image = self.normalize(image)
            image, sam_index, _ = self.resize(image, sam_index)
            image, sam_index, _ = self.horizontal_flip(image, sam_index)

            images[i] = image
            sam_boxes[i] = sam_index

        image_list = self.get_image_list(images)
        return image_list, sam_boxes, None  # None to remain compatible with standard transform.

    def get_image_list(self, images):
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list

    def discovery_augmentations(self, image):
        """Do color jitter, grayscale, and gaussian blur augmentations."""
        transform = T.Compose(
            [
                T.ColorJitter(0.8, 0.8, 0.8, 0.2, prob=0.8),
                T.GrayScale(num_output_channels=3, prob=0.2),
                T.GaussianBlur(sigma=[0.1, 2.0], prob=0.5),
            ]
        )
        image, _ = transform(image)

        return image

    def resize(
        self,
        image: Tensor,
        sam_boxes: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        bbox = sam_boxes
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        sam_boxes = bbox

        if target is None:
            return image, sam_boxes, target

        # Resize the target boxes and SAM boxes.
        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, sam_boxes, target

    def horizontal_flip(self, image, sam_boxes, target=None, p=0.5):
        """Do a horizontal flip if in training mode with probability p. Otherwise, do nothing."""
        if not self.training or torch.rand(1) >= p:
            return image, sam_boxes, target

        image = F.hflip(image)
        _, _, width = F.get_dimensions(image)
        sam_boxes[:, [0, 2]] = width - sam_boxes[:, [2, 0]]

        if target is not None:
            target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]

        return image, sam_boxes, target

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            # if "masks" in pred:
            #     masks = pred["masks"]
            #     masks = paste_masks_in_image(masks, boxes, o_im_s)
            #     result[i]["masks"] = masks
        return result


class RegionProposalNetworkSAM(nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """

    def __init__(
        self,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def filter_proposals(self, proposals, scores):
        # During discovery, TUM does no NMS and keeps the top 50 proposals.
        if self.nms_thresh <= 1.0:
            for i in range(len(proposals)):
                keep = box_ops.nms(proposals[i], scores[i], self.nms_thresh)  # TODO: could also use batched_nms here.
                proposals[i], scores[i] = proposals[i][keep], scores[i][keep]
        return proposals, scores

    def forward(self, proposals, scores, is_discovery_train=False):
        if is_discovery_train:
            proposals, scores = self.filter_proposals(proposals, scores)

        return proposals, scores


class SAMRPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    def __init__(self, in_channels: int, resolution, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        # self.cls_logits = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.cls_logits = nn.Linear(self.conv[-1].out_channels * resolution, 1)
        # self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        # bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            t = t.flatten()
            logits.append(self.cls_logits(t))
            # bbox_reg.append(self.bbox_pred(t))
        return torch.stack(logits)  # , bbox_reg


class RegionProposalNetworkSAMObjectness(RegionProposalNetwork):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """

    # def __init__(
    #     self,
    #     anchor_generator: AnchorGenerator,
    #     head: nn.Module,
    #     # Faster-RCNN Training
    #     fg_iou_thresh: float,
    #     bg_iou_thresh: float,
    #     batch_size_per_image: int,
    #     positive_fraction: float,
    #     # Faster-RCNN Inference
    #     pre_nms_top_n: Dict[str, int],
    #     post_nms_top_n: Dict[str, int],
    #     nms_thresh: float,
    #     score_thresh: float = 0.0,
    # ) -> None:
    #     super().__init__()
    #     self.anchor_generator = anchor_generator
    #     self.head = head
    #     self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    #     # used during training
    #     self.box_similarity = box_ops.box_iou

    #     self.proposal_matcher = det_utils.Matcher(
    #         fg_iou_thresh,
    #         bg_iou_thresh,
    #         allow_low_quality_matches=True,
    #     )

    #     self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
    #     # used during testing
    #     self._pre_nms_top_n = pre_nms_top_n
    #     self._post_nms_top_n = post_nms_top_n
    #     self.nms_thresh = nms_thresh
    #     self.score_thresh = score_thresh
    #     self.min_size = 1e-3

    # def pre_nms_top_n(self) -> int:
    #     if self.training:
    #         return self._pre_nms_top_n["training"]
    #     return self._pre_nms_top_n["testing"]

    # def post_nms_top_n(self) -> int:
    #     if self.training:
    #         return self._post_nms_top_n["training"]
    #     return self._post_nms_top_n["testing"]

    # def assign_targets_to_anchors(
    #     self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    # ) -> Tuple[List[Tensor], List[Tensor]]:
    #     labels = []
    #     matched_gt_boxes = []
    #     for anchors_per_image, targets_per_image in zip(anchors, targets):
    #         gt_boxes = targets_per_image["boxes"]

    #         if gt_boxes.numel() == 0:
    #             # Background image (negative example)
    #             device = anchors_per_image.device
    #             matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
    #             labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
    #         else:
    #             match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
    #             matched_idxs = self.proposal_matcher(match_quality_matrix)
    #             # get the targets corresponding GT for each proposal
    #             # NB: need to clamp the indices because we can have a single
    #             # GT in the image, and matched_idxs can be -2, which goes
    #             # out of bounds
    #             matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

    #             labels_per_image = matched_idxs >= 0
    #             labels_per_image = labels_per_image.to(dtype=torch.float32)

    #             # Background (negative examples)
    #             bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
    #             labels_per_image[bg_indices] = 0.0

    #             # discard indices that are between thresholds
    #             inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
    #             labels_per_image[inds_to_discard] = -1.0

    #         labels.append(labels_per_image)
    #         matched_gt_boxes.append(matched_gt_boxes_per_image)
    #     return labels, matched_gt_boxes

    # def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
    #     r = []
    #     offset = 0
    #     for ob in objectness.split(num_anchors_per_level, 1):
    #         num_anchors = ob.shape[1]
    #         pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
    #         _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
    #         r.append(top_n_idx + offset)
    #         offset += num_anchors
    #     return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        box_features: Tensor,
        objectness: Tensor,
        # image_shapes: List[Tuple[int, int]],
        # num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        num_images = len(proposals)
        # device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        box_features = box_features.detach()
        # objectness = objectness.reshape(num_images, -1)
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        objectness = torch.split(objectness, boxes_per_image)
        box_features = torch.split(box_features, boxes_per_image)

        # box_features = box_features.reshape(torch.cat(proposals).shape)

        # levels = [
        #     torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        # ]
        # levels = torch.cat(levels, 0)
        # levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # image_range = torch.arange(num_images, device=device)
        # batch_idx = image_range[:, None]

        # objectness = objectness[batch_idx, top_n_idx]
        # levels = levels[batch_idx, top_n_idx]
        # proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = [torch.sigmoid(o) for o in objectness]
        # objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_features = []
        final_scores = []
        for boxes, scores, features in zip(proposals, objectness_prob, box_features):
            scores = scores.squeeze()
            # boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, features = boxes[keep], scores[keep], features[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, features = boxes[keep], scores[keep], features[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.nms(boxes, scores, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores, features = boxes[keep], scores[keep], features[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
            final_features.append(features)

        return final_boxes, final_scores, final_features

    def compute_loss(self, objectness: Tensor, labels: List[Tensor]) -> Tensor:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        # regression_targets = torch.cat(regression_targets, dim=0)

        # box_loss = F.smooth_l1_loss(
        #     pred_bbox_deltas[sampled_pos_inds],
        #     regression_targets[sampled_pos_inds],
        #     beta=1 / 9,
        #     reduction="sum",
        # ) / (sampled_inds.numel())

        objectness_loss = nn.functional.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss  # , box_loss

    def forward(
        self,
        sam_boxes,
        box_features,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Args:
            sam_boxes: the bounding boxes obtained from SAM.
            box_features: features computed from the SAM boxes that are
                used for computing the predictions.
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        # features = list(features.values())
        objectness = self.head(box_features)
        # anchors = self.anchor_generator(images, features)

        # num_images = len(anchors)
        # num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        # objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # proposals = proposals.view(num_images, -1, 4)
        boxes, scores, features = self.filter_proposals(sam_boxes, box_features, objectness)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(sam_boxes, targets)
            loss_objectness = self.compute_loss(objectness, labels)
            losses = {
                "loss_objectness": loss_objectness,
            }
        return boxes, torch.cat(features), losses


class GeneralizedRCNNSAM(GeneralizedRCNN):
    """
    Main class for Generalized R-CNN with SAM as RPN.

    Args:
        backbone (nn.Module):
        rpn (RegionProposalNetworkSAM):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def forward(self, batch):
        """
        Args:
            images (list[Tensor]): images to be processed
            sam_boxes (list[Tensor]): boxes outputted by SAM
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if batch["targets"] is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in batch["targets"]:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in batch["images"]:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, sam_boxes, targets = self.transform(batch["images"], batch["sam_boxes"], batch["targets"])
        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        for boxes_idx, boxes in enumerate(sam_boxes):
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {boxes_idx}. {batch['img_ids'][boxes_idx]}",
                )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, scores = self.rpn(sam_boxes, batch["iou_scores"])
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        # losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    def get_box_features(self, batch, is_discovery_train=False, num_views=2):
        """Get the box features for the given images and boxes, i.e.
        feature extraction for the image, RPN, RoI pooling, features for the RoIs."""
        # TODO: batch the different views to enable a single forward pass.
        all_features = []
        for _ in range(num_views):
            images, sam_boxes, _ = self.transform(
                batch["images"], batch["sam_boxes"], is_discovery_train=is_discovery_train
            )
            proposals, _ = self.rpn(sam_boxes, batch["iou_scores"], is_discovery_train=is_discovery_train)
            features = self.backbone(images.tensors.to(sam_boxes[0].device))

            box_features = self.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            box_features = self.roi_heads.box_head(box_features)

            all_features.append(box_features)

        # Ensure each view yields the same amount of boxes. Can differ slightly between different augmentations+NMS.
        num_features = min(map(len, all_features))
        all_features = [feats[:num_features] for feats in all_features]

        return all_features


class FasterRCNNSAM(GeneralizedRCNNSAM):
    # Copied from torchvision's FasterRCNN class.
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=300,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        bg_weight=1.0,
        num_bg_classes=1,
        **kwargs,
    ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        self.num_classes = num_classes
        self.num_bg_classes = num_bg_classes

        out_channels = backbone.out_channels

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetworkSAM(rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh, rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=0)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        # Need access to these values for changing RoIHeads in discovery phase.
        self.box_fg_iou_thresh = box_fg_iou_thresh
        self.box_bg_iou_thresh = box_bg_iou_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.bbox_reg_weights = bbox_reg_weights

        box_rpn = RegionProposalNetworkSAMObjectness(
            None,
            SAMRPNHead(backbone.out_channels, resolution**2),
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        roi_heads = SAMRoIHeads(
            box_rpn,
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            bg_weight=bg_weight,
            num_bg_classes=num_bg_classes,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransformSAM(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)

    def freeze(self):
        """Freeze all components but the classifier for discovery training."""
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the classification head.
        # for param in self.roi_heads.box_predictor.cls_score.parameters():
        for param in self.classifier.parameters():
            param.requires_grad = True

    @property
    def classifier(self):
        return self.roi_heads.box_predictor.cls_score
