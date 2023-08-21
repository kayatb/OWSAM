from utils.misc import add_padding, box_xywh_to_xyxy
from modelling.fasterrcnn_sam import FasterRCNNSAM

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead


class LinearClassifier(nn.Module):
    """A simple classification head on top of the hidden mask features extracted from SAM to classify the masks."""

    def __init__(self, num_layers, hidden_dim, num_classes, dropout_prob, pad_num=700, input_dim=1024):
        super().__init__()

        self.num_classes = num_classes
        self.pad_num = pad_num
        self.input_dim = input_dim

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.ELU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.ELU())

        self.classifier = nn.Linear(hidden_dim, self.num_classes + 1)  # +1 for the no-object/background class

    def forward(self, batch):
        # x = self.layers(batch["mask_features"])
        x = self.layers(batch["crop_features"].float())
        # x = batch["crop_features"].float()
        class_logits = self.classifier(x)

        padded_class_logits = add_padding(
            class_logits, batch["num_masks"], self.num_classes, self.pad_num, batch["boxes"].device
        )

        return {
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pad_num=700):
        super().__init__()
        self.num_classes = num_classes
        self.pad_num = pad_num

        # self.resnet = resnet18(weights=None)
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(dim, num_classes + 1)  # +1 for no-object/background class

    def forward(self, batch):
        class_logits = self.resnet(batch["crops"])

        padded_class_logits = add_padding(
            class_logits, batch["num_masks"], self.num_classes, self.pad_num, batch["boxes"].device
        )

        return {
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }


class SAMRPN(nn.Module):
    """Boxes extracted from SAM as ROI proposals (i.e. SAM is RPN), then ROI align on
    the feature map extracted from the whole image, and classification of those ROIs."""

    def __init__(self, num_classes, feature_extractor_ckpt=None, trainable_backbone_layers=5, pad_num=700):
        """Args:
        num_classes: number of classes to predict, without the background class
        feature_extractor_ckpt: location of the checkpoint for the feature extractor
        image_size: square size to which the images are resized
        trainable_backbone_layers: number of non-frozen layers starting from the final block. Valid values
        are between 0 and 5, with 5 meaning all blocks are trainable.
        freeze: whether every component but the classification head is frozen. This is needed for discovery."""
        super().__init__()
        self.num_classes = num_classes
        self.pad_num = pad_num

        box_head_dict = None
        classifier_dict = None
        if feature_extractor_ckpt:
            self.feature_extractor = self.load_resnet50_fpn_with_moco(feature_extractor_ckpt, trainable_backbone_layers)
        else:
            self.feature_extractor, box_head_dict, classifier_dict = self.load_pretrained_model(
                trainable_backbone_layers
            )

        self.box_roi_pool, self.box_head, self.classifier = self.load_roi_heads(
            self.num_classes + 1, box_head_dict, classifier_dict
        )

    def load_resnet50_fpn_with_moco(self, checkpoint_path, trainable_backbone_layers):
        moco_checkpoint = torch.load(checkpoint_path)

        # Rename MoCo pre-trained keys (taken from official MoCo repo):
        # https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L250
        state_dict = moco_checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # Retain only encoder_q up to before the embedding layer.
            if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                # Remove prefix.
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # Delete renamed or unused k.
            del state_dict[k]

        backbone = resnet50(weights=None)
        # Change the last ResNet block strides from (2, 2) to (1, 1)
        # to increase the output feature map from 14x14 to 28x28.
        backbone.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        backbone.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # Load the weights into ResNet-50
        backbone.load_state_dict(state_dict, strict=False)

        # Turn it into an FPN feature extractor.
        feature_extractor = _resnet_fpn_extractor(
            backbone, trainable_layers=trainable_backbone_layers, norm_layer=nn.BatchNorm2d
        )

        return feature_extractor

    def load_pretrained_model(
        self,
        trainable_backbone_layers,
        url="https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    ):
        """Load the weights for a Faster R-CNN network pre-trained on COCO and use SAM as the RPN."""
        # State dict for the complete pre-trained Faster R-CNN model.
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

        # Rename the state dict keys to those accepted by the model.
        backbone_state_dict = {}
        box_head_dict = {}
        classifier_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith("backbone"):
                backbone_state_dict[k[len("backbone.") :]] = state_dict[k]
            elif k.startswith("roi_heads.box_head"):
                box_head_dict[k[len("roi_heads.box_head.") :]] = state_dict[k]
            elif k.startswith("roi_heads.box_predictor.cls_score"):
                classifier_dict[k[len("roi_heads.box_predictor.cls_score.") :]] = state_dict[k]

        feature_extractor = _resnet_fpn_extractor(resnet50(weights=None), trainable_layers=trainable_backbone_layers)
        feature_extractor.load_state_dict(backbone_state_dict)

        return feature_extractor, box_head_dict, classifier_dict

    def load_roi_heads(self, num_classes, box_head_state_dict=None, classifier_state_dict=None):
        """Load the RoI pooler and the classifier."""
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        out_channels = self.feature_extractor.out_channels
        box_head = TwoMLPHead(out_channels * resolution**2, representation_size)
        if box_head_state_dict:
            box_head.load_state_dict(box_head_state_dict)

        representation_size = 1024
        classifier = nn.Linear(representation_size, num_classes)
        if classifier_state_dict:
            classifier.load_state_dict(classifier_state_dict)

        return box_roi_pool, box_head, classifier

    def forward(self, batch):
        with torch.set_grad_enabled(not self.freeze):
            features = self.feature_extractor(batch["images"])

            proposals = batch["trans_boxes"]  # Boxes resized same as the image.

            image_shapes = [img.shape[1:] for img in batch["images"]]
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)

            x = box_features.flatten(start_dim=1)

        class_logits = self.classifier(x)

        padded_class_logits = add_padding(
            class_logits, batch["num_masks"], self.num_classes, self.pad_num, batch["boxes"].device
        )

        return {
            "pred_logits": padded_class_logits,
            "pred_boxes": batch["boxes"],  # Boxes belonging to original image.
            "iou_scores": batch["iou_scores"],  # Used for mAP calculation
        }

    def freeze(self):
        """Freeze all components but the classifier for discovery training."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.box_roi_pool.parameters():
            param.requires_grad = False

        for param in self.box_head.parameters():
            param.requires_grad = False


def load_resnet50_fpn_with_moco(checkpoint_path, trainable_backbone_layers):
    moco_checkpoint = torch.load(checkpoint_path)

    # Rename MoCo pre-trained keys (taken from official MoCo repo):
    # https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L250
    state_dict = moco_checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # Retain only encoder_q up to before the embedding layer.
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # Remove prefix.
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        # Delete renamed or unused k.
        del state_dict[k]

    backbone = resnet50(weights=None)
    # Change the last ResNet block strides from (2, 2) to (1, 1)
    # to increase the output feature map from 14x14 to 28x28.
    backbone.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    backbone.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # Load the weights into ResNet-50
    backbone.load_state_dict(state_dict, strict=False)

    # Turn it into an FPN feature extractor.
    feature_extractor = _resnet_fpn_extractor(
        backbone, trainable_layers=trainable_backbone_layers, norm_layer=nn.BatchNorm2d
    )

    return feature_extractor


def SAMFasterRCNN(
    num_classes,
    checkpoint_path,
    trainable_backbone_layers=5,
    rpn_nms_thresh=1.01,
    bg_weight=1.0,
    num_bg_classes=1,
    **kwargs
):
    backbone = load_resnet50_fpn_with_moco(checkpoint_path, trainable_backbone_layers)
    model = FasterRCNNSAM(
        backbone,
        num_classes,
        min_size=[640, 672, 704, 736, 768, 800],
        max_size=1333,
        rpn_nms_thresh=rpn_nms_thresh,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=1000,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        box_score_thresh=0.05,
        bg_weight=bg_weight,
        num_bg_classes=num_bg_classes,
    )

    return model


if __name__ == "__main__":
    from data.datasets.fasterrcnn_data import ImageData
    from tqdm import tqdm
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torch.hub import load_state_dict_from_url

    device = "cpu"

    dataset = ImageData(
        "mask_features/val_32",
        "../datasets/coco/annotations/instances_val2017.json",
        "../datasets/coco",
        device,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageData.collate_fn)

    # Num classes should include background
    # ID 0 is reserved for background, so class IDs should start from 1.
    model = SAMFasterRCNN(81, "checkpoints/moco_v2_800ep_pretrain.pth.tar")
    # model = SAMRPN(90)  # For pre-trained faster_r_cnn weights

    # model = fasterrcnn_resnet50_fpn()
    # url = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    # state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    # model.load_state_dict(state_dict)
    model.to(device)
    # model.freeze()
    # print("total params:", sum(p.numel() for p in model.parameters()))
    # print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model.load_state_dict(state_dict)
    # model.eval()

    for batch in tqdm(dataloader):
        losses = model(batch)
        print(losses)
