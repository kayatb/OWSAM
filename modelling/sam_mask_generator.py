from segment_anything import SamAutomaticMaskGenerator

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area

from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamMaskGenerator(SamAutomaticMaskGenerator):
    # def __init__(self):
    def generate(self, image, embedding):
        # Generate masks
        mask_data = self._generate_masks(image, embedding)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                # "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
                "mask_feature": mask_data["mask_features"][idx],
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image, embedding):
        img_size = image.shape[:2]
        img_w, img_h = img_size

        self.predictor.set_image(image)  # TODO: change predictor to take embedding instead of image.

        # Get points for the image.
        points_scale = np.array(img_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        # Generate masks for the image in batches.
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, img_size, [0, 0, img_h, img_w], img_size
            )  # TODO: adapt for outputting mask features
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates with non-maximum suppression.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        data.to_numpy()

        return data


if __name__ == "__main__":
    from segment_anything import sam_model_registry
    from PIL import Image

    device = "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    mask_generator = SamMaskGenerator(sam)

    image = Image.open("input.jpg")
    embedding = torch.Tensor(torch.Size([256, 64, 64]))
    masks = mask_generator.generate(np.array(image), embedding)
    for mask in masks:
        print(mask["bbox"])
