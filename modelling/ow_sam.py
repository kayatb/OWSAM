from modelling.sam_mask_generator import OWMaskDecoder
from segment_anything.modeling import ImageEncoderViT, PromptEncoder, Sam, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import batch_iterator, batched_mask_to_box, box_xyxy_to_xywh

import torch
from functools import partial
import numpy as np


def build_owsam(checkpoint="checkpoints/sam_vit_h_4b8939.pth"):
    encoder_embed_dim = 1280
    encoder_depth = 32
    encoder_num_heads = 16
    encoder_global_attn_indexes = [7, 15, 23, 31]
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = OWSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=OWMaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


class OWSam(Sam):
    @torch.no_grad()
    def forward(self, batched_input, multimask_output):
        self.transform = ResizeLongestSide(1024)  # TODO: move this to __init__
        image_embeddings = torch.stack([d["embed"] for d in batched_input], dim=0)
        batch_size = len(batched_input)
        num_preds = 1024  # TODO: don't hardcode this (point_coords.shape[0])
        device = image_embeddings.device

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # boxes = [torch.zeros((batch_size, num_preds, 4), device=device)]
            # mask_features = torch.zeros((batch_size, num_preds, 256), device=device)  # Don't hardcode mask feature dim
            # iou_predictions = -torch.ones((batch_size, num_preds), device=device)
            # masks = torch.zeros((batch_size, num_preds, image_record["original_size"]), device=device)
            batch_mask_features = []
            batch_iou_predictions = []
            batch_masks = []
            batch_boxes = []

            for p in batch_iterator(64, image_record["point_coords"]):  # TODO: don't hardcode points per batch
                point_coords = torch.cat(p)

                points = (
                    point_coords[:, None, :],  # points[:, None, :],
                    torch.ones(point_coords.shape[0], dtype=torch.int, device=device)[:, None],
                )

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                )
                low_res_masks, iou_predictions, mask_features = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                input_size = self.transform.get_preprocess_shape(
                    image_record["original_size"][0], image_record["original_size"][1], self.transform.target_length
                )
                masks = self.postprocess_masks(
                    low_res_masks,
                    input_size=input_size,
                    original_size=image_record["original_size"],
                )
                masks = masks > self.mask_threshold
                boxes = batched_mask_to_box(masks)

                batch_mask_features.append(mask_features)
                batch_iou_predictions.append(iou_predictions)
                batch_masks.append(masks)
                batch_boxes.append(boxes)

            outputs.append(
                {
                    "masks": batch_masks,
                    "iou_predictions": torch.stack(batch_iou_predictions),
                    "mask_features": torch.stack(batch_mask_features),
                    "boxes": torch.stack([box_xyxy_to_xywh(box) for box in batch_boxes]),
                }
            )
        return outputs


if __name__ == "__main__":
    from data.img_embeds_dataset import ImageEmbeds

    device = "cpu"

    sam = build_owsam(checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    dataset = ImageEmbeds(
        "img_embeds", "../datasets/coco/annotations/instances_val2017.json", sam.device, points_per_side=32
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    for batch in dataloader:
        outputs = sam(batch, multimask_output=False)
        print(len(outputs))
        for output in outputs:
            print(output["mask_features"].shape)
