from modelling.sam_mask_generator import OWMaskDecoder
from segment_anything.modeling import ImageEncoderViT, PromptEncoder, Sam, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide

import torch
from functools import partial


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

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (
                    image_record["point_coords"][:, None, :],
                    image_record["point_labels"][:, None],
                )
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
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
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "mask_features": mask_features,
                }
            )
        return outputs


if __name__ == "__main__":
    from data.img_embeds_dataset import ImageEmbeds

    device = "cpu"

    sam = build_owsam(checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)

    dataset = ImageEmbeds(
        "img_embeds", "../datasets/coco/annotations/instances_val2017.json", sam.device, points_per_side=2
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=ImageEmbeds.collate_fn)

    for batch in dataloader:
        outputs = sam(batch, multimask_output=False)
        print(len(outputs))
        for output in outputs:
            print(output["masks"].shape)
