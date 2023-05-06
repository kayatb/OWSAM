from data.img_embeds_dataset import ImageEmbeds
from data.image_dataset import ImageDataset
from modelling.sam_mask_generator import OWSamMaskGenerator

from segment_anything import sam_model_registry

import argparse
import os
import io

import gzip

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SAMFeatureExtractor",
        description="Extract mask features from SAM for given COCO dataset",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", "--embed-dir", help="Dir of the pre-extracted image embeddings")
    group.add_argument("-i", "--image-dir", help="Dir of the images.")
    parser.add_argument(
        "-s",
        "--save-feat-dir",
        required=True,
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "-m",
        "--save-mask-dir",
        required=True,
        help="Directory to save extracted masks",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Whether to use the GPU or not.")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Whether to resume extraction where it left off.")

    return parser.parse_args()


def save_all_masks(mask_generator, dataloader, save_feat_dir, save_mask_dir, resume=False, model=None):
    # If a model is given, the image embeddings are calculated from scratch instead of using pre-extracted ones.
    os.makedirs(save_feat_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    for batch in tqdm(dataloader):
        if model:
            img_embeds = model.image_encoder(batch["img"].to(model.device))
            batch_size = batch["img"].shape[0]

        else:
            img_embeds = batch["embed"]
            batch_size = batch["embed"].shape[0]

        # batch contains {"embed": embeds, "original_size": original_sizes, "img_id": img_ids}
        for i in range(batch_size):
            if model:
                orig_size = (batch["orig_w"][i].item(), batch["orig_h"][i].item())
                img_id = int(batch["fname"][i])
            else:
                orig_size = batch["original_size"][i]
                img_id = batch["img_id"][i]

            save_path_feat = os.path.join(save_feat_dir, f"{img_id}.pt")
            save_path_mask = os.path.join(save_mask_dir, f"{img_id}.gz")

            # If resuming extraction, skip files that already exist.
            # TODO: the image embedding extraction is not skipped, which slows things down unnecessarily.
            if resume and os.path.isfile(save_path_feat) and os.path.isfile(save_path_mask):
                continue

            # output is a list of dicts (one for each mask generated for that image), with keys:
            # 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'mask_feature'
            output = mask_generator.generate(img_embeds[i].unsqueeze(0), orig_size)

            # First, save the mask before deleting it.
            masks = []
            for m in output:
                masks.append(m["segmentation"])

            # For each image, save a gzipped file with all the masks and the img_id as filename.
            buffer = io.BytesIO()
            torch.save(masks, buffer)

            with gzip.open(save_path_mask, "wb") as fp:
                buffer.seek(0)
                fp.write(buffer.read())

            # Now delete the masks and point coords we no longer need.
            for mask in output:
                del mask["point_coords"]
                del mask["segmentation"]

            torch.save(output, save_path_feat)


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.gpu else "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")

    mask_generator = OWSamMaskGenerator(sam)
    sam.to(device=device)  # Should be done after the decoder has been changed by the mask generator

    if args.embed_dir:
        print("Start extraction of SAM masks from pre-calculated image embeddings...")
        dataset = ImageEmbeds(args.embed_dir, device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ImageEmbeds.collate_fn)
        model = None
    else:
        print("Start extraction of SAM masks without pre-calculated image embeddings...")
        dataset = ImageDataset(args.image_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        model = sam

    save_all_masks(mask_generator, dataloader, args.save_feat_dir, args.save_mask_dir, resume=args.resume, model=model)
