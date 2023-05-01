from data.img_embeds_dataset import ImageEmbeds
from data.image_dataset import ImageDataset
from modelling.sam_mask_generator import OWSamMaskGenerator

from segment_anything import sam_model_registry

import argparse
import os
import io

# import tarfile
import gzip

# import numpy as np
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
        "--save-dir",
        required=True,
        help="Directory to save extracted features",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Whether to use the GPU or not.")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")

    return parser.parse_args()


def save_all_masks(mask_generator, dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(dataloader):
        # batch contains {"embed": embeds, "original_size": original_sizes, "img_id": img_ids}
        batch_size = batch["embed"].shape[0]
        for i in range(batch_size):
            output = mask_generator.generate(batch["embed"][i].unsqueeze(0), batch["original_size"][i])
            # output is a list of dicts (one for each mask generated for that image), with keys:
            # 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'mask_feature'
            # For each image, save a tar file with all the masks. The tar file has the img_id as filename.
            # with tarfile.open(f"{batch["img_id"]}.tar.gz", "w:gz") as fp:
            buffer = io.BytesIO()
            torch.save(output, buffer)

            with gzip.open(os.path.join(save_dir, f"{batch['img_id'][i]}.gz"), "wb") as fp:
                buffer.seek(0)
                fp.write(buffer.read())


def save_all_masks_without_embeds(model, mask_generator, dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(dataloader):
        img_embeds = model.image_encoder(batch["img"].to(model.device))
        # batch contains {"img", "orig_h", "orig_w", "fname"}
        batch_size = batch["img"].shape[0]
        for i in range(batch_size):
            img_id = int(batch["fname"][i])
            orig_size = (batch["orig_w"][i].item(), batch["orig_h"][i].item())
            output = mask_generator.generate(img_embeds[i].unsqueeze(0), orig_size)
            # output is a list of dicts (one for each mask generated for that image), with keys:
            # 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'mask_feature'
            # For each image, save a tar file with all the masks. The tar file has the img_id as filename.
            # with tarfile.open(f"{batch["img_id"]}.tar.gz", "w:gz") as fp:
            buffer = io.BytesIO()
            torch.save(output, buffer)

            with gzip.open(os.path.join(save_dir, f"{img_id}.gz"), "wb") as fp:
                buffer.seek(0)
                fp.write(buffer.read())


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.gpu else "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")

    mask_generator = OWSamMaskGenerator(sam)
    sam.to(device=device)  # Should be done after the decoder has been changed by the mask generator

    if args.embed_dir:
        print("EMBEDDING")
        dataset = ImageEmbeds(args.embed_dir, device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ImageEmbeds.collate_fn)
        save_all_masks(mask_generator, dataloader, args.save_dir)
    else:
        print("IMAGES")
        dataset = ImageDataset(args.embed_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        save_all_masks_without_embeds(sam, mask_generator, dataloader, args.save_dir)
