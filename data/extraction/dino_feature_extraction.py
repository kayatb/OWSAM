from data.datasets.bbox_crop_dataset import BBoxCropDataset

import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SAMFeatureExtractor",
        description="Extract mask features from SAM for given COCO dataset",
    )
    parser.add_argument("-b", "--box-dir", required=True, help="Dir of the pre-extracted image embeddings")
    parser.add_argument("-i", "--image-dir", required=True, help="Dir of the images.")
    parser.add_argument(
        "-s",
        "--save-dir",
        required=True,
        help="Directory to save extracted features",
    )

    parser.add_argument("-g", "--gpu", action="store_true", help="Whether to use the GPU or not.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume extraction where it left off.")

    return parser.parse_args()


def save_dino_features(model, dataloader, save_dir, device):
    print("Starting DINO feature extraction")
    print(f"All features are saved in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(dataloader):
        crops = batch["crops"].squeeze().to(device)
        if crops.dim() < 4:  # In case only a single bbox in this image, add a batch dim.
            crops = crops.unsqueeze(0)

        with torch.no_grad():
            features = model(crops)

        torch.save(features.half(), os.path.join(save_dir, f"{batch['img_id'].item()}.pt"))
        break


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if args.gpu else "cpu"

    dataset = BBoxCropDataset(args.image_dir, args.box_dir)
    # Batch size per image must be 1, since number of boxes differ between images and we don't want padding here.
    dataloader = DataLoader(dataset, batch_size=1)

    # dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    # dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    dinov2_vitl14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)
    # dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").to(device)

    save_dino_features(dinov2_vitl14, dataloader, args.save_dir, device)
