"""
Extract the image embeddings from SAM for each image in the dataset. The image embeddings are the direct output of the
SAM image encoder. This is the heaviest computation part of the model, so pre-extracting the embeddings saves a
lot of time later on.
The image embeddings are saved in the provided directory with the same file name as the original image they're
generated from.
"""
from data.image_dataset import ImageDataset

from segment_anything import sam_model_registry

import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SAMFeatureExtractor",
        description="Extract mask features from SAM for given COCO dataset",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        required=True,
        help="Dir of the dataset images",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        required=True,
        help="Directory to save image embeddings",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Whether to use the GPU or not.")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")

    return parser.parse_args()


# Taken from the SAM demo notebooks.
def show_anns(anns):
    """Show the mask annotations superimposed on the original image."""
    import matplotlib.pyplot as plt

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def save_all_image_embeddings(model, data, save_dir):
    """Extract and save image embeddings from the model for all images in the dataset.
    Each embedding is saved in `save_dir` and has the same filename as the original image it came from."""
    os.makedirs(save_dir, exist_ok=True)

    print("Starting saving and extraction of the image embeddings...")
    print(f"All embeddings are saved in `{os.path.abspath(save_dir)}`")

    for batch in tqdm(data):
        with torch.no_grad():
            img_embeds = model.image_encoder(batch["img"])
            print(img_embeds.shape)
            print(batch["fname"])

        # Save each image embedding in the batch with the smallest data type possible and without any grads.
        for i in range(img_embeds.shape[0]):
            torch.save(img_embeds[i].detach().half(), os.path.join(save_dir, f"{batch['fname'][i]}.pt"))


# def extract_and_save_features(mask_generator, image, save_loc):
#     """Extract the features for all masks found in the image and save them."""
#     masks = mask_generator.generate(image)
#     mask_feature = {"mask": masks["segmentation"], "feature": masks["mask_feature"]}


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.gpu else "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)

    dataset = ImageDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    save_all_image_embeddings(sam, dataloader, args.save_dir)
