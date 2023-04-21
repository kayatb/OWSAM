"""
Extract the image embeddings from SAM for each image in the dataset. The image embeddings are the direct output of the
SAM image encoder. This is the heaviest computation part of the model, so pre-extracting the embeddings saves a
lot of time later on.
The image embeddings are saved in the provided directory with the same file name as the original image they're
generated from.
"""
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


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

    return parser.parse_args()


# Taken from the SAM demo notebooks.
def show_anns(anns):
    """Show the mask annotations superimposed on the original image."""
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


# Taken and adapted from the SAM Predictor.
def extract_image_embedding(model, image, transform):
    """Extract the image embedding from the model encoder."""
    # Transform the image to the form expected by the model
    input_image = transform.apply_image(image)
    input_image = torch.as_tensor(input_image, device=model.device)
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

    # with torch.no_grad():
    #     # Preprocess the image (e.g. normalization)
    #     input_image = model.preprocess(input_image)

    #     # Obtain the image features.
    #     img_embed = model.image_encoder(input_image)

    # return img_embed


def save_all_image_embeddings(model, dataset_dir, save_dir):
    """Extract and save image embeddings from the model for all images in the dataset.
    Each embedding is saved in `save_dir` and has the same filename as the original image it came from."""
    os.makedirs(save_dir, exist_ok=True)
    print("Starting saving and extraction of the image embeddings...")
    print(f"All embeddings are saved in `{os.path.abspath(save_dir)}`")

    transform = ResizeLongestSide(sam.image_encoder.img_size)

    for img_file in tqdm(os.listdir(dataset_dir)):
        img = Image.open(os.path.join(dataset_dir, img_file)).convert("RGB")
        try:
            img_embed = extract_image_embedding(model, np.array(img), transform)
        except RuntimeError:
            print(img_file)
        # Save the tensor with the smallest data type possible and without any grads.
        # torch.save(img_embed.squeeze().detach().half(), os.path.join(save_dir, f"{img_file[:-4]}.pt"))


# def extract_and_save_features(mask_generator, image, save_loc):
#     """Extract the features for all masks found in the image and save them."""
#     masks = mask_generator.generate(image)
#     mask_feature = {"mask": masks["segmentation"], "feature": masks["mask_feature"]}


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.gpu else "cpu"

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)

    save_all_image_embeddings(sam, args.dataset_dir, args.save_dir)
