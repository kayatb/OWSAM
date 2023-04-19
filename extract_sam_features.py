"""
Extract masks and corresponding mask features from SAM for each image in the COCO dataset.
Iterate over each image to obtain the masks and their features.
Then, save each mask+feature in a .pt PyTorch file. Merge all mask files belonging to a single image in a tar-file with
the same name as the input image. All these tar-files are saved in the same directory.
train_data/
    img0.tar
        mask_feature0.pt
        ...
        mask_featureN.pt
    ...
    imgN.tar
"""
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide

import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoDetection
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
        "-a",
        "--ann-file",
        required=True,
        help="Location of file containing the COCO annotations",
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Whether to use the GPU or not.")

    return parser.parse_args()


def show_anns(anns):
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


def extract_image_embedding(model, image, transform):
    """Extract the image embedding from the model encoder."""
    # Transform the image to the form expected by the model
    input_image = transform.apply_image(image)
    input_image = torch.as_tensor(input_image, device=model.device)
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Preprocess the image (e.g. normalization)
    input_image = model.preprocess(input_image)

    # Obtain the image features.
    img_embed = model.image_encoder(input_image)

    return img_embed


def extract_and_save_features(mask_generator, image, save_loc):
    """Extract the features for all masks found in the image and save them."""
    masks = mask_generator.generate(image)
    mask_feature = {"mask": masks["segmentation"], "feature": masks["mask_feature"]}


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.gpu else "cpu"
    dataset = CocoDetection(args.dataset_dir, args.ann_file)

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    transform = ResizeLongestSide(sam.image_encoder.img_size)

    for img, _ in dataset:
        # masks = mask_generator.generate(np.array(img))
        # print(len(masks))
        # print(masks[0].keys())

        # print(masks[0]["mask_feature"].shape)
        img_embed = extract_image_embedding(sam, np.array(img), transform)
        print(img_embed.shape)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(img)
        # show_anns(masks)
        # plt.axis("off")
        # plt.show()

        break
