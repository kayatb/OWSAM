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

import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoDetection


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


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if args.gpu else "cpu"
    dataset = CocoDetection(args.dataset_dir, args.ann_file)

    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for img, _ in dataset:
        masks = mask_generator.generate(np.array(img))
        print(len(masks))
        print(masks[0].keys())

        print(masks[0]["mask_feature"].shape)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(img)
        # show_anns(masks)
        # plt.axis("off")
        # plt.show()

        break
