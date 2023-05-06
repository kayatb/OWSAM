import torch
import gzip
import io
import os
from tqdm import tqdm

mask_feature_dir = "mask_features/train_all"
mask_feature_save_dir = "mask_features_new/train_all"
masks_save_dir = "masks/train_all"

for file in tqdm(os.listdir(mask_feature_dir)):
    with gzip.open(os.path.join(mask_feature_dir, file), "rb") as fp:
        mask_data = torch.load(io.BytesIO(fp.read()), map_location="cpu")

    # First, save the mask before deleting it.
    masks = []
    for m in mask_data:
        masks.append(m["segmentation"])

    buffer = io.BytesIO()
    torch.save(masks, buffer)

    with gzip.open(os.path.join(masks_save_dir, file), "wb") as fp:
        buffer.seek(0)
        fp.write(buffer.read())

    # Now delete the masks and point coords we no longer need.
    for mask in mask_data:
        del mask["point_coords"]
        del mask["segmentation"]
    # print(file, os.path.join(mask_feature_save_dir, f"{file[:-3]}.pt"))
    torch.save(mask_data, os.path.join(mask_feature_save_dir, f"{file[:-3]}.pt"))


# print(mask_data[0].keys())
# print(mask_data[0]["segmentation"].dtype)  # numpy array with dtype bool
# print(type(mask_data[0]["area"]))  # Int
# print(type(mask_data[0]["bbox"][0]))  # List of ints
# print(type(mask_data[0]["predicted_iou"]))  # Float
# print(type(mask_data[0]["point_coords"][0][0])) # List of lists of floats
# print(type(mask_data[0]["stability_score"])) # Float
# print(mask_data[0]["mask_feature"].dtype)  # Numpy array with dtype float32

# print(mask_data[0]["segmentation"])


# print(mask_data[0]["predicted_iou"])
