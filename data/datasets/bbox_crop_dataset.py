from utils.misc import crop_bboxes_from_img, box_xywh_to_xyxy

import os
import torch
import torchvision.transforms as T
from PIL import Image


class BBoxCropDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, box_dir):
        self.img_dir = img_dir
        self.files = os.listdir(self.img_dir)
        self.box_dir = box_dir

    def __getitem__(self, idx):
        img_id = int(os.path.splitext(self.files[idx])[0])

        mask_data = torch.load(os.path.join(self.box_dir, f"{img_id}.pt"))

        # Skip boxes with zero height or width.
        boxes = torch.as_tensor([d["bbox"] for d in mask_data if d["bbox"][2] > 0 and d["bbox"][3] > 0])
        boxes = box_xywh_to_xyxy(boxes)  # Convert to expected format for cropping.

        with Image.open(os.path.join(self.img_dir, self.files[idx])) as img:
            img = img.convert("RGB")
            crops = crop_bboxes_from_img(img, boxes)

        for i, crop in enumerate(crops):
            crops[i] = self.preprocess(crop)

        return {"crops": torch.stack(crops), "img_id": img_id}

    def __len__(self):
        return len(self.files)

    def preprocess(self, img):
        transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img = transform(img)[:3]  # .unsqueeze(0)

        return img


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = BBoxCropDataset("../datasets/coco/val2017", "mask_features/val_all")
    # print(dataset[0][0].shape)

    # print(len(dataset[442]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for batch in tqdm(dataloader):
        print(batch.keys())
        break
