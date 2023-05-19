# import utils.coco_ids_without_anns as empty_ids
import utils.coco_ids_without_anns
import utils.lvis_ids_without_anns

import pickle
import torch
import torch.distributed as dist
import numpy as np


def labels_to_onehot(labels, num_classes):
    """Labels to one-hot encoding."""
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)


def onehot_to_labels(onehot):
    """One-hot encoding to labels."""
    return torch.argmax(onehot, dim=-1)


def filter_empty_imgs(ids, dataset="coco"):
    """Filter out image IDs for images that only contain the background class and
    thus have no annotations."""
    if dataset == "coco":
        empty_train_ids = utils.coco_ids_without_anns.train_ids
        empty_val_ids = utils.coco_ids_without_anns.val_ids
    elif dataset == "lvis":
        empty_train_ids = utils.lvis_ids_without_anns.train_ids
        empty_val_ids = utils.lvis_ids_without_anns.val_ids
    else:
        raise ValueError(f"Unkown dataset `{dataset}` given. Choose either `lvis` or `coco`.")

    filtered_ids = []
    for id in ids:
        # id = int(os.path.splitext(file)[0])
        if id in empty_train_ids or id in empty_val_ids:
            continue
        filtered_ids.append(id)
    return filtered_ids


def crop_bboxes_from_img(img, boxes):
    """Crop the bounding boxes from the image.
    img is a PIL Image. Boxes are expected in (x, y, w, h) format."""
    crops = []

    for box in boxes:
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        crop = img.crop(np.array(box))
        crops.append(crop)

    return crops


def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [x, y, (x + w), (y + h)]

    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    xmin, ymin, xmax, ymax = x.unbind(-1)
    b = [xmin, ymin, xmax - xmin, ymax - ymin]

    return torch.stack(b, dim=-1)


# ===== Copy-paste from DETR =====
# https://github.com/facebookresearch/detr/blob/main/util/misc.py


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# ===== End of copy-paste =====
