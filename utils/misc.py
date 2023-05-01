import utils.coco_ids_without_anns as empty_ids

import os


def filter_empty_imgs(files):
    """Filter out image IDs for images that only contain the background class and
    thus have no annotations."""
    filtered_files = []
    for file in files:
        id = int(os.path.splitext(file)[0])
        if id in empty_ids.train_ids or id in empty_ids.val_ids:
            continue
        filtered_files.append(file)
    return filtered_files
