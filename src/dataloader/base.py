import numpy as np
import abc
import os
import json
import random
from typing import Any, Tuple, List
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from collections import defaultdict
from src.utils.utils import ensure_path

class BaseDataset(VisionDataset, metaclass=abc.ABCMeta):
    def __init__(self, root, transform=None, **kwargs) -> None:
        super().__init__(root)
        if transform is None:
            assert "'transform' should be specified!"
        else:
            self.transform = transform 
        self.is_path = False
        self.helper_transform = None

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx) -> Tuple[Any, Any]:
        pass

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = (impath, int(label), classname)
                out.append(item)
            return out

        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def split_trainval(trainval_lst: List[Tuple], p_val: float=0.2):
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval_lst):
            label = item[1]
            tracker[label].append(idx)

        train_lst, val_lst = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval_lst[idx]
                if n < n_val:
                    val_lst.append(item)
                else:
                    train_lst.append(item)

        return train_lst, val_lst

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item[0]
                label = item[1]
                classname = item[2]
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    def read_and_split_data(self, image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = (im, y, c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    """Writes to a json file."""
    ensure_path(os.path.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))