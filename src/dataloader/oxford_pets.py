from __future__ import annotations

import numpy as np
import os
import PIL.Image
from typing import Any, Tuple, List
from src.dataloader.base import BaseDataset

PETS_CLASSNAMES = ['abyssinian', 'american_bulldog', 'american_pit_bull_terrier', \
                   'basset_hound', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', \
                   'british_shorthair', 'chihuahua', 'egyptian_mau', 'english_cocker_spaniel', \
                   'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', \
                   'japanese_chin', 'keeshond', 'leonberger', 'maine_coon', 'miniature_pinscher', \
                   'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian_blue', 'saint_bernard', \
                   'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese', 'sphynx', 'staffordshire_bull_terrier', \
                   'wheaten_terrier', 'yorkshire_terrier']

class OxfordPets(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, 'oxford_pets')
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        if os.path.exists(self.split_path):
            train_lst, val_lst, test_lst = self.read_split(self.split_path, self.image_dir)
        else:
            trainval_lst = self.read_data(split_file="trainval.txt")
            test_lst = self.read_data(split_file="test.txt")
            train_lst, val_lst = self.split_trainval(trainval_lst)
            self.save_split(train_lst, val_lst, test_lst, self.split_path, self.image_dir)

        self.data = np.array([data for data, _, _ in test_lst])
        self.targets = np.array([label for _, label, _ in test_lst])
        self.cls_names = np.array([name for _, _, name in test_lst])
            
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label, class_name = self.data[idx], self.targets[idx], self.cls_names[idx]
        image = PIL.Image.open(image_file).convert("RGB")
    
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, class_name

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1 
                item = (impath, label, breed)
                items.append(item)

        return items