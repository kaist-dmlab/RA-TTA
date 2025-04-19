from __future__ import annotations
import numpy as np
import os
import PIL.Image
import re
from typing import Any, Tuple, List
from src.dataloader.base import BaseDataset

RESISC45_CLASSNAMES = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

class RESISC45(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, "resisc45")
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "test_splits.txt")

        if os.path.exists(self.split_path):
            with open(os.path.join(self.dataset_dir, f'test_splits.txt')) as f:
                test_splits = f.readlines()
                test_splits = [line.strip() for line in test_splits]
                test_lst = self.read_split(test_splits, self.image_dir)
        else:
            train_lst, val_lst, test_lst = self.read_and_split_data(self.image_dir)
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

    @staticmethod
    def read_split(test_splits, path_prefix):
        out = []
        for impath in test_splits:
            classname = re.sub(r'_\d+\.jpg$', '', impath)
            impath = os.path.join(path_prefix, classname, impath)
            label = RESISC45_CLASSNAMES.index(classname)
            item = (impath, int(label), classname)
            out.append(item)
        return out

