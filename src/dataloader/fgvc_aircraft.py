from __future__ import annotations
import numpy as np
import os
import PIL.Image
from typing import Any, Tuple
from src.dataloader.base import BaseDataset

AIRCRAFT_CLASSNAMES = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', \
            '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Panavia Tornado', 'Tu-134', 'Tu-154', 'Yak-42']


class FGVCAircraft(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, 'fgvc_aircraft')
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}
        
        # train_lst = self.read_data(cname2lab, "images_variant_train.txt")
        # val_lst = self.read_data(cname2lab, "images_variant_val.txt")
        test_lst = self.read_data(cname2lab, "images_variant_test.txt")
        self.data = np.array([data for data, _, _ in test_lst])
        self.targets = np.array([label for _, label, _ in test_lst])
        self.cls_names = np.array([name for _, _, name in test_lst])
            
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_path, label, class_name = self.data[idx], self.targets[idx], self.cls_names[idx]
        image = PIL.Image.open(image_path).convert("RGB")
    
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        if self.is_path:
            return image, label, class_name, image_path
        else:
            return image, label, class_name
            

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = (impath, label, classname)
                items.append(item)

        return items