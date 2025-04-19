from __future__ import annotations

import numpy as np
import os
import re
import PIL.Image
from typing import Any, Tuple, List
from src.dataloader.base import BaseDataset

UCF101_CLASSNAMES = ['Apply_Eye_Makeup', 'Apply_Lipstick', 'Archery', 'Baby_Crawling', 'Balance_Beam', 'Band_Marching', 'Baseball_Pitch', 'Basketball', 'Basketball_Dunk', 'Bench_Press', 'Biking', 'Billiards', 'Blow_Dry_Hair', 'Blowing_Candles', 'Body_Weight_Squats', 'Bowling', 'Boxing_Punching_Bag', 'Boxing_Speed_Bag', 'Breast_Stroke', 'Brushing_Teeth', 'Clean_And_Jerk', 'Cliff_Diving', 'Cricket_Bowling', 'Cricket_Shot', 'Cutting_In_Kitchen', 'Diving', 'Drumming', 'Fencing', 'Field_Hockey_Penalty', 'Floor_Gymnastics', 'Frisbee_Catch', 'Front_Crawl', 'Golf_Swing', 'Haircut', 'Hammering', 'Hammer_Throw', 'Handstand_Pushups', 'Handstand_Walking', 'Head_Massage', 'High_Jump', 'Horse_Race', 'Horse_Riding', 'Hula_Hoop', 'Ice_Dancing', 'Javelin_Throw', 'Juggling_Balls', 'Jumping_Jack', 'Jump_Rope', 'Kayaking', 'Knitting', 'Long_Jump', 'Lunges', 'Military_Parade', 'Mixing', 'Mopping_Floor', 'Nunchucks', 'Parallel_Bars', 'Pizza_Tossing', 'Playing_Cello', 'Playing_Daf', 'Playing_Dhol', 'Playing_Flute', 'Playing_Guitar', 'Playing_Piano', 'Playing_Sitar', 'Playing_Tabla', 'Playing_Violin', 'Pole_Vault', 'Pommel_Horse', 'Pull_Ups', 'Punch', 'Push_Ups', 'Rafting', 'Rock_Climbing_Indoor', 'Rope_Climbing', 'Rowing', 'Salsa_Spin', 'Shaving_Beard', 'Shotput', 'Skate_Boarding', 'Skiing', 'Skijet', 'Sky_Diving', 'Soccer_Juggling', 'Soccer_Penalty', 'Still_Rings', 'Sumo_Wrestling', 'Surfing', 'Swing', 'Table_Tennis_Shot', 'Tai_Chi', 'Tennis_Swing', 'Throw_Discus', 'Trampoline_Jumping', 'Typing', 'Uneven_Bars', 'Volleyball_Spiking', 'Walking_With_Dog', 'Wall_Pushups', 'Writing_On_Board', 'Yo_Yo']

class UCF101(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, 'ucf101')
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")

        if os.path.exists(self.split_path):
            train_lst, val_lst, test_lst = self.read_split(self.split_path, self.image_dir)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # conver to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test_lst = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            train_lst, val_lst = self.split_trainval(trainval)
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

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = (impath, label, renamed_action)
                items.append(item)

        return items