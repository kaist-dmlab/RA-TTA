from __future__ import annotations
import numpy as np
import os
import PIL.Image
from typing import Any, Tuple, List
from src.dataloader.base import BaseDataset, read_json

CALTECH256_CLASSNAMES = ['ak47',
 'american-flag',
 'backpack',
 'baseball-bat',
 'baseball-glove',
 'basketball-hoop',
 'bat',
 'bathtub',
 'bear',
 'beer-mug',
 'billiards',
 'binoculars',
 'birdbath',
 'blimp',
 'bonsai',
 'boom-box',
 'bowling-ball',
 'bowling-pin',
 'boxing-glove',
 'brain',
 'breadmaker',
 'buddha',
 'bulldozer',
 'butterfly',
 'cactus',
 'cake',
 'calculator',
 'camel',
 'cannon',
 'canoe',
 'car-tire',
 'cartman',
 'cd',
 'centipede',
 'cereal-box',
 'chandelier',
 'chess-board',
 'chimp',
 'chopsticks',
 'cockroach',
 'coffee-mug',
 'coffin',
 'coin',
 'comet',
 'computer-keyboard',
 'computer-monitor',
 'computer-mouse',
 'conch',
 'cormorant',
 'covered-wagon',
 'cowboy-hat',
 'crab',
 'desk-globe',
 'diamond-ring',
 'dice',
 'dog',
 'dolphin',
 'doorknob',
 'drinking-straw',
 'duck',
 'dumb-bell',
 'eiffel-tower',
 'electric-guitar',
 'elephant',
 'elk',
 'ewer',
 'eyeglasses',
 'fern',
 'fighter-jet',
 'fire-extinguisher',
 'fire-hydrant',
 'fire-truck',
 'fireworks',
 'flashlight',
 'floppy-disk',
 'football-helmet',
 'french-horn',
 'fried-egg',
 'frisbee',
 'frog',
 'frying-pan',
 'galaxy',
 'gas-pump',
 'giraffe',
 'goat',
 'golden-gate-bridge',
 'goldfish',
 'golf-ball',
 'goose',
 'gorilla',
 'grand-piano',
 'grapes',
 'grasshopper',
 'guitar-pick',
 'hamburger',
 'hammock',
 'harmonica',
 'harp',
 'harpsichord',
 'hawksbill',
 'head-phones',
 'helicopter',
 'hibiscus',
 'homer-simpson',
 'horse',
 'horseshoe-crab',
 'hot-air-balloon',
 'hot-dog',
 'hot-tub',
 'hourglass',
 'house-fly',
 'human-skeleton',
 'hummingbird',
 'ibis',
 'ice-cream-cone',
 'iguana',
 'ipod',
 'iris',
 'jesus-christ',
 'joy-stick',
 'kangaroo',
 'kayak',
 'ketch',
 'killer-whale',
 'knife',
 'ladder',
 'laptop',
 'lathe',
 'leopard',
 'license-plate',
 'lightbulb',
 'light-house',
 'lightning',
 'llama',
 'mailbox',
 'mandolin',
 'mars',
 'mattress',
 'megaphone',
 'menorah',
 'microscope',
 'microwave',
 'minaret',
 'minotaur',
 'motorbike',
 'mountain-bike',
 'mushroom',
 'mussels',
 'necktie',
 'octopus',
 'ostrich',
 'owl',
 'palm-pilot',
 'palm-tree',
 'paperclip',
 'paper-shredder',
 'pci-card',
 'penguin',
 'people',
 'pez-dispenser',
 'photocopier',
 'picnic-table',
 'playing-card',
 'porcupine',
 'pram',
 'praying-mantis',
 'pyramid',
 'raccoon',
 'radio-telescope',
 'rainbow',
 'refrigerator',
 'revolver',
 'rifle',
 'rotary-phone',
 'roulette-wheel',
 'saddle',
 'saturn',
 'school-bus',
 'scorpion',
 'screwdriver',
 'segway',
 'self-propelled-lawn-mower',
 'sextant',
 'sheet-music',
 'skateboard',
 'skunk',
 'skyscraper',
 'smokestack',
 'snail',
 'snake',
 'sneaker',
 'snowmobile',
 'soccer-ball',
 'socks',
 'soda-can',
 'spaghetti',
 'speed-boat',
 'spider',
 'spoon',
 'stained-glass',
 'starfish',
 'steering-wheel',
 'stirrups',
 'sunflower',
 'superman',
 'sushi',
 'swan',
 'swiss-army-knife',
 'sword',
 'syringe',
 'tambourine',
 'teapot',
 'teddy-bear',
 'teepee',
 'telephone-box',
 'tennis-ball',
 'tennis-court',
 'tennis-racket',
 'theodolite',
 'toaster',
 'tomato',
 'tombstone',
 'top-hat',
 'touring-bike',
 'tower-pisa',
 'traffic-light',
 'treadmill',
 'triceratops',
 'tricycle',
 'trilobite',
 'tripod',
 't-shirt',
 'tuning-fork',
 'tweezer',
 'umbrella',
 'unicorn',
 'vcr',
 'video-projector',
 'washing-machine',
 'watch',
 'waterfall',
 'watermelon',
 'welding-mask',
 'wheelbarrow',
 'windmill',
 'wine-bottle',
 'xylophone',
 'yarmulke',
 'yo-yo',
 'zebra',
 'airplane',
 'car-side',
 'greyhound',
 'tennis-shoes',
 'toad',
 'clutter']

class Caltech256(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, 'caltech-256')
        self.image_dir = os.path.join(self.dataset_dir, "256_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_Caltech256.json")
        
        if os.path.exists(self.split_path):
            train_lst, val_lst, test_lst = self.read_split(self.split_path, self.image_dir)
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

        if label > 252:
            label = label - 1
            
        return image, label, class_name
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                if 'faces-easy' in classname:
                    continue
                if classname[-4:] == '-101':
                    classname = classname.split('.')[1][:-4]
                else:
                    classname = classname.split('.')[1]
                impath = os.path.join(path_prefix, impath)
                item = (impath, int(label), classname)
                out.append(item)
            return out

        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
