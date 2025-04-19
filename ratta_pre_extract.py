import warnings
warnings.filterwarnings("ignore")
import os
import torch
import clip
import argparse
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from src.utils.utils import ensure_path, set_seed

parser = argparse.ArgumentParser() 
parser.add_argument('--root', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--net_name', type=str, default='ViT-B/16')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int)
parser.add_argument('--max_threads', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='./test_embedding')

# RA-TTA
parser.add_argument('--augmentation_size', type=int, default=100)
args = parser.parse_args()
device = f'cuda:{args.gpu}'
set_seed(args.seed)

dataset  = args.dataset
net_name = args.net_name
args.save_dir = os.path.join(args.save_dir, net_name.replace('/','_'), str(args.seed))
ensure_path(args.save_dir)

print('RATTA', os.path.join(args.save_dir, dataset))
print(f"Seed: {args.seed}")

# RATTA config
AS = args.augmentation_size
print(f"Augmentation size: {AS}")

from src.dataloader.stanford_cars import CARS_CLASSNAMES, StanfordCars
from src.dataloader.oxford_flowers import FLOWERS102_CLASSNAMES, OxfordFlowers
from src.dataloader.food101 import FOOD101_CLASSNAMES, Food101
from src.dataloader.oxford_pets import PETS_CLASSNAMES, OxfordPets
from src.dataloader.dtd import DTD_CLASSNAMES, DTD
from src.dataloader.cub200 import CUB_CLASSNAMES, CUB200
from src.dataloader.fgvc_aircraft import AIRCRAFT_CLASSNAMES, FGVCAircraft
from src.dataloader.resisc45 import RESISC45_CLASSNAMES, RESISC45
from src.dataloader.sun397 import SUN397_CLASSNAMES, SUN397
from src.dataloader.ucf101 import UCF101_CLASSNAMES, UCF101
from src.dataloader.caltech101 import CALTECH101_CLASSNAMES, Caltech101
from src.dataloader.caltech256 import CALTECH256_CLASSNAMES, Caltech256
from src.dataloader.utils import _transform
import torchvision.datasets as datasets

_CLASSNAMES = {
        'stanford_cars': CARS_CLASSNAMES,
        'flowers102': FLOWERS102_CLASSNAMES,
        'food101': FOOD101_CLASSNAMES,
        'oxford_pets': PETS_CLASSNAMES,
        'dtd': DTD_CLASSNAMES,
        'cub200': CUB_CLASSNAMES,
        'fgvc_aircraft': AIRCRAFT_CLASSNAMES,
        'resisc45': RESISC45_CLASSNAMES,
		'sun397': SUN397_CLASSNAMES,
		'ucf101': UCF101_CLASSNAMES,
		'caltech101': CALTECH101_CLASSNAMES,
		'caltech256': CALTECH256_CLASSNAMES,
        }

_DATASETS = {
        'stanford_cars': StanfordCars,
        'flowers102': OxfordFlowers,
        'food101': Food101,
        'oxford_pets': OxfordPets,
        'dtd': DTD,
        'cub200': CUB200,
        'fgvc_aircraft': FGVCAircraft,
        'resisc45': RESISC45,
		'sun397': SUN397,
		'ucf101': UCF101,
		'caltech101': Caltech101,
		'caltech256': Caltech256,
        }

IMAGE_SIZE = 336 if '336' in net_name else 224
CLASSNAMES = [i.replace('_', ' ') for i in _CLASSNAMES[args.dataset]]
CHANNEL_SIZE = 3

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def aug_crop(image, preprocess):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed

class CropAugmenter(object):
    def __init__(self, base_transform, n_views):
        self.base_transform = base_transform
        self.preprocess = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                                ),
                            ])
        self.n_views = n_views

    def __call__(self, x):
        image = self.base_transform(x)
        views = [aug_crop(x, self.preprocess) for _ in range(self.n_views)]
        return image, views

def create_aug_dataset_class(dataset):
    # Assuming _DATASETS is a dictionary or module with dataset classes
    parent_class = _DATASETS[dataset]
    if dataset == 'caltech256':
        class AugDataset(parent_class):
            def __init__(self, root, transform=None, target_transform=None):
                super().__init__(root=root)
                self.transform = transform
                self.target_transform = target_transform

            def __getitem__(self, idx):
                if isinstance(self.data, np.ndarray):
                    image_path, label = self.data[idx], self.targets[idx]
                    image = Image.open(image_path).convert("RGB")
               
                if self.transform:
                    image, aug_images = self.transform(image)
                if self.target_transform:
                    label = self.target_transform(label)
                
                if label > 252:
                    label = label - 1
                return image, aug_images, label
    else:
        class AugDataset(parent_class):
            def __init__(self, root, transform=None, target_transform=None):
                super().__init__(root=root)
                self.transform = transform
                self.target_transform = target_transform

            def __getitem__(self, idx):
                if isinstance(self.data, np.ndarray):
                    image_path, label = self.data[idx], self.targets[idx]
                    image = Image.open(image_path).convert("RGB")
                else:
                    sample = self.data.iloc[idx]
                    image_path = os.path.join(self.root, self.base_folder, sample.filepath)
                    image = self.loader(image_path)
                    label = sample.target - 1  # Targets start at 1 by default, so shift to 0
                
                if self.transform:
                    image, aug_images = self.transform(image)
                if self.target_transform:
                    label = self.target_transform(label)
                    
                return image, aug_images, label
    return AugDataset

image_transform = _transform(IMAGE_SIZE)
augmenter = CropAugmenter(base_transform=image_transform, n_views=AS)
tst_dataset = create_aug_dataset_class(dataset)
tst_dataset = tst_dataset(root=args.root, transform=augmenter)
tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

clip_model, _ = clip.load(net_name, 
                          device=device) 

clip_model.eval() # returned clip model is in evaluation mode.
clip_model.requires_grad_(False)
logit_scale = clip_model.logit_scale.exp()

pre_extracted_embeddings = []
with torch.no_grad():
    for i, (x, aug_x, y) in enumerate(tqdm(tst_loader)):
        tst_x = x.to(device=device)
        aug_x = torch.stack(aug_x, dim=0) # torch.Size([BS, AS, 3, 224, 224])
        aug_x = aug_x.permute(1,0,2,3,4).reshape(-1, CHANNEL_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        aug_x = aug_x.to(device)
        
        tst_x_embeds = clip_model.encode_image(tst_x)
        tst_x_embeds = tst_x_embeds / tst_x_embeds.norm(dim=-1, keepdim=True)

        aug_x_embeds = clip_model.encode_image(aug_x)
        aug_x_embeds = aug_x_embeds / aug_x_embeds.norm(dim=-1, keepdim=True)

        pre_extracted_embeddings.append((tst_x_embeds.cpu(), aug_x_embeds.cpu(), y))

save_path = os.path.join(args.save_dir, f"{args.dataset}.pth")
torch.save(pre_extracted_embeddings, save_path)
print(f"Successfully saved image features to [{save_path}]")
