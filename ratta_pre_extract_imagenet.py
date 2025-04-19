import warnings
warnings.filterwarnings("ignore")
import os
import torch
import clip
import argparse
import time
from tqdm import tqdm
from torchvision import transforms
from src.utils.utils import ensure_path, set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--net_name', type=str, default='ViT-B/16')
parser.add_argument('--seed', type=int, default=42)
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

from src.dataloader.imagenet import IN_CLASSNAMES
from src.dataloader.imagenet_rendition import IN_R_CLASSNAMES
from src.dataloader.imagenet_adversarial import IN_A_CLASSNAMES
from src.dataloader.imagenet_v2 import IN_V2_CLASSNAMES
from src.dataloader.imagenet_sketch import IN_SKETCH_CLASSNAMES
from src.dataloader.utils import _transform
import torchvision.datasets as datasets

_CLASSNAMES = {
        'imagenet_1k': IN_CLASSNAMES,
        'imagenet_rendition': IN_R_CLASSNAMES, 
        'imagenet_adversarial': IN_A_CLASSNAMES,
        'imagenet_v2': IN_V2_CLASSNAMES,
        'imagenet_sketch': IN_SKETCH_CLASSNAMES
        }

_DATASETS = {
        'imagenet_1k': os.path.join(args.root, 'imagenet', 'images', 'val'),
        'imagenet_rendition': os.path.join(args.root, 'imagenet-rendition', 'imagenet-r'),
        'imagenet_adversarial': os.path.join(args.root, 'imagenet-adversarial'),
        'imagenet_v2': os.path.join(args.root, 'imagenetv2'),
        'imagenet_sketch': os.path.join(args.root, 'imagenet-sketch')
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


image_transform = _transform(IMAGE_SIZE)
augmenter = CropAugmenter(base_transform=image_transform, n_views=AS)
tst_dir = _DATASETS[dataset]
tst_dataset = datasets.ImageFolder(tst_dir, transform=augmenter)
tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

clip_model, _ = clip.load(net_name, 
                          device=device) 

clip_model.eval() # returned clip model is in evaluation mode.
clip_model.requires_grad_(False)
logit_scale = clip_model.logit_scale.exp()

pre_extracted_embeddings = []
with torch.no_grad():
    for i, ((xs), y) in enumerate(tqdm(tst_loader)):
        start = time.time()

        x = xs[0]
        aug_x = xs[1]    
    
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
