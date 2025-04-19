import logging
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.vision import VisionDataset
BICUBIC = InterpolationMode.BICUBIC
logger = logging.getLogger('RATTA')


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class AuxDataset(VisionDataset):
    def __init__(self, 
                 root, 
                 database=None,
                 dataset=None,
                 transform=None, 
                 **kwargs) -> None:
        super().__init__(root)
        self.data = []
        self.targets = []

        self.transform = transform 

        retrieved_img_path = os.path.join(root, database, 'extracted_images', dataset, 'retrieved_imgs')
        retrieved_img_path_folders = [x for x in os.listdir(retrieved_img_path) if os.path.isdir(os.path.join(retrieved_img_path, x))]
        for folder in retrieved_img_path_folders:
            file_lst = os.listdir(os.path.join(retrieved_img_path, folder))
            for img in file_lst:
                if img.endswith('jpg'):
                    self.data.append(os.path.join(retrieved_img_path, folder, img)) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
    
        if self.transform:
            image = self.transform(image)

        return image