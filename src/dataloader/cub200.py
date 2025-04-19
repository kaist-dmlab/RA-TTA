import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


CUB_CLASSNAMES = ['Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove-billed Ani',
 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', "Brewer's Blackbird",
 'Red-winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink', 'Indigo Bunting',
 'Lazuli Bunting', 'Painted Bunting', 'Cardinal bird', 'Spotted Catbird', 'Gray Catbird', 'Yellow-breasted Chat',
 'Eastern Towhee', "Chuck-will's Widow", "Brandt's Cormorant", 'Red-faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird',
 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black-billed Cuckoo', 'Mangrove Cuckoo', 'Yellow-billed Cuckoo',
 'Gray-crowned-Rosy Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great-Crested Flycatcher',
 'Least Flycatcher', 'Olive-sided Flycatcher', 'Scissor-tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow-bellied Flycatcher',
 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat-tailed Grackle',
 'Eared Grebe', 'Horned Grebe', 'Pied-billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak',
 'Pine Grosbeak', 'Rose-breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull', "Heermann's Gull",
 'Herring Gull', 'Ivory Gull', 'Ring-billed Gull', 'Slaty-backed Gull', 'Western Gull', "Anna's Hummingbird",
 'Ruby-throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long-tailed Jaeger', 'Pomarine Jaeger',
 'Blue Jay', 'Florida Scrub Jay', 'Green Jay', 'Dark-eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher',
 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White-throated Kingfisher', 'Red-legged Kittiwake',
 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red-breasted Merganser',
 'Mockingbird', 'Nighthawk', "Clark's Nutcracker", 'White-breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole',
 'Orchard Oriole', "Scott's Oriole", 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western-Wood Pewee', "Say's Phoebe",
 'American Pipit', 'Whip-poor Will', 'Horned Puffin', 'Common Raven', 'White-necked Raven', 'American Redstart',
 "Greater Roadrunner", 'Loggerhead Shrike', 'Great-Grey Shrike', "Baird's Sparrow", 'Black-throated Sparrow', "Brewer's Sparrow",
 'Chipping Sparrow', 'Clay-colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow',
 "Harris's Sparrow", "Henslow's Sparrow", "Le-Conte's Sparrow", "Lincoln's Sparrow", "Nelson's Sparrow", 'Savannah Sparrow',
 'Seaside Sparrow', 'Song Sparrow', 'American Tree Sparrow', 'Vesper Sparrow', 'White-crowned Sparrow', 'White-throated Sparrow',
 'Cape-Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager',
 'Summer Tanager', 'Arctic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern',
 'Least Tern', 'Green-tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black-capped Vireo', 'Blue-headed Vireo',
 'Philadelphia Vireo', 'Red-eyed Vireo', 'Warbling Vireo', 'White-eyed Vireo', 'Yellow-throated Vireo', 'Bay-breasted Warbler',
 'Black-and-white Warbler', 'Black-throated-Blue Warbler', 'Blue-winged Warbler', 'Canada Warbler', 'Cape-May Warbler',
 'Cerulean Warbler', 'Chestnut-sided Warbler', 'Golden-winged Warbler', 'Hooded Warbler', 'Kentucky Warbler',
 'Magnolia Warbler', 'Mourning Warbler', 'Yellow rumped Warbler', 'Nashville Warbler', 'Orange-crowned Warbler',
 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', "Swainson's Warbler", 'Tennessee Warbler',
 "Wilson's Warbler", 'Worm-eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush',
 'Bohemian Waxwing', 'Cedar Waxwing', 'American-Three-toed Woodpecker', 'Pileated Woodpecker', 'Red-bellied Woodpecker',
 'Red-cockaded Woodpecker', 'Red-headed Woodpecker', 'Downy Woodpecker', "Bewick's Wren", 'Cactus Wren',
 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']

class CUB200(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, 
                 root: str, 
                 train=False, # for TTA set-up
                 transform=None, 
                 loader=default_loader, 
                 download=False):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        label = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            image = self.transform(img)

        return image, label, CUB_CLASSNAMES[label]

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

if __name__ == '__main__':
    dataset = CUB200()