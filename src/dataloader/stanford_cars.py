from __future__ import annotations

import numpy as np
import os
import PIL.Image
from typing import Any, Tuple, List
from scipy.io import loadmat
from src.dataloader.base import BaseDataset

CARS_CLASSNAMES = ['2000 AM General Hummer SUV', '2012 Acura RL Sedan', '2012 Acura TL Sedan', '2008 Acura TL Type-S', '2012 Acura TSX Sedan', '2001 Acura Integra Type R', \
        '2012 Acura ZDX Hatchback', '2012 Aston Martin V8 Vantage Convertible', '2012 Aston Martin V8 Vantage Coupe', '2012 Aston Martin Virage Convertible', \
        '2012 Aston Martin Virage Coupe', '2008 Audi RS 4 Convertible', '2012 Audi A5 Coupe', '2012 Audi TTS Coupe', '2012 Audi R8 Coupe', '1994 Audi V8 Sedan', \
        '1994 Audi 100 Sedan', '1994 Audi 100 Wagon', '2011 Audi TT Hatchback', '2011 Audi S6 Sedan', '2012 Audi S5 Convertible', '2012 Audi S5 Coupe', '2012 Audi S4 Sedan', \
        '2007 Audi S4 Sedan', '2012 Audi TT RS Coupe', '2012 BMW ActiveHybrid 5 Sedan', '2012 BMW 1 Series Convertible', '2012 BMW 1 Series Coupe', '2012 BMW 3 Series Sedan', \
        '2012 BMW 3 Series Wagon', '2007 BMW 6 Series Convertible', '2007 BMW X5 SUV', '2012 BMW X6 SUV', '2012 BMW M3 Coupe', '2010 BMW M5 Sedan', '2010 BMW M6 Convertible', \
        '2012 BMW X3 SUV', '2012 BMW Z4 Convertible', '2012 Bentley Continental Supersports Conv. Convertible', '2009 Bentley Arnage Sedan', '2011 Bentley Mulsanne Sedan', \
        '2012 Bentley Continental GT Coupe', '2007 Bentley Continental GT Coupe', '2007 Bentley Continental Flying Spur Sedan', '2009 Bugatti Veyron 16.4 Convertible', \
        '2009 Bugatti Veyron 16.4 Coupe', '2012 Buick Regal GS', '2007 Buick Rainier SUV', '2012 Buick Verano Sedan', '2012 Buick Enclave SUV', '2012 Cadillac CTS-V Sedan', \
        '2012 Cadillac SRX SUV', '2007 Cadillac Escalade EXT Crew Cab', '2012 Chevrolet Silverado 1500 Hybrid Crew Cab', '2012 Chevrolet Corvette Convertible', \
        '2012 Chevrolet Corvette ZR1', '2007 Chevrolet Corvette Ron Fellows Edition Z06', '2012 Chevrolet Traverse SUV', '2012 Chevrolet Camaro Convertible', \
        '2010 Chevrolet HHR SS', '2007 Chevrolet Impala Sedan', '2012 Chevrolet Tahoe Hybrid SUV', '2012 Chevrolet Sonic Sedan', '2007 Chevrolet Express Cargo Van', \
        '2012 Chevrolet Avalanche Crew Cab', '2010 Chevrolet Cobalt SS', '2010 Chevrolet Malibu Hybrid Sedan', '2009 Chevrolet TrailBlazer SS', \
        '2012 Chevrolet Silverado 2500HD Regular Cab', '2007 Chevrolet Silverado 1500 Classic Extended Cab', '2007 Chevrolet Express Van', '2007 Chevrolet Monte Carlo Coupe', \
        '2007 Chevrolet Malibu Sedan', '2012 Chevrolet Silverado 1500 Extended Cab', '2012 Chevrolet Silverado 1500 Regular Cab', '2009 Chrysler Aspen SUV', \
        '2010 Chrysler Sebring Convertible', '2012 Chrysler Town and Country Minivan', '2010 Chrysler 300 SRT-8', '2008 Chrysler Crossfire Convertible', \
        '2008 Chrysler PT Cruiser Convertible', '2002 Daewoo Nubira Wagon', '2012 Dodge Caliber Wagon', '2007 Dodge Caliber Wagon', '1997 Dodge Caravan Minivan', \
        '2010 Dodge Ram Pickup 3500 Crew Cab', '2009 Dodge Ram Pickup 3500 Quad Cab', '2009 Dodge Sprinter Cargo Van', '2012 Dodge Journey SUV', '2010 Dodge Dakota Crew Cab', \
        '2007 Dodge Dakota Club Cab', '2008 Dodge Magnum Wagon', '2011 Dodge Challenger SRT8', '2012 Dodge Durango SUV', '2007 Dodge Durango SUV', '2012 Dodge Charger Sedan', \
        '2009 Dodge Charger SRT-8', '1998 Eagle Talon Hatchback', '2012 FIAT 500 Abarth', '2012 FIAT 500 Convertible', '2012 Ferrari FF Coupe', '2012 Ferrari California Convertible', '2012 Ferrari 458 Italia Convertible', '2012 Ferrari 458 Italia Coupe', '2012 Fisker Karma Sedan', '2012 Ford F-450 Super Duty Crew Cab', '2007 Ford Mustang Convertible', '2007 Ford Freestar Minivan', '2009 Ford Expedition EL SUV', '2012 Ford Edge SUV', '2011 Ford Ranger SuperCab', '2006 Ford GT Coupe', '2012 Ford F-150 Regular Cab', '2007 Ford F-150 Regular Cab', '2007 Ford Focus Sedan', '2012 Ford E-Series Wagon Van', '2012 Ford Fiesta Sedan', '2012 GMC Terrain SUV', '2012 GMC Savana Van', '2012 GMC Yukon Hybrid SUV', '2012 GMC Acadia SUV', '2012 GMC Canyon Extended Cab', '1993 Geo Metro Convertible', '2010 HUMMER H3T Crew Cab', '2009 HUMMER H2 SUT Crew Cab', '2012 Honda Odyssey Minivan', '2007 Honda Odyssey Minivan', '2012 Honda Accord Coupe', '2012 Honda Accord Sedan', '2012 Hyundai Veloster Hatchback', '2012 Hyundai Santa Fe SUV', '2012 Hyundai Tucson SUV', '2012 Hyundai Veracruz SUV', '2012 Hyundai Sonata Hybrid Sedan', '2007 Hyundai Elantra Sedan', '2012 Hyundai Accent Sedan', '2012 Hyundai Genesis Sedan', '2012 Hyundai Sonata Sedan', '2012 Hyundai Elantra Touring Hatchback', '2012 Hyundai Azera Sedan', '2012 Infiniti G Coupe IPL', '2011 Infiniti QX56 SUV', '2008 Isuzu Ascender SUV', '2012 Jaguar XK XKR', '2012 Jeep Patriot SUV', '2012 Jeep Wrangler SUV', '2012 Jeep Liberty SUV', '2012 Jeep Grand Cherokee SUV', '2012 Jeep Compass SUV', '2008 Lamborghini Reventon Coupe', '2012 Lamborghini Aventador Coupe', '2012 Lamborghini Gallardo LP 570-4 Superleggera', '2001 Lamborghini Diablo Coupe', '2012 Land Rover Range Rover SUV', '2012 Land Rover LR2 SUV', '2011 Lincoln Town Car Sedan', '2012 MINI Cooper Roadster Convertible', '2012 Maybach Landaulet Convertible', '2011 Mazda Tribute SUV', '2012 McLaren MP4-12C Coupe', '1993 Mercedes-Benz 300-Class Convertible', '2012 Mercedes-Benz C-Class Sedan', '2009 Mercedes-Benz SL-Class Coupe', '2012 Mercedes-Benz E-Class Sedan', '2012 Mercedes-Benz S-Class Sedan', '2012 Mercedes-Benz Sprinter Van', '2012 Mitsubishi Lancer Sedan', '2012 Nissan Leaf Hatchback', '2012 Nissan NV Passenger Van', '2012 Nissan Juke Hatchback', '1998 Nissan 240SX Coupe', '1999 Plymouth Neon Coupe', '2012 Porsche Panamera Sedan', '2012 Ram C/V Cargo Van Minivan', '2012 Rolls-Royce Phantom Drophead Coupe Convertible', '2012 Rolls-Royce Ghost Sedan', '2012 Rolls-Royce Phantom Sedan', '2012 Scion xD Hatchback', '2009 Spyker C8 Convertible', '2009 Spyker C8 Coupe', '2007 Suzuki Aerio Sedan', '2012 Suzuki Kizashi Sedan', '2012 Suzuki SX4 Hatchback', '2012 Suzuki SX4 Sedan', '2012 Tesla Model S Sedan', '2012 Toyota Sequoia SUV', '2012 Toyota Camry Sedan', '2012 Toyota Corolla Sedan', '2012 Toyota 4Runner SUV', '2012 Volkswagen Golf Hatchback', '1991 Volkswagen Golf Hatchback', '2012 Volkswagen Beetle Hatchback', '2012 Volvo C30 Hatchback', '1993 Volvo 240 Sedan', '2007 Volvo XC90 SUV', '2012 smart fortwo Convertible']

class StanfordCars(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, 'stanford_cars')
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        if os.path.exists(self.split_path):
            train_lst, val_lst, test_lst = self.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval_lst = self.read_data("cars_train", trainval_file, meta_file)
            test_lst = self.read_data("cars_test", test_file, meta_file)
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
        
        if self.is_path:
            return image, label, class_name, image_file
        else:
            return image, label, class_name
            
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = (impath, label, classname)
            items.append(item)

        return items