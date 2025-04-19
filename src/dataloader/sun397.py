from __future__ import annotations

import numpy as np
import os
import PIL.Image
from typing import Any, Tuple, List
from src.dataloader.base import BaseDataset

SUN397_CLASSNAMES = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'outdoor apartment_building', 'indoor apse', 'aquarium', 'aqueduct', 'arch', 'archive', 'outdoor arrival_gate', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'outdoor athletic_field', 'public atrium', 'attic', 'auditorium', 'auto_factory', 'badlands', 'indoor badminton_court', 'baggage_claim', 'shop bakery', 'exterior balcony', 'interior balcony', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'outdoor basketball_court', 'bathroom', 'batters_box', 'bayou', 'indoor bazaar', 'outdoor bazaar', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'indoor bistro', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'indoor booth', 'botanical_garden', 'indoor bow_window', 'outdoor bow_window', 'bowling_alley', 'boxing_ring', 'indoor brewery', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal', 'urban canal', 'candy_store', 'canyon', 'backseat car_interior', 'frontseat car_interior', 'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral', 'indoor cavern', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'indoor chicken_coop', 'outdoor chicken_coop', 'childs_room', 'indoor church', 'outdoor church', 'classroom', 'clean_room', 'cliff', 'indoor cloister', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'outdoor control_tower', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'exterior covered_bridge', 'creek', 'crevasse', 'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists_office', 'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'home dinette', 'vehicle dinette', 'dining_car', 'dining_room', 'discotheque', 'dock', 'outdoor doorway', 'dorm_room', 'driveway', 'outdoor driving_range', 'drugstore', 'electrical_substation', 'door elevator', 'interior elevator', 'elevator_shaft', 'engine_room', 'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'fastfood_restaurant', 'cultivated field', 'wild field', 'fire_escape', 'fire_station', 'indoor firing_range', 'fishpond', 'indoor florist_shop', 'food_court', 'broadleaf forest', 'needleleaf forest', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'indoor garage', 'garbage_dump', 'gas_station', 'exterior gazebo', 'indoor general_store', 'outdoor general_store', 'gift_shop', 'golf_course', 'indoor greenhouse', 'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'outdoor hot_tub', 'outdoor hotel', 'hotel_room', 'house', 'outdoor hunting_lodge', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'indoor ice_skating_rink', 'outdoor ice_skating_rink', 'iceberg', 'igloo', 'industrial_area', 'outdoor inn', 'islet', 'indoor jacuzzi', 'indoor jail', 'jail_cell', 'jewelry_shop', 'kasbah', 'indoor kennel', 'outdoor kennel', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'outdoor labyrinth', 'natural lake', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'indoor library', 'outdoor library', 'outdoor lido_deck', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'indoor market', 'outdoor market', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'water moat', 'outdoor monastery', 'indoor mosque', 'outdoor mosque', 'motel', 'mountain', 'mountain_snowy', 'indoor movie_theater', 'indoor museum', 'music_store', 'music_studio', 'outdoor nuclear_power_plant', 'nursery', 'oast_house', 'outdoor observatory', 'ocean', 'office', 'office_building', 'outdoor oil_refinery', 'oilrig', 'operating_room', 'orchard', 'outdoor outhouse', 'pagoda', 'palace', 'pantry', 'park', 'indoor parking_garage', 'outdoor parking_garage', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'indoor pilothouse', 'outdoor planetarium', 'playground', 'playroom', 'plaza', 'indoor podium', 'outdoor podium', 'pond', 'establishment poolroom', 'home poolroom', 'outdoor power_plant', 'promenade_deck', 'indoor pub', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'indoor shopping_mall', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'baseball stadium', 'football stadium', 'indoor stage', 'staircase', 'street', 'subway_interior', 'platform subway_station', 'supermarket', 'sushi_bar', 'swamp', 'indoor swimming_pool', 'outdoor swimming_pool', 'indoor synagogue', 'outdoor synagogue', 'television_studio', 'east_asia temple', 'south_asia temple', 'indoor tennis_court', 'outdoor tennis_court', 'outdoor tent', 'indoor_procenium theater', 'indoor_seats theater', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'outdoor track', 'train_railway', 'platform train_station', 'tree_farm', 'tree_house', 'trench', 'coral_reef underwater', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'indoor volleyball_court', 'outdoor volleyball_court', 'waiting_room', 'indoor warehouse', 'water_tower', 'block waterfall', 'fan waterfall', 'plunge waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'barrel_storage wine_cellar', 'bottle_storage wine_cellar', 'indoor wrestling_ring', 'yard', 'youth_hostel']

class SUN397(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        **kwargs
    ) -> None:
        super().__init__(root, transform)

        self.dataset_dir = os.path.join(root, 'sun397')
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")

        if os.path.exists(self.split_path):
            train_lst, val_lst, test_lst = self.read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.dataset_dir, "SUN397", "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:] 
                    classnames.append(line)
            cname2lab = {c: i for i, c in enumerate(classnames)}
            trainval = self.read_data(cname2lab, "Training_01.txt")
            test_lst = self.read_data(cname2lab, "Testing_01.txt")
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
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                item = (impath, label, classname)
                items.append(item)

        return items