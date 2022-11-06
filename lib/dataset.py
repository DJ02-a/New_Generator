import glob
import torch
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from lib.utils_mask import label_converter, to_one_hot

from lib import utils

class SingleFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list_images, dataset_root_list_masks, isMaster):
        self.img_size = 512
        
        self.image_path_list, self.image_num_list = utils.get_all_images(dataset_root_list_images)
        self.mask_path_list, self.mask_num_list = utils.get_all_images(dataset_root_list_masks)
        
        # self.image_path_list = glob.glob('/home/jjy/Datasets/celeba/train/images/*.*')
        # self.mask_path_list = glob.glob('/home/jjy/Datasets/celeba/train/label/*.*')
        
        self.transforms_gray = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.transforms_color = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")
            print(f"Dataset of {sum(self.mask_num_list)} masks constructed for the training.")


    def __getitem__(self, item):
        idx = 0
        while item >= self.image_num_list[idx]:
            item -= self.image_num_list[idx]
            idx += 1
            
            
        image_path = self.image_path_list[idx][item]
        mask_path = self.mask_path_list[idx][item]
        
        color_image = Image.open(image_path)
        gray_image = color_image.convert("L")

        mask = Image.open(mask_path).resize((self.img_size,self.img_size),Image.NEAREST)
        _mask = label_converter(mask) # face parsing -> simple
        mask_one_hot = to_one_hot(_mask) # [1, H, W] --> [12, H, W] (New label)

        return self.transforms_gray(gray_image), self.transforms_color(color_image), mask_one_hot

    def __len__(self):
        return sum(self.image_num_list)

    # def __getitem__(self, item):
        
    #     image_path = self.image_path_list[item]
    #     mask_path = self.mask_path_list[item]
        
    #     color_image = Image.open(image_path)
    #     gray_image = color_image.convert("L")
        
    #     mask = Image.open(mask_path).resize((self.img_size,self.img_size),Image.NEAREST)
    #     _mask = label_converter(mask) # face parsing -> simple
    #     mask_one_hot = to_one_hot(_mask) # [1, H, W] --> [12, H, W] (New label)

    #     return self.transforms_gray(gray_image), self.transforms_color(color_image), mask_one_hot

    # def __len__(self):
    #     return len(self.image_path_list)


class SingleFaceDatasetValid(Dataset):
    def __init__(self, dataset_root_list, isMaster):
        self.img_size = 512
        
        self.image_path_list = glob.glob('/home/jjy/Datasets/celeba/train/images/*.*')[:40]
        self.mask_path_list = glob.glob('/home/jjy/Datasets/celeba/train/label/*.*')[:40]
        
        self.transforms_gray = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.transforms_color = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        
        image_path = self.image_path_list[item]
        mask_path = self.mask_path_list[item]
        
        color_image = Image.open(image_path)
        gray_image = color_image.convert("L")
        
        mask = Image.open(mask_path).resize((self.img_size,self.img_size),Image.NEAREST)
        _mask = label_converter(mask)
        mask_one_hot = to_one_hot(_mask)

        return self.transforms_gray(gray_image), self.transforms_color(color_image), mask_one_hot

    def __len__(self):
        return len(self.image_path_list)

class PairedFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster, same_prob=0.2):
        self.same_prob = same_prob
        self.image_path_list, self.image_num_list = utils.get_all_images(dataset_root_list)
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        idx = 0
        while item >= self.image_num_list[idx]:
            item -= self.image_num_list[idx]
            idx += 1
        image_path = self.image_path_list[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        if random.random() > self.same_prob:
            image_path = random.choice(self.image_path_list[random.randint(0, len(self.image_path_list)-1)])
            Xt = Image.open(image_path).convert("RGB")
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.image_num_list)


class PairedFaceDatasetValid(Dataset):
    def __init__(self, valid_data_dir, isMaster):
        
        self.source_path_list = sorted(glob.glob(f"{valid_data_dir}/source/*.*g"))
        self.target_path_list = sorted(glob.glob(f"{valid_data_dir}/target/*.*g"))

        # take the smaller number if two dirs have different numbers of images
        self.num = min(len(self.source_path_list), len(self.target_path_list))
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the validation.")

    def __getitem__(self, idx):
        
        Xs = Image.open(self.source_path_list[idx]).convert("RGB")
        Xt = Image.open(self.target_path_list[idx]).convert("RGB")

        return self.transforms(Xs), self.transforms(Xt)

    def __len__(self):
        return self.num


