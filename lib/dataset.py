import glob
import torch
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
# from lib.utils_mask import label_converter, to_one_hot
from lib.utils_mask import label_converter, face_mask2one_hot, part_mask2one_hot, converter
import cv2
from lib import utils
import numpy as np

class SingleFaceDatasetTrain(Dataset):
    def __init__(self, args, isMaster):
        self.args = args
        self.face_size = 512
        self.part_size = 128
        self._transforms()
        
        self.num_dict = {
            'color':0,
            'head':0,
            'l_brow':0,
            'r_brow':0,
            'l_eye':0,
            'r_eye':0,
            'nose':0,
            'mouth':0,
        }
        
        
        self.image_path_list = utils.get_all_images(self.args.train_color_images)[:-40]
        self.mask_path_list = utils.get_all_images(self.args.train_color_masks)[:-40]
        self.num_dict['color'] = len(self.image_path_list)
        self.num_dict['head'] = len(self.image_path_list)
        
        self.l_brow_path_list =  utils.get_all_images(self.args.train_l_brow_images)[:-40]
        self.mask_l_brow_path_list =  utils.get_all_images(self.args.train_l_brow_mask)[:-40]
        self.num_dict['l_brow'] = len(self.l_brow_path_list)
        
        self.r_brow_path_list =  utils.get_all_images(self.args.train_r_brow_images)[:-40]
        self.mask_r_brow_path_list =  utils.get_all_images(self.args.train_r_brow_mask)[:-40]
        self.num_dict['r_brow'] = len(self.r_brow_path_list)
        
        self.l_eye_path_list =  utils.get_all_images(self.args.train_l_eye_images)[:-40]
        self.mask_l_eye_path_list =  utils.get_all_images(self.args.train_l_eye_mask)[:-40]
        self.num_dict['l_eye'] = len(self.l_eye_path_list)
        
        self.r_eye_path_list =  utils.get_all_images(self.args.train_r_eye_images)[:-40]
        self.mask_r_eye_path_list =  utils.get_all_images(self.args.train_r_eye_mask)[:-40]
        self.num_dict['r_eye'] = len(self.r_eye_path_list)
        
        self.nose_path_list =  utils.get_all_images(self.args.train_nose_images)[:-40]
        self.mask_nose_path_list =  utils.get_all_images(self.args.train_nose_mask)[:-40]
        self.num_dict['nose'] = len(self.nose_path_list)
        
        
        self.mouth_path_list =  utils.get_all_images(self.args.train_mouth_images)[:-40]
        self.mask_mouth_path_list =  utils.get_all_images(self.args.train_mouth_mask)[:-40]
        self.num_dict['mouth'] = len(self.mouth_path_list)
        
        
        
        if isMaster:
            print(f"Image Dataset of {self.num_dict['color']} image constructed for the training.")
            print(f"L brow Dataset of {self.num_dict['head']} image constructed for the training.")
            print(f"R brow Dataset of {self.num_dict['l_brow']} image constructed for the training.")
            print(f"L eye Dataset of {self.num_dict['r_brow']} image constructed for the training.")
            print(f"R eye Dataset of {self.num_dict['l_eye']} image constructed for the training.")
            print(f"Nose Dataset of {self.num_dict['r_eye']} image constructed for the training.")
            print(f"Nose Dataset of {self.num_dict['nose']} image constructed for the training.")
            print(f"Mouth Dataset of {self.num_dict['mouth']} image constructed for the training.")

    def __getitem__(self, _):
        
        color_img, color_gray, color_mask = self.data_pp(self.image_path_list, self.mask_path_list, False, 'color')
        head_img, head_gray, head_mask = self.data_pp(self.image_path_list, self.mask_path_list, False, 'head')
        l_brow_img, l_brow_gray, l_brow_mask = self.data_pp(self.l_brow_path_list, self.mask_l_brow_path_list, False, 'l_brow')
        r_brow_img, r_brow_gray, r_brow_mask = self.data_pp(self.r_brow_path_list, self.mask_r_brow_path_list, False, 'r_brow')
        l_eye_img, l_eye_gray, l_eye_mask = self.data_pp(self.l_eye_path_list, self.mask_l_eye_path_list, False, 'l_eye')
        r_eye_img, r_eye_gray, r_eye_mask = self.data_pp(self.r_eye_path_list, self.mask_r_eye_path_list, False, 'r_eye')
        nose_img, nose_gray, nose_mask = self.data_pp(self.nose_path_list, self.mask_nose_path_list, False, 'nose')
        mouth_img, mouth_gray, mouth_mask = self.data_pp(self.mouth_path_list, self.mask_mouth_path_list, False, 'mouth')
        

        return color_img, color_gray, color_mask, head_img, head_gray, head_mask, l_brow_img, l_brow_gray, l_brow_mask, \
                r_brow_img, r_brow_gray, r_brow_mask, l_eye_img, l_eye_gray, l_eye_mask, r_eye_img, r_eye_gray, r_eye_mask, \
                    nose_img, nose_gray, nose_mask, mouth_img, mouth_gray, mouth_mask

    def __len__(self):
        return len(self.image_path_list)
    
    def data_pp(self, image_path_list, mask_path_list, equalizeHist=False, part=None):    
        idx = random.randint(0, self.num_dict[part]-1)
        image_path = image_path_list[idx]
        mask_path = mask_path_list[idx]
        
        color_image = Image.open(image_path)
        gray_image = color_image.convert("L")
        mask = Image.open(mask_path)
        if equalizeHist:
            gray_image = cv2.equalizeHist(np.array(gray_image))
            gray_image = Image.fromarray(gray_image)

        if part in ['head','color']:
            color_tensor = self.face_transforms_color(color_image)
            gray_tensor = self.face_transforms_gray(gray_image)
            _mask = label_converter(mask) # face parsing -> simple
            mask_one_hot = face_mask2one_hot(_mask, 512) # [1, H, W] --> [12, H, W] (New label)
            if part == 'color':
                skin_pixels = torch.masked_select(color_tensor, mask_one_hot[1].bool()).reshape(3, -1)
                self.skin_mean = torch.mean(skin_pixels, dim=1).unsqueeze(1)
            
        else:     
            color_tensor = self.part_transforms_color(color_image)
            gray_tensor = self.part_transforms_gray(gray_image)
            mask = mask.resize((self.part_size,self.part_size),Image.NEAREST)
            mask_one_hot = part_mask2one_hot(mask, part)
            
            color_tensor = self.fill_skin_region(color_tensor, mask_one_hot, part)

        return color_tensor, gray_tensor, mask_one_hot
    
    def fill_skin_region(self, color_tensor, mask_one_hot, part):
        c, h, w = color_tensor.shape
        ch_idx = converter[part]['label'][1]
        # color_pixels = torch.masked_select(color_tensor, (1-mask_one_hot[ch_idx]).bool()).reshape(c, -1) # 3, pixel_num
        # mean_color = torch.mean(color_pixels, dim=1).unsqueeze(1) # n 3
        scatter_pixel_num = (h * w) - mask_one_hot[ch_idx].bool().sum()
        mean_colors = self.skin_mean.repeat(1, int(scatter_pixel_num))
        color_tensor.masked_scatter_((1 - mask_one_hot[ch_idx]).bool(), mean_colors)
        
        return color_tensor
    
    def _transforms(self):
        self.face_transforms_gray = transforms.Compose([
            transforms.Resize((self.face_size,self.face_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.face_transforms_color = transforms.Compose([
            transforms.Resize((self.face_size,self.face_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.part_transforms_gray = transforms.Compose([
            transforms.Resize((self.part_size,self.part_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.part_transforms_color = transforms.Compose([
            transforms.Resize((self.part_size,self.part_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class SingleFaceDatasetValid(Dataset):
    def __init__(self, args, isMaster):
        self.args = args
        self.face_size = 512
        self.part_size = 128
        self._transforms()
        
        self.num_dict = {
            'color':0,
            'head':0,
            'l_brow':0,
            'r_brow':0,
            'l_eye':0,
            'r_eye':0,
            'nose':0,
            'mouth':0,
        }
        
        
        self.image_path_list = utils.get_all_images(self.args.train_color_images)[-40:]
        self.mask_path_list = utils.get_all_images(self.args.train_color_masks)[-40:]
        self.num_dict['color'] = len(self.image_path_list)
        self.num_dict['head'] = len(self.image_path_list)
        
        self.l_brow_path_list =  utils.get_all_images(self.args.train_l_brow_images)[-40:]
        self.mask_l_brow_path_list =  utils.get_all_images(self.args.train_l_brow_mask)[-40:]
        self.num_dict['l_brow'] = len(self.l_brow_path_list)
        
        self.r_brow_path_list =  utils.get_all_images(self.args.train_r_brow_images)[-40:]
        self.mask_r_brow_path_list =  utils.get_all_images(self.args.train_r_brow_mask)[-40:]
        self.num_dict['r_brow'] = len(self.r_brow_path_list)
        
        self.l_eye_path_list =  utils.get_all_images(self.args.train_l_eye_images)[-40:]
        self.mask_l_eye_path_list =  utils.get_all_images(self.args.train_l_eye_mask)[-40:]
        self.num_dict['l_eye'] = len(self.l_eye_path_list)
        
        self.r_eye_path_list =  utils.get_all_images(self.args.train_r_eye_images)[-40:]
        self.mask_r_eye_path_list =  utils.get_all_images(self.args.train_r_eye_mask)[-40:]
        self.num_dict['r_eye'] = len(self.r_eye_path_list)
        
        self.nose_path_list =  utils.get_all_images(self.args.train_nose_images)[-40:]
        self.mask_nose_path_list =  utils.get_all_images(self.args.train_nose_mask)[-40:]
        self.num_dict['nose'] = len(self.nose_path_list)
        
        
        self.mouth_path_list =  utils.get_all_images(self.args.train_mouth_images)[-40:]
        self.mask_mouth_path_list =  utils.get_all_images(self.args.train_mouth_mask)[-40:]
        self.num_dict['mouth'] = len(self.mouth_path_list)
        
        
        
        if isMaster:
            print(f"Image Dataset of {self.num_dict['color']} image constructed for the training.")
            print(f"L brow Dataset of {self.num_dict['head']} image constructed for the training.")
            print(f"R brow Dataset of {self.num_dict['l_brow']} image constructed for the training.")
            print(f"L eye Dataset of {self.num_dict['r_brow']} image constructed for the training.")
            print(f"R eye Dataset of {self.num_dict['l_eye']} image constructed for the training.")
            print(f"Nose Dataset of {self.num_dict['r_eye']} image constructed for the training.")
            print(f"Nose Dataset of {self.num_dict['nose']} image constructed for the training.")
            print(f"Mouth Dataset of {self.num_dict['mouth']} image constructed for the training.")

    def __getitem__(self, _):
        color_img, color_gray, color_mask = self.data_pp(self.image_path_list, self.mask_path_list, False, 'color')
        head_img, head_gray, head_mask = self.data_pp(self.image_path_list, self.mask_path_list, False, 'head')
        l_brow_img, l_brow_gray, l_brow_mask = self.data_pp(self.l_brow_path_list, self.mask_l_brow_path_list, False, 'l_brow')
        r_brow_img, r_brow_gray, r_brow_mask = self.data_pp(self.r_brow_path_list, self.mask_r_brow_path_list, False, 'r_brow')
        l_eye_img, l_eye_gray, l_eye_mask = self.data_pp(self.l_eye_path_list, self.mask_l_eye_path_list, False, 'l_eye')
        r_eye_img, r_eye_gray, r_eye_mask = self.data_pp(self.r_eye_path_list, self.mask_r_eye_path_list, False, 'r_eye')
        nose_img, nose_gray, nose_mask = self.data_pp(self.nose_path_list, self.mask_nose_path_list, False, 'nose')
        mouth_img, mouth_gray, mouth_mask = self.data_pp(self.mouth_path_list, self.mask_mouth_path_list, False, 'mouth')
        

        return color_img, color_gray, color_mask, head_img, head_gray, head_mask, l_brow_img, l_brow_gray, l_brow_mask, \
                r_brow_img, r_brow_gray, r_brow_mask, l_eye_img, l_eye_gray, l_eye_mask, r_eye_img, r_eye_gray, r_eye_mask, \
                    nose_img, nose_gray, nose_mask, mouth_img, mouth_gray, mouth_mask

    def __len__(self):
        return len(self.image_path_list)
    
    def data_pp(self, image_path_list, mask_path_list, equalizeHist=False, part=None):    
        idx = random.randint(0, self.num_dict[part]-1)
        image_path = image_path_list[idx]
        mask_path = mask_path_list[idx]
        
        color_image = Image.open(image_path)
        gray_image = color_image.convert("L")
        mask = Image.open(mask_path)
        if equalizeHist:
            gray_image = cv2.equalizeHist(np.array(gray_image))
            gray_image = Image.fromarray(gray_image)

        if part in ['head','color']:
            color_tensor = self.face_transforms_color(color_image)
            gray_tensor = self.face_transforms_gray(gray_image)
            _mask = label_converter(mask) # face parsing -> simple
            mask_one_hot = face_mask2one_hot(_mask, 512) # [1, H, W] --> [12, H, W] (New label)
            if part == 'color':
                skin_pixels = torch.masked_select(color_tensor, mask_one_hot[1].bool()).reshape(3, -1)
                self.skin_mean = torch.mean(skin_pixels, dim=1).unsqueeze(1)
            
        else:     
            color_tensor = self.part_transforms_color(color_image)
            gray_tensor = self.part_transforms_gray(gray_image)
            mask = mask.resize((self.part_size,self.part_size),Image.NEAREST)
            mask_one_hot = part_mask2one_hot(mask, part)
            
            color_tensor = self.fill_skin_region(color_tensor, mask_one_hot, part)

        return color_tensor, gray_tensor, mask_one_hot
    
    def fill_skin_region(self, color_tensor, mask_one_hot, part):
        c, h, w = color_tensor.shape
        ch_idx = converter[part]['label'][1]
        # color_pixels = torch.masked_select(color_tensor, (1-mask_one_hot[ch_idx]).bool()).reshape(c, -1) # 3, pixel_num
        # mean_color = torch.mean(color_pixels, dim=1).unsqueeze(1) # n 3
        scatter_pixel_num = (h * w) - mask_one_hot[ch_idx].bool().sum()
        mean_colors = self.skin_mean.repeat(1, int(scatter_pixel_num))
        color_tensor.masked_scatter_((1 - mask_one_hot[ch_idx]).bool(), mean_colors)
        
        return color_tensor
    
    def _transforms(self):
        self.face_transforms_gray = transforms.Compose([
            transforms.Resize((self.face_size,self.face_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.face_transforms_color = transforms.Compose([
            transforms.Resize((self.face_size,self.face_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.part_transforms_gray = transforms.Compose([
            transforms.Resize((self.part_size,self.part_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.part_transforms_color = transforms.Compose([
            transforms.Resize((self.part_size,self.part_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

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


