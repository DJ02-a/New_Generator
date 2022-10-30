import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from MyModel.sub_nets.C_NET import C_Net
from MyModel.sub_nets.New_Generator import My_Generator
from MyModel.utils.PSP import GradualStyleEncoder

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.structure_encoder = GradualStyleEncoder(1)
        self.color_encoder = GradualStyleEncoder()
        self.blend_encoder = GradualStyleEncoder(4)
        
        self.c_net = C_Net()
        self.new_generator = My_Generator(64, 128, 512)

    def forward(self, gray_image, color_image, color_flip_image, gray_one_hot, color_flip_one_hot):
        gray_feature, _ = self.structure_encoder(gray_image)
        color_flip_feature, _ = self.color_encoder(color_flip_image)
        
        _gray_image = F.interpolate(gray_image, (128, 128))
        _gray_feature = F.interpolate(gray_feature, (128, 128))
        _color_flip_feature = F.interpolate(color_flip_feature, (128, 128))
        _color_flip_image = F.interpolate(color_flip_image, (128, 128))
        _gray_one_hot = F.interpolate(gray_one_hot, (128, 128))
        _color_flip_one_hot = F.interpolate(color_flip_one_hot, (128,128))
        
        # interpolate for c_net
        # output shape : b 3 512 512
        color_reference_image = self.c_net(_gray_feature, _color_flip_feature, _color_flip_image, _gray_one_hot, _color_flip_one_hot)
        _color_reference_image = F.interpolate(color_reference_image, (512, 512))
        
        # blend color and structure
        # input : b 4(3+1) 128 128 / output : b 64 128 128
        blended_feature_map, blended_image = self.blend_encoder(torch.cat((color_reference_image, _gray_image), dim=1))
        _blended_image = F.interpolate(blended_image, (512,512))
        
        # get skin mean feature map
        # output : b 64 128 128
        mean_skin_feature_map, _, _ = self.fill_innerface_with_skin_mean(blended_feature_map, _gray_one_hot)
        _mean_skin_feature_map = self.inference(mean_skin_feature_map, _gray_feature, _gray_one_hot,  _gray_feature, _gray_one_hot, _gray_feature, _gray_one_hot, _gray_feature, _gray_one_hot)
        
        result = self.new_generator(_mean_skin_feature_map, _mean_skin_feature_map)
        head_masks = torch.sum(gray_one_hot[:,1:], dim=1, keepdim=True)
        result = result * head_masks + color_image * (1-head_masks)

        return result, _color_reference_image, _blended_image
    
    def fill_innerface_with_skin_mean(self, feature_map, mask):
        b, c, _, _ = feature_map.size()
        
        # skin mean
        _feature_map = torch.zeros_like(feature_map)
        head_masks, skin_means = [], []
        for batch_idx in range(b):
            for label_idx in [1]:
                _skin_mask = mask[batch_idx, label_idx].unsqueeze(0)
                inner_face_mask = torch.sum(mask[batch_idx,1:10], dim=0)
                
                skin_area = torch.masked_select(feature_map[batch_idx],_skin_mask.bool()).reshape(c,-1)
                inner_face_pixel = torch.sum(inner_face_mask)
                _inner_face_mask = inner_face_mask.unsqueeze(0)
                skin_mean = skin_area.mean(1)
                ch_skin_area = skin_mean.reshape(-1,1).repeat(1,int(inner_face_pixel.item()))
                _feature_map[batch_idx].masked_scatter_(inner_face_mask.bool(), ch_skin_area)

                skin_means.append(skin_mean)
                
            _feature_map[batch_idx] = feature_map[batch_idx] * (1 - _inner_face_mask) + _feature_map[batch_idx] * _inner_face_mask
            
            _hair_mask = mask[batch_idx, 10].unsqueeze(0)
            head_masks.append((_inner_face_mask + _hair_mask))
        head_masks = torch.stack(head_masks,dim=0)
        return _feature_map, head_masks, skin_means

    def inference(self, base_feature_map, eye_brow_feature_map, eye_brow_one_hot, eye_feature_map, eye_one_hot, nose_feature_map, nose_one_hot, mouth_feature_map, mouth_one_hot):
        eye_brow_mask = torch.sum(eye_brow_one_hot[:,2:4], dim=1, keepdim=True)
        eye_mask = torch.sum(eye_one_hot[:,4:6], dim=1, keepdim=True)
        mouth_mask = torch.sum(mouth_one_hot[:,6:9], dim=1, keepdim=True)
        nose_mask = nose_one_hot[:,9].unsqueeze(1)
        
        # switched_feature_map = torch.zeros_like(base_feature_map)
        switched_feature_map = base_feature_map
        switched_feature_map = switched_feature_map * (1 - eye_brow_mask) + eye_brow_feature_map * eye_brow_mask
        switched_feature_map = switched_feature_map * (1 - eye_mask) + eye_feature_map * eye_mask
        switched_feature_map = switched_feature_map * (1 - nose_mask) + nose_feature_map * nose_mask
        switched_feature_map = switched_feature_map * (1 - mouth_mask) + mouth_feature_map * mouth_mask

        return switched_feature_map
    

    def transpose_components(self, structure, feature_map, mask):
        b, c, _, _ = feature_map.size()
        
        transposed_gray = torch.zeros_like(structure)
        transposed_feature_map = torch.zeros_like(feature_map)
        transposed_mask = torch.zeros_like(mask)
        
        inner_face_mask = torch.sum(mask[:,1:10], dim=1)
        skin_mask = mask[:,1]
        bg_mask = mask[:,0]
        l_eye_brow_mask = mask[:,2].unsqueeze(1)
        r_eye_brow_mask = mask[:,3].unsqueeze(1)
        l_eye_mask = mask[:,4].unsqueeze(1)
        r_eye_mask = mask[:,5].unsqueeze(1)
        mouth_mask = torch.sum(mask[:,6:9], dim=1).unsqueeze(1)
        nose_mask = mask[:,9].unsqueeze(1)
        
        # fill out of inner face region  with original values
        transposed_gray = transposed_gray * inner_face_mask.unsqueeze(1) + structure * (1 - inner_face_mask).unsqueeze(1)
        transposed_feature_map = inner_face_mask.unsqueeze(1) * transposed_feature_map + (1 - inner_face_mask.unsqueeze(1)) * feature_map
        transposed_mask[:,0] = mask[:,0]
        transposed_mask[:,1] = inner_face_mask
        transposed_mask[:,10] = mask[:,10]
        transposed_mask[:,11] = mask[:,11]
        
        # get skin mean
        for batch_idx in range(b):
            _skin_mask = skin_mask[batch_idx].unsqueeze(0)
            
            skin_area = torch.masked_select(feature_map[batch_idx], _skin_mask.bool()).reshape(c,-1)
            inner_face_pixel = torch.sum(inner_face_mask[batch_idx])
            skin_mean = skin_area.mean(1)
            ch_skin_area = skin_mean.reshape(-1,1).repeat(1,int(inner_face_pixel.item()))
            transposed_feature_map[batch_idx].masked_scatter_(inner_face_mask[batch_idx].bool(), ch_skin_area)

        mask_list = [l_eye_brow_mask, r_eye_brow_mask, l_eye_mask, r_eye_mask, mouth_mask, nose_mask]
        index_list = [[2],[3],[4],[5],[6,7,8],[9]]
        for component_mask, indexes in zip(mask_list, index_list):
            component_feature_map = feature_map * component_mask
            
            # roll
            # y_roll, x_roll = random.randrange(-5,5), random.randrange(-5,5)
            y_roll, x_roll =0,0
            _component_mask = torch.roll(component_mask, shifts=(y_roll, x_roll), dims=(-2, -1))
            _component_feature_map = torch.roll(component_feature_map, shifts=(y_roll, x_roll), dims=(-2, -1))
            
            # interpolate
            # y_multi, x_multi = random.uniform(0.8,1.2), random.uniform(0.8,1.2)       
            y_multi, x_multi = 1,1      
            _component_mask = F.interpolate(_component_mask, scale_factor=(y_multi, x_multi))
            _component_feature_map = F.interpolate(_component_feature_map, scale_factor=(y_multi, x_multi))

            _component_mask = transforms.CenterCrop(512)(_component_mask)
            _component_feature_map = transforms.CenterCrop(512)(_component_feature_map)
            
            # overwrite resized component feature map
            transposed_feature_map = _component_mask * _component_feature_map + (1 - _component_mask) * transposed_feature_map
            
            # for image visualization
            component_gray = structure * component_mask
            _component_gray = torch.roll(component_gray, shifts=(y_roll, x_roll), dims=(-2, -1))
            _component_gray = F.interpolate(_component_gray, scale_factor=(y_multi, x_multi))
            _component_gray = transforms.CenterCrop(512)(_component_gray)
            
            transposed_gray = _component_mask * _component_gray + (1 - _component_mask) * transposed_gray
            transposed_mask[:,1] -= _component_mask.squeeze()
            # #@# for mask...
            for index in indexes:
                index_mask = mask[:,index].unsqueeze(1)
                _index_mask = torch.roll(index_mask, shifts=(y_roll, x_roll), dims=(-2, -1))
                _index_mask = F.interpolate(_index_mask, scale_factor=(y_multi, x_multi))
                _index_mask = transforms.CenterCrop(512)(_index_mask)
                transposed_mask[:,index] = _index_mask.squeeze()
                
        return transposed_feature_map, transposed_gray, transposed_mask
