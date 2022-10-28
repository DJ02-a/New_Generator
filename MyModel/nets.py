import torch
import torch.nn as nn
import torch.nn.functional as F

from MyModel.sub_nets.C_NET import C_Net
from MyModel.sub_nets.New_Generator import My_Generator
from MyModel.utils.PSP import GradualStyleEncoder

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.structure_encoder = GradualStyleEncoder(1)
        self.color_encoder = GradualStyleEncoder()
        
        self.c_net = C_Net()
        self.new_generator = My_Generator()

    def forward(self, gray_image, color_image, color_flip_image, gray_one_hot, color_flip_one_hot):
        
        gray_feature = self.structure_encoder(gray_image)
        _gray_feature = F.interpolate(gray_feature, (128,128))
        _gray_one_hot = F.interpolate(gray_one_hot, (128,128))
        
        color_flip_feature = self.color_encoder(color_flip_image)
        _color_flip_feature = F.interpolate(color_flip_feature, (128,128))
        _color_flip_image = F.interpolate(color_flip_image, (128,128))
        _color_flip_one_hot = F.interpolate(color_flip_one_hot, (128,128))
        
        eye_brow_mask = (_gray_one_hot[:,2] + _gray_one_hot[:,3]).unsqueeze(1)
        eye_mask = (_gray_one_hot[:,4] + _gray_one_hot[:,5]).unsqueeze(1)
        nose_mask = (_gray_one_hot[:,9]).unsqueeze(1)
        mouth_mask = (_gray_one_hot[:,6] + _gray_one_hot[:,7] + _gray_one_hot[:,8]).unsqueeze(1)
        
        mean_skin_feature_map, head_masks = self.fill_innerface_with_skin_mean(_gray_feature, _gray_one_hot)
        _mean_skin_feature_map = self.inference(mean_skin_feature_map, _gray_feature, eye_brow_mask, _gray_feature, eye_mask, _gray_feature, nose_mask, _gray_feature, mouth_mask)
        color_reference_image = self.c_net(_mean_skin_feature_map, _color_flip_feature, _color_flip_image, _gray_one_hot, _color_flip_one_hot)

            
        mix_features = torch.cat((_mean_skin_feature_map, color_reference_image), dim=1)
        result = self.new_generator(mix_features)
        
        head_masks = F.interpolate(head_masks, (512,512), mode='bilinear')
        result = result * head_masks + color_image * (1-head_masks)

        return result, color_reference_image
    
    def fill_innerface_with_skin_mean(self, feature_map, mask):
        b, c, _, _ = feature_map.size()
        
        # skin mean
        _feature_map = torch.zeros_like(feature_map)
        head_masks = []
        for batch_idx in range(b):
            for label_idx in [1]:
                _skin_mask = mask[batch_idx, label_idx].unsqueeze(0)
                inner_face_mask = torch.sum(mask[batch_idx,1:10], dim=0)
                
                skin_area = torch.masked_select(feature_map[batch_idx],_skin_mask.bool()).reshape(c,-1)
                inner_face_pixel = torch.sum(inner_face_mask)
                _inner_face_mask = inner_face_mask.unsqueeze(0)
                ch_skin_area = skin_area.mean(1).reshape(-1,1).repeat(1,int(inner_face_pixel.item()))
                _feature_map[batch_idx].masked_scatter_(inner_face_mask.bool(), ch_skin_area)

            _feature_map[batch_idx] = feature_map[batch_idx] * (1 - _inner_face_mask) + _feature_map[batch_idx] * _inner_face_mask
            
            _hair_mask = mask[batch_idx, 10].unsqueeze(0)
            head_masks.append((_inner_face_mask + _hair_mask))
        head_masks = torch.stack(head_masks,dim=0)
        return _feature_map, head_masks

    def inference(self, base_feature_map, eye_brow_feature_map, eye_brow_mask, eye_feature_map, eye_mask, nose_feature_map, nose_mask, mouth_feature_map, mouth_mask):
        switched_feature_map = torch.zeros_like(base_feature_map)
        switched_feature_map = base_feature_map * (1 - eye_brow_mask) + eye_brow_feature_map * eye_brow_mask
        switched_feature_map = base_feature_map * (1 - eye_mask) + eye_feature_map * eye_mask
        switched_feature_map = base_feature_map * (1 - nose_mask) + nose_feature_map * nose_mask
        switched_feature_map = base_feature_map * (1 - mouth_mask) + mouth_feature_map * mouth_mask

        return switched_feature_map