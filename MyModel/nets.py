from ctypes import Union
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
        self.crop_size = 64
        self.img_size = 128
        self.input_size = input_size
        self.structure_encoder = GradualStyleEncoder(1)
        self.color_encoder = GradualStyleEncoder()
        self.blend_encoder = GradualStyleEncoder(4)
        
        self.c_net = C_Net(self.crop_size)
        self.new_generator = My_Generator(64, 128, 512)



    def forward(self, gray_image, color_image, color_flip_image, gray_one_hot, color_flip_one_hot):
        _gray_image = F.interpolate(gray_image,(self.img_size,self.img_size))
        _color_flip_image = F.interpolate(color_flip_image,(self.img_size,self.img_size))
        _gray_one_hot = F.interpolate(gray_one_hot,(self.img_size,self.img_size))
        _color_flip_one_hot = F.interpolate(color_flip_one_hot,(self.img_size,self.img_size))
        b,c,h,w = _gray_image.size()
        
        # step 1 : get part of face component from gray image and mask
        l_eyebrow_gray = _gray_image
        r_eyebrow_gray = _gray_image
        l_eye_gray = _gray_image
        r_eye_gray = _gray_image
        nose_gray = _gray_image
        mouth_gray = _gray_image
        
        l_eyebrow_gray_one_hot = _gray_one_hot
        r_eyebrow_gray_one_hot = _gray_one_hot
        l_eye_gray_one_hot = _gray_one_hot
        r_eye_gray_one_hot = _gray_one_hot
        nose_gray_one_hot = _gray_one_hot
        mouth_gray_one_hot = _gray_one_hot
        
        # fix 128 128 crop size
        # one hot to component mask
        resized_l_eyebrow, resized_l_eyebrow_one_hot = self.pp_transformation(l_eyebrow_gray, l_eyebrow_gray_one_hot, [2])
        resized_r_eyebrow, resized_r_eyebrow_one_hot = self.pp_transformation(r_eyebrow_gray, r_eyebrow_gray_one_hot, [3])
        resized_l_eye, resized_l_eye_one_hot = self.pp_transformation(l_eye_gray, l_eye_gray_one_hot, [4])
        resized_r_eye, resized_r_eye_one_hot = self.pp_transformation(r_eye_gray, r_eye_gray_one_hot, [5])
        resized_nose, resized_nose_one_hot = self.pp_transformation(nose_gray, nose_gray_one_hot, [9])
        resized_mouth, resized_mouth_one_hot = self.pp_transformation(mouth_gray, mouth_gray_one_hot, [6,7,8])
        
        resized_l_eyebrow_feature, _ = self.structure_encoder(resized_l_eyebrow)
        resized_r_eyebrow_feature, _ = self.structure_encoder(resized_r_eyebrow)
        resized_l_eye_feature, _ = self.structure_encoder(resized_l_eye)
        resized_r_eye_feature, _ = self.structure_encoder(resized_r_eye)
        resized_nose_feature, _ = self.structure_encoder(resized_nose)
        resized_mouth_feature, _ = self.structure_encoder(resized_mouth)
        
        color_flip_feature, _ = self.color_encoder(_color_flip_image)
        
        # C-Net
        l_eyebrow_color_reference_image = self.c_net(resized_l_eyebrow_feature, color_flip_feature, _color_flip_image, resized_l_eyebrow_one_hot, _color_flip_one_hot)
        r_eyebrow_color_reference_image = self.c_net(resized_r_eyebrow_feature, color_flip_feature, _color_flip_image, resized_r_eyebrow_one_hot, _color_flip_one_hot)
        l_eye_color_reference_image = self.c_net(resized_l_eye_feature, color_flip_feature, _color_flip_image, resized_l_eye_one_hot, _color_flip_one_hot)
        r_eye_color_reference_image = self.c_net(resized_r_eye_feature, color_flip_feature, _color_flip_image, resized_r_eye_one_hot, _color_flip_one_hot)
        nose_color_reference_image = self.c_net(resized_nose_feature, color_flip_feature, _color_flip_image, resized_nose_one_hot, _color_flip_one_hot)
        mouth_color_reference_image = self.c_net(resized_mouth_feature, color_flip_feature, _color_flip_image, resized_mouth_one_hot, _color_flip_one_hot)
        
        blend_l_eyebrow_feature, _ = self.blend_encoder(torch.cat((l_eyebrow_color_reference_image, resized_l_eyebrow),dim=1))
        blend_r_eyebrow_feature, _ = self.blend_encoder(torch.cat((r_eyebrow_color_reference_image, resized_r_eyebrow),dim=1))
        blend_l_eye_feature, _ = self.blend_encoder(torch.cat((l_eye_color_reference_image, resized_l_eye),dim=1))
        blend_r_eye_feature, _ = self.blend_encoder(torch.cat((r_eye_color_reference_image, resized_r_eye),dim=1))
        blend_nose_feature, _ = self.blend_encoder(torch.cat((nose_color_reference_image, resized_nose),dim=1))
        blend_mouth_feature, _ = self.blend_encoder(torch.cat((mouth_color_reference_image, resized_mouth),dim=1))
        
        gray_image_feature, _ = self.structure_encoder(_gray_image)
        gray_image_color_reference = self.c_net(gray_image_feature, color_flip_feature, _color_flip_image, _gray_one_hot, _color_flip_one_hot)
        _gray_image_color_reference = F.interpolate(gray_image_color_reference,(self.img_size,self.img_size))
        blend_gray_image_feature, _ = self.blend_encoder(torch.cat((_gray_image_color_reference, _gray_image),dim=1))
        
        new_mask = torch.zeros((b,1,h,w), device=gray_image.device)
        new_feature_map = torch.zeros_like(gray_image_feature, device=gray_image.device)
        new_feature_map, new_mask = self.overwrite_face_component_feature(new_feature_map, new_mask, blend_l_eyebrow_feature, resized_l_eyebrow_one_hot, _gray_one_hot, [2])
        new_feature_map, new_mask = self.overwrite_face_component_feature(new_feature_map, new_mask, blend_r_eyebrow_feature, resized_r_eyebrow_one_hot, _gray_one_hot, [3])
        new_feature_map, new_mask = self.overwrite_face_component_feature(new_feature_map, new_mask, blend_l_eye_feature, resized_l_eye_one_hot, _gray_one_hot, [4])
        new_feature_map, new_mask = self.overwrite_face_component_feature(new_feature_map, new_mask, blend_r_eye_feature, resized_r_eye_one_hot, _gray_one_hot, [5])
        new_feature_map, new_mask = self.overwrite_face_component_feature(new_feature_map, new_mask, blend_nose_feature, resized_nose_one_hot, _gray_one_hot, [9])
        new_feature_map, new_mask = self.overwrite_face_component_feature(new_feature_map, new_mask, blend_mouth_feature, resized_mouth_one_hot, _gray_one_hot, [6,7,8])
        new_feature_map = new_mask * new_feature_map + (1 - new_mask) * blend_gray_image_feature
        
        _new_feature_map = F.interpolate(new_feature_map, (self.img_size,self.img_size))
        result = self.new_generator(_new_feature_map, _new_feature_map)
        head_masks = torch.sum(gray_one_hot[:,1:], dim=1, keepdim=True)
        
        
        result = result * head_masks + color_image * (1-head_masks)

        return result
    
    def pp_transformation(self, source_image, source_one_hot, indexes):
        # resize b for  zero canvas * source image, 배수 -> 중앙 위치 파악하기, 마지막에 centor crop 128
        b, c, h, w = source_one_hot.size()
        
        # get mask
        whole_mask = torch.zeros((b,1,h,w), device=source_one_hot.device)
        component_mask = torch.zeros((b, c, h, w), device=source_one_hot.device)
        center_component_mask = torch.zeros((b,c,self.crop_size,self.crop_size), device=source_one_hot.device)
        center_source_image_part_masked = torch.zeros((b,1,self.crop_size,self.crop_size), device=source_one_hot.device)
        for index in indexes:
            whole_mask[:,0] += source_one_hot[:,index]
            component_mask[:,index] = source_one_hot[:,index]
            
        for b_idx in range(b):
            if whole_mask[b_idx].sum() != 0:
                y_pixels, x_pixels = torch.where(whole_mask[b_idx][0]==1)
                mid_y, mid_x = y_pixels.type(torch.float32).mean().type(torch.int32), x_pixels.type(torch.float32).mean().type(torch.int32)
                whole_mask_part     = transforms.functional.crop(whole_mask[b_idx], top=mid_y-self.crop_size//2, left=mid_x-self.crop_size//2, height=self.crop_size, width=self.crop_size)
                component_mask_part = transforms.functional.crop(component_mask[b_idx], top=mid_y-self.crop_size//2, left=mid_x-self.crop_size//2, height=self.crop_size, width=self.crop_size)
                source_image_part   = transforms.functional.crop(source_image[b_idx], top=mid_y-self.crop_size//2, left=mid_x-self.crop_size//2, height=self.crop_size, width=self.crop_size)
                
                component_mask_part[1] = (1 - whole_mask_part)    # mark skin region
                _source_image_part_masked = source_image_part * whole_mask_part #Q!
                            
                # resize
                y_multi, x_multi = random.uniform(0.8,1.2), random.uniform(0.8,1.2)      
                resized_component_mask_part         = F.interpolate(component_mask_part.unsqueeze(0), scale_factor=(y_multi, x_multi)).squeeze()
                resized_source_image_part_masked    = F.interpolate(_source_image_part_masked.unsqueeze(0), scale_factor=(y_multi, x_multi)).squeeze()
                
                center_component_mask[b_idx]           = transforms.CenterCrop(self.crop_size)(resized_component_mask_part)
                center_source_image_part_masked[b_idx] = transforms.CenterCrop(self.crop_size)(resized_source_image_part_masked)
            else:
                center_component_mask[b_idx] = torch.zeros((c,self.crop_size,self.crop_size),device=component_mask.device)
                center_source_image_part_masked[b_idx] = torch.zeros((1,self.crop_size,self.crop_size),device=source_image.device)

        return center_source_image_part_masked, center_component_mask
    
    def overwrite_face_component_feature(self, canvas, mask_canvas, new_colored_component_feature, new_component_one_hot, origin_one_hot, indexes):
        # b for  center crop 512 x,y roll
        b, _, h, w = origin_one_hot.size()
        new_component_mask = torch.zeros((b,1,self.crop_size,self.crop_size), device=new_colored_component_feature.device)
        origin_masks = torch.zeros((b,1,h,w), device=new_colored_component_feature.device)
        for index in indexes:
            new_component_mask[:,0] += new_component_one_hot[:,index]
            origin_masks[:,0] += origin_one_hot[:, index]
            
        new_component_mask = new_component_mask.clamp(0,1)
        origin_masks = origin_masks.clamp(0,1)
        
        center_new_component_mask = transforms.CenterCrop(h)(new_component_mask)
        center_new_component_feature = transforms.CenterCrop(h)(new_colored_component_feature)
        for b_idx in range(b):
            origin_mask = origin_masks[b_idx]
            y_pixels, x_pixels = torch.where(origin_mask[0]==1)
            
            mid_y, mid_x = y_pixels.type(torch.float32).mean().type(torch.int32), x_pixels.type(torch.float32).mean().type(torch.int32)
            mid_new_component_mask = torch.roll(center_new_component_mask[b_idx], shifts=(mid_y-h//2, mid_x-w//2), dims=(-2, -1))
            mid_new_component_feature = torch.roll(center_new_component_feature[b_idx], shifts=(mid_y-h//2, mid_x-w//2), dims=(-2, -1))
            
            y_roll, x_roll = random.randrange(-5,5), random.randrange(-5,5)
            moved_new_component_mask = torch.roll(mid_new_component_mask, shifts=(y_roll, x_roll), dims=(-2, -1))
            moved_new_component_feature = torch.roll(mid_new_component_feature, shifts=(y_roll, x_roll), dims=(-2, -1))
            
            
            union_mask = (origin_mask + moved_new_component_mask).clamp(0,1) #@#
            mask_canvas[b_idx] += union_mask
            canvas[b_idx] = moved_new_component_feature * union_mask + canvas[b_idx] * (1 - union_mask)
            
        return canvas, mask_canvas
    
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
