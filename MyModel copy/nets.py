# from ctypes import Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# from MyModel.sub_nets.C_NET import C_Net
from MyModel.sub_nets.New_Generator import My_Generator
from packages import New_G

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.crop_size = 128
        self.mid_size = 256
        self.img_size = 512
        # self.structure_encoder = GradualStyleEncoder()
        # # self.partial_encoder = GradualStyleEncoder()
        # self.blend_encoder = GradualStyleEncoder(6)
        self.new_g = New_G(128)
        
        
        # self.c_net = C_Net()
        self.new_generator = My_Generator(32, 128, 512)

    def set_dict(self):

        self.base = {
            "color_ref": {
                "C_img": None,
                "G_img": None,
                "O_mask": None,
            },

            "skin_ref": {
                "index": 1,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "G_img": None,
                "C_img": None,
                "O_mask": torch.zeros((self.batch_size, 12, self.mid_size ,self.mid_size), device='cuda'),
                "B_mask": torch.zeros((self.batch_size, 1, self.mid_size ,self.mid_size), device='cuda'),
                "OP_mask": torch.zeros((self.batch_size, 12, self.mid_size ,self.mid_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.mid_size ,self.mid_size), device='cuda'), # Binary, partial mask
            },

            "fake":{
                "C_feature": torch.zeros((self.batch_size, 32, self.mid_size ,self.mid_size), device='cuda'), # plz check channel size
                "C_feature_mix": torch.zeros((self.batch_size, 32, self.mid_size ,self.mid_size), device='cuda'), # plz check channel size
                'C_feature_union': torch.zeros((self.batch_size, 32, self.mid_size, self.mid_size), device='cuda'),
                'O_mask': torch.zeros((self.batch_size, 12, self.mid_size, self.mid_size), device='cuda'),
                'B_mask': torch.zeros((self.batch_size, 1, self.mid_size, self.mid_size), device='cuda'),
                'O_mask_union': torch.zeros((self.batch_size, 12, self.mid_size, self.mid_size), device='cuda'),
                'B_mask_union': torch.zeros((self.batch_size, 1, self.mid_size, self.mid_size), device='cuda'),

            }
        }
        self.comp = {
            "Lbrow_ref": {
                
                "index": 2,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "C_img": None,
                "G_img": None,
                "B_mask": None,
                "O_mask": None,
                "CPM_img": torch.zeros((self.batch_size, 3, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "GPM_img": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "OP_mask": torch.zeros((self.batch_size, 12, self.crop_size ,self.crop_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Binary, partial mask
            },

            "Rbrow_ref": {
                
                "index": 3,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "C_img": None,
                "G_img": None,
                "B_mask": None,
                "O_mask": None,
                "CPM_img": torch.zeros((self.batch_size, 3, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "GPM_img": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "OP_mask": torch.zeros((self.batch_size, 12, self.crop_size ,self.crop_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Binary, partial mask
            },

            "Leye_ref": {
                
                "index": 4,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "C_img": None,
                "G_img": None,
                "B_mask": None,
                "O_mask": None,
                "CPM_img": torch.zeros((self.batch_size, 3, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "GPM_img": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "OP_mask": torch.zeros((self.batch_size, 12, self.crop_size ,self.crop_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Binary, partial mask
            },

            "Reye_ref": {
                
                "index": 5,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "C_img": None,
                "G_img": None,
                "B_mask": None,
                "O_mask": None,
                "CPM_img": torch.zeros((self.batch_size, 3, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "GPM_img": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "OP_mask": torch.zeros((self.batch_size, 12, self.crop_size ,self.crop_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Binary, partial mask
            },

            "nose_ref": {
                
                "index": 9,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "C_img": None,
                "G_img": None,
                "B_mask": None,
                "O_mask": None,
                "CPM_img": torch.zeros((self.batch_size, 3, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "GPM_img": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "OP_mask": torch.zeros((self.batch_size, 12, self.crop_size ,self.crop_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Binary, partial mask
            },

            "mouth_ref": {
                
                "index": 6,
                "scale": torch.zeros((self.batch_size, 2), device='cuda'), # scale_x, scale_y
                "shift": torch.zeros((self.batch_size, 2), device='cuda'), # shift_x, shift_y
                "center": torch.zeros((self.batch_size, 2), device='cuda'), # center_x, center_y

                "G_img": None,
                "G_img": None,
                "B_mask": None,
                "O_mask": None, # B_mask.shape: [B, 1, H, W]
                "CPM_img": torch.zeros((self.batch_size, 3, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "GPM_img": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Gray, Partial, Masked image, 
                "OP_mask": torch.zeros((self.batch_size, 12, self.crop_size ,self.crop_size), device='cuda'), # One-hot, partial mask
                "BP_mask": torch.zeros((self.batch_size, 1, self.crop_size ,self.crop_size), device='cuda'), # Binary, partial mask
            },
        }

    def set_components(self, C_imgs, G_imgs, O_masks):

        # color_ref, skin_ref, Lbrow_ref, Rbrow_ref, Leye_ref, Reye_ref, nose_ref, mouth_ref

        self.base['color_ref']['C_img'] = torch.flip(C_imgs[0], dims=(-1,)) # [B 3 mH mW]
        self.base['color_ref']['O_mask'] = torch.flip(O_masks[0], dims=(-1,)) # [B 12 mH mW]
        # self.base['color_ref']['C_feature'] = self.structure_encoder(self.base['color_ref']['C_img'])[0] # [B 3 mH mW] -> [B 64 mH mW]
        
        self.base['skin_ref']['C_img'] = C_imgs[1] # [B 3 mH mW]
        self.base['skin_ref']['G_img'] = G_imgs[1] # [B 1 mH mW]
        self.base['skin_ref']['O_mask'] = O_masks[1] # [B 12 mH mW]

        for component_name, C_img, G_img, O_mask in zip(self.comp, C_imgs[2:], G_imgs[2:], O_masks[2:]):
            comp = self.comp[component_name]
            index = comp['index']
            comp['C_img'] = C_img # [B 3 mH mW]
            comp['G_img'] = G_img # [B 1 mH mW]
            comp['O_mask'] = O_mask # [B 12 mH mW]
            comp['B_mask'] = O_mask[:, index].unsqueeze(1) # [B 1 mH mW]

    def forward(self, G_imgs, C_imgs, O_masks):
        self.batch_size, _, _, _ = G_imgs[0].size()
        self.set_dict()
        self.set_components(G_imgs, C_imgs, O_masks)
        
        # self.base['skin_ref']['C_feature'] = self.structure_encoder(self.base['skin_ref']['C_img'])[0] # [B 3 mH mW] -> [B 64 mH mW]
        # self.base['skin_ref']['C_ref'] = self.do_RC(self.base['color_ref'], self.base['skin_ref'], 128) # [B 3 128 128] # 64? 128? 
        # self.base['skin_ref']['C_feature_colored'], self.base['skin_ref']['C_img_colored'] =\
        #     self.blend_encoder(torch.cat((self.base['skin_ref']['C_ref'], self.base['skin_ref']['C_img']),dim=1)) # [B 64 128 128], [B 3 128 128]

        # b, 3, 256, 256 / b 32 256 256
        self.base['skin_ref']['C_img_colored'], self.base['skin_ref']['C_feature_colored'], self.base['skin_ref']['C_ref'] = \
            self.new_g.generator(self.base['skin_ref']['G_img'].repeat(1,3,1,1), self.base['color_ref']['C_img'], self.base['skin_ref']['O_mask'], self.base['color_ref']['O_mask'], 64) # 
        # self.base['skin_ref']['C_feature_colored'] = F.interpolate(self.base['skin_ref']['C_feature_colored'], (128,128))
        
        for component_name in self.comp:
            comp = self.comp[component_name]
            # get center of component mask
            for b_idx in range(self.batch_size):
                ys, xs = torch.where(comp['B_mask'][b_idx, 0]==1)
                cx, cy = xs.type(torch.float32).mean().type(torch.int32), ys.type(torch.float32).mean().type(torch.int32)
                comp['center'][b_idx][0], comp['center'][b_idx][1] = cx, cy # [B 1 mH mW] 기준으로 하는 cx, cy
                    
            self.get_partial(comp)
            
            # # get GPM_color_blend
            # comp['CPM_feature'] = self.structure_encoder(comp['CPM_img'])[0] # [B 64 cH cW] # color, partial, masked image
            # comp['CPM_ref'] = self.do_RC(self.base['color_ref'], comp, 64) # [B 3 cH cW] # 64? 128?
            # # print(comp['CPM_ref'].size())
            # # print(comp['CPM_img'].size())
            # comp['CPM_feature_colored'] = self.blend_encoder(torch.cat((comp['CPM_ref'], comp['CPM_img']), dim=1))[0] # [B 64 cH cW] # gray, partial, masked image
            
            # with torch.no_grad():
                
            comp['CPM_ref'], comp['CPM_feature_colored'], _ = \
                self.new_g.generator(comp['GPM_img'].repeat(1,3,1,1), self.base['color_ref']['C_img'], comp['OP_mask'], self.base['color_ref']['O_mask'], 128) # 

            comp['CPM_ref'] = F.interpolate(comp['CPM_ref'],(128,128))
            comp['CPM_feature_colored'] = F.interpolate(comp['CPM_feature_colored'],(128,128))
                
            # update new_feature_map, union_mask, fake_one_hot 
            self.add_partial(comp)
        self.base['fake']['C_feature_mix'] = (1 - self.base['fake']['B_mask_union']) * self.base['skin_ref']['C_feature_colored'] + self.base['fake']['B_mask_union'] * self.base['fake']['C_feature_union']
        # import pdb; pdb.set_trace()

        result = self.new_generator(self.base['fake']['C_feature_mix'], self.base['fake']['C_feature_mix'])
        head_masks = torch.sum(self.base['skin_ref']['O_mask'][:, 1:], dim=1, keepdim=True)
        head_masks = F.interpolate(head_masks, (self.img_size, self.img_size))
        result = F.interpolate(result, (self.img_size, self.img_size))
        C_img = F.interpolate(self.base['skin_ref']['C_img'], (self.img_size, self.img_size))
        result = result * head_masks + C_img * (1-head_masks)
        
        return result, self.base, self.comp
    
    def get_partial(self, comp):

        # resize b for  zero canvas * source image, 배수 -> 중앙 위치 파악하기, 마지막에 centor crop 128

        index = comp['index'] 
        C_img = comp['C_img'] # [B 3 mH mW]
        G_img = comp['G_img'] # [B 3 mH mW]
        B_mask = comp['B_mask'] # [B 1 mH mW]
        
        for b_idx in range(self.batch_size):
            if B_mask[b_idx].sum():
                cx, cy = comp['center'][b_idx]
                scale_x, scale_y = 1, 1
                # scale_x, scale_y = random.uniform(0.9,1.1), random.uniform(0.9,1.1) 
                comp['scale'][b_idx][0], comp['scale'][b_idx][1] = scale_x, scale_y
                half_x, half_y = int(self.crop_size//2 * scale_x), int(self.crop_size//2 * scale_y)
                cx, cy = int(cx), int(cy)
                # 
                try:
                    CP_img     = transforms.functional.crop(F.pad(C_img[b_idx],(64,64,64,64)), top=64+cy-self.crop_size//2, left=64+cx-self.crop_size//2, height=self.crop_size, width=self.crop_size)
                    GP_img     = transforms.functional.crop(F.pad(G_img[b_idx],(64,64,64,64)), top=64+cy-self.crop_size//2, left=64+cx-self.crop_size//2, height=self.crop_size, width=self.crop_size)
                    BP_mask     = transforms.functional.crop(F.pad(B_mask[b_idx],(64,64,64,64)), top=64+cy-self.crop_size//2, left=64+cx-self.crop_size//2, height=self.crop_size, width=self.crop_size)
                    
                    # bp_mask_ys, bp_mask_xs = torch.where(BP_mask[b_idx,0]==1)
                    # bp_mask_mid_y, bp_mask_mid_x = bp_mask_ys.type(torch.float32).mean().type(torch.int32), bp_mask_xs.type(torch.float32).mean().type(torch.int32)
                
                    
                    # _CP_img = C_img[b_idx, :, cy-half_y:cy+half_y,cx-half_x:cx+half_x] # [3, self.crop_size * scale_y, self.crop_size * scale_x]
                    # _GP_img = G_img[b_idx, :, cy-half_y:cy+half_y,cx-half_x:cx+half_x] # [1, self.crop_size * scale_y, self.crop_size * scale_x]
                    # _BP_mask = B_mask[b_idx, :, cy-half_y:cy+half_y,cx-half_x:cx+half_x] # [1, self.crop_size * scale_y, self.crop_size * scale_x]
                    
                    CP_img = F.interpolate(CP_img.unsqueeze(0), (self.crop_size, self.crop_size)).squeeze() # [3, cH cW]
                    GP_img = F.interpolate(GP_img.unsqueeze(0), (self.crop_size, self.crop_size)).squeeze() # [1, cH cW]
                    BP_mask = F.interpolate(BP_mask.unsqueeze(0), (self.crop_size, self.crop_size)).squeeze() # [cH cW]
                    
                    comp['OP_mask'][b_idx, index] = BP_mask # [cH cW]
                    comp['OP_mask'][b_idx, 1] = (1 - BP_mask) # [cH cW] # '1' is skin index
                    comp['BP_mask'][b_idx] = BP_mask.unsqueeze(0) # [1, cH cW]
                    comp['CPM_img'][b_idx] = CP_img * BP_mask.unsqueeze(0) # Color, Partial, Masked image # [3, cH cW]
                    comp['GPM_img'][b_idx] = (GP_img * BP_mask).unsqueeze(0) # Color, Partial, Masked image # [3, cH cW]
                
                except:
                    import pdb;pdb.set_trace()
                
                    
                
                
    def add_partial(self, comp):
        # new_feature_map, union_mask, self.ref_dict[component_name]['GPM_feature'], self.ref_dict[component_name]['OP_mask'], fake_one_hot, O_mask
        # b for  center crop 512 x,y roll

        O_mask_skin = self.base['skin_ref']['O_mask'] # One-hot mask of skin reference image # [B 12 mH mW]
        index = comp['index'] 
        BP_mask = comp['BP_mask'] # [B 1 cH cW]
        CPM_feature_colored = comp['CPM_feature_colored'] # [B 64 cH cW] 64
            
        BP_mask_padded = transforms.CenterCrop(self.mid_size)(BP_mask) # [B 1 mH mW]   / 34.5792, 32.2471
        CPM_feature_colored_padded = transforms.CenterCrop(self.mid_size)(CPM_feature_colored) # [B 64 mH mW] 128

        for b_idx in range(self.batch_size):
            cx, cy = int(comp['center'][b_idx][0]), int(comp['center'][b_idx][1]) # before 65 99 / roll(after) 65 101
            BP_mask_rollback = torch.roll(BP_mask_padded[b_idx], shifts=(cy-self.mid_size//2, cx-self.mid_size//2), dims=(-2, -1)) # [1 mH mW]
            # if v
            CPM_feature_colored_rollback = torch.roll(CPM_feature_colored_padded[b_idx], shifts=(cy-self.mid_size//2, cx-self.mid_size//2), dims=(-2, -1)) # [64 mH mW]
            
            shift_x, shift_y = 0, 0
            # shift_x, shift_y = random.randrange(-3,3), random.randrange(-3,3)
            comp['shift'][b_idx][0], comp['shift'][b_idx][1] = shift_x, shift_y
            BP_mask_shifted = torch.roll(BP_mask_rollback, shifts=(shift_y, shift_x), dims=(-2, -1)) # [1 mH mW]
            CPM_feature_colored_shifted = torch.roll(CPM_feature_colored_rollback, shifts=(shift_y, shift_x), dims=(-2, -1)) # [64 mH mW]
            
            # union_mask = torch.logical_or(O_mask_skin[b_idx, index], BP_mask_shifted[0]).int()  # [mH mW]
            union_mask = (O_mask_skin[b_idx, index]+BP_mask_shifted[0]).clamp(0,1)  # [mH mW]
            self.base['fake']['B_mask'][b_idx] = BP_mask_shifted # [1 mH mW]
            self.base['fake']['O_mask'][b_idx][index] = BP_mask_shifted[0] # [mH mW]
            self.base['fake']['B_mask_union'][b_idx, 0] += union_mask # [mH mW]
            self.base['fake']['O_mask_union'][b_idx][index] = union_mask # [mH mW]
            self.base['fake']['C_feature_union'][b_idx] += CPM_feature_colored_shifted * union_mask.unsqueeze(0) # [64 mH mW]

    def do_RC(self, color_ref, comp_ref, size):

        color_mask = F.interpolate(color_ref['O_mask'], (size,size))
        color_feature = F.interpolate(color_ref['C_feature'], (size,size))
        color_img = F.interpolate(color_ref['C_img'], (size,size))
        
        if 'C_feature' in comp_ref: # for skin ref and color ref
            comp_feature = F.interpolate(comp_ref['C_feature'], (size,size))
            comp_mask = F.interpolate(comp_ref['O_mask'],(size,size))
        else: # for others
            comp_feature = F.interpolate(comp_ref['CPM_feature'], (size,size))
            comp_mask = F.interpolate(comp_ref['OP_mask'],(size,size))

        canvas = torch.ones_like(color_img) * -1
        b, c, _, _ = comp_feature.size()

        for b_idx in range(b):
            for c_idx in range(1, 12):
                if comp_mask[b_idx,c_idx].sum() == 0 or comp_mask[b_idx,c_idx].sum() == 1 or color_mask[b_idx,c_idx].sum() == 0 or color_mask[b_idx,c_idx].sum() == 1:
                    continue

                comp_matrix = torch.masked_select(comp_feature[b_idx], comp_mask[b_idx,c_idx].bool()).reshape(c, -1) # 64, pixel_num_A
                comp_matrix_bar = comp_matrix - comp_matrix.mean(1, keepdim=True) # (64, 1)
                comp_matrix_norm = torch.norm(comp_matrix_bar, dim=0, keepdim=True)
                comp_matrix_ = comp_matrix_bar / comp_matrix_norm

                rgb_matrix = torch.masked_select(color_feature[b_idx], color_mask[b_idx,c_idx].bool()).reshape(c, -1) # 64, pixel_num_B
                rgb_matrix_bar = rgb_matrix - rgb_matrix.mean(1, keepdim=True) # 64, pixel_num_B
                rgb_matrix_norm = torch.norm(rgb_matrix_bar, dim=0, keepdim=True)
                rgb_matrix_ = rgb_matrix_bar / rgb_matrix_norm
               
                correlation_matrix = torch.matmul(comp_matrix_.transpose(0,1), rgb_matrix_)
                if torch.isnan(correlation_matrix).sum():
                    import pdb; pdb.set_trace()
                correlation_matrix = F.softmax(correlation_matrix,dim=1)
                
                rgb_pixels = torch.masked_select(color_img[b_idx], color_mask[b_idx,c_idx].bool()).reshape(3,-1)
                colorized_matrix = torch.matmul(correlation_matrix, rgb_pixels.transpose(0,1)).transpose(0,1)

                canvas[b_idx].masked_scatter_(comp_mask[b_idx,c_idx].bool(), colorized_matrix) # 3 128 128
                # import pdb; pdb.set_trace()

        return canvas

    # def fill_innerface_with_skin_mean(self, feature_map, mask):
    #     b, c, _, _ = feature_map.size()
        
    #     # skin mean
    #     _feature_map = torch.zeros_like(feature_map)
    #     head_masks, skin_means = [], []
    #     for batch_idx in range(b):
    #         for label_idx in [1]:
    #             _skin_mask = mask[batch_idx, label_idx].unsqueeze(0)
    #             inner_face_mask = torch.sum(mask[batch_idx,1:10], dim=0)
                
    #             skin_area = torch.masked_select(feature_map[batch_idx],_skin_mask.bool()).reshape(c,-1)
    #             inner_face_pixel = torch.sum(inner_face_mask)
    #             _inner_face_mask = inner_face_mask.unsqueeze(0)
    #             skin_mean = skin_area.mean(1)
    #             ch_skin_area = skin_mean.reshape(-1,1).repeat(1,int(inner_face_pixel.item()))
    #             _feature_map[batch_idx].masked_scatter_(inner_face_mask.bool(), ch_skin_area)

    #             skin_means.append(skin_mean)
                
    #         _feature_map[batch_idx] = feature_map[batch_idx] * (1 - _inner_face_mask) + _feature_map[batch_idx] * _inner_face_mask
            
    #         _hair_mask = mask[batch_idx, 10].unsqueeze(0)
    #         head_masks.append((_inner_face_mask + _hair_mask))
    #     head_masks = torch.stack(head_masks,dim=0)
    #     return _feature_map, head_masks, skin_means

    # def inference(self, base_feature_map, eye_brow_feature_map, eye_brow_one_hot, eye_feature_map, eye_one_hot, nose_feature_map, nose_one_hot, mouth_feature_map, mouth_one_hot):
    #     eye_brow_mask = torch.sum(eye_brow_one_hot[:,2:4], dim=1, keepdim=True)
    #     eye_mask = torch.sum(eye_one_hot[:,4:6], dim=1, keepdim=True)
    #     mouth_mask = torch.sum(mouth_one_hot[:,6:9], dim=1, keepdim=True)
    #     nose_mask = nose_one_hot[:,9].unsqueeze(1)
        
    #     # switched_feature_map = torch.zeros_like(base_feature_map)
    #     switched_feature_map = base_feature_map
    #     switched_feature_map = switched_feature_map * (1 - eye_brow_mask) + eye_brow_feature_map * eye_brow_mask
    #     switched_feature_map = switched_feature_map * (1 - eye_mask) + eye_feature_map * eye_mask
    #     switched_feature_map = switched_feature_map * (1 - nose_mask) + nose_feature_map * nose_mask
    #     switched_feature_map = switched_feature_map * (1 - mouth_mask) + mouth_feature_map * mouth_mask

    #     return switched_feature_map
    

    # def transpose_components(self, structure, feature_map, mask):
    #     b, c, _, _ = feature_map.size()
        
    #     transposed_gray = torch.zeros_like(structure)
    #     transposed_feature_map = torch.zeros_like(feature_map)
    #     transposed_mask = torch.zeros_like(mask)
        
    #     inner_face_mask = torch.sum(mask[:,1:10], dim=1)
    #     skin_mask = mask[:,1]
    #     bg_mask = mask[:,0]
    #     l_eye_brow_mask = mask[:,2].unsqueeze(1)
    #     r_eye_brow_mask = mask[:,3].unsqueeze(1)
    #     l_eye_mask = mask[:,4].unsqueeze(1)
    #     r_eye_mask = mask[:,5].unsqueeze(1)
    #     mouth_mask = torch.sum(mask[:,6:9], dim=1).unsqueeze(1)
    #     nose_mask = mask[:,9].unsqueeze(1)
        
    #     # fill out of inner face region  with original values
    #     transposed_gray = transposed_gray * inner_face_mask.unsqueeze(1) + structure * (1 - inner_face_mask).unsqueeze(1)
    #     transposed_feature_map = inner_face_mask.unsqueeze(1) * transposed_feature_map + (1 - inner_face_mask.unsqueeze(1)) * feature_map
    #     transposed_mask[:,0] = mask[:,0]
    #     transposed_mask[:,1] = inner_face_mask
    #     transposed_mask[:,10] = mask[:,10]
    #     transposed_mask[:,11] = mask[:,11]
        
    #     # get skin mean
    #     for batch_idx in range(b):
    #         _skin_mask = skin_mask[batch_idx].unsqueeze(0)
            
    #         skin_area = torch.masked_select(feature_map[batch_idx], _skin_mask.bool()).reshape(c,-1)
    #         inner_face_pixel = torch.sum(inner_face_mask[batch_idx])
    #         skin_mean = skin_area.mean(1)
    #         ch_skin_area = skin_mean.reshape(-1,1).repeat(1,int(inner_face_pixel.item()))
    #         transposed_feature_map[batch_idx].masked_scatter_(inner_face_mask[batch_idx].bool(), ch_skin_area)

    #     mask_list = [l_eye_brow_mask, r_eye_brow_mask, l_eye_mask, r_eye_mask, mouth_mask, nose_mask]
    #     index_list = [[2],[3],[4],[5],[6,7,8],[9]]
    #     for component_mask, indexes in zip(mask_list, index_list):
    #         component_feature_map = feature_map * component_mask
            
    #         # roll
    #         # y_roll, x_roll = random.randrange(-5,5), random.randrange(-5,5)
    #         y_roll, x_roll =0,0
    #         _component_mask = torch.roll(component_mask, shifts=(y_roll, x_roll), dims=(-2, -1))
    #         _component_feature_map = torch.roll(component_feature_map, shifts=(y_roll, x_roll), dims=(-2, -1))
            
    #         # interpolate
    #         # y_multi, x_multi = 1,1      
    #         y_multi, x_multi = random.uniform(0.8,1.2), random.uniform(0.8,1.2)       
    #         _component_mask = F.interpolate(_component_mask, scale_factor=(y_multi, x_multi))
    #         _component_feature_map = F.interpolate(_component_feature_map, scale_factor=(y_multi, x_multi))

    #         _component_mask = transforms.CenterCrop(512)(_component_mask)
    #         _component_feature_map = transforms.CenterCrop(512)(_component_feature_map)
            
    #         # overwrite resized component feature map
    #         transposed_feature_map = _component_mask * _component_feature_map + (1 - _component_mask) * transposed_feature_map
            
    #         # for image visualization
    #         component_gray = structure * component_mask
    #         _component_gray = torch.roll(component_gray, shifts=(y_roll, x_roll), dims=(-2, -1))
    #         _component_gray = F.interpolate(_component_gray, scale_factor=(y_multi, x_multi))
    #         _component_gray = transforms.CenterCrop(512)(_component_gray)
            
    #         transposed_gray = _component_mask * _component_gray + (1 - _component_mask) * transposed_gray
    #         transposed_mask[:,1] -= _component_mask.squeeze()
    #         # #@# for mask...
    #         for index in indexes:
    #             index_mask = mask[:,index].unsqueeze(1)
    #             _index_mask = torch.roll(index_mask, shifts=(y_roll, x_roll), dims=(-2, -1))
    #             _index_mask = F.interpolate(_index_mask, scale_factor=(y_multi, x_multi))
    #             _index_mask = transforms.CenterCrop(512)(_index_mask)
    #             transposed_mask[:,index] = _index_mask.squeeze()
                
    #     return transposed_feature_map, transposed_gray, transposed_mask
