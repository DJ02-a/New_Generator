import torch
import torch.nn as nn
import torch.nn.functional as F

class C_Net(nn.Module):
    def __init__(self,):
        super(C_Net, self).__init__()


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


    # def do_RC_part(self, color_ref, comp_ref, size):
    #     index = comp_ref['index']
    #     canvas = torch.ones_like(comp_ref['CPM_img']) * -1
    #     b, c, h, w = comp_ref['CPM_feature'].size()

        
    #     for b_idx in range(b):
    #         for c_idx in [1, index]: # for 2 indexes, which are 1 and component index
    #             comp_mask, rgb_mask = F.interpolate(comp_ref['O_mask'],(size,size))[b_idx,c_idx], color_ref['O_mask'][b_idx,c_idx]
    #             # comp_mask, rgb_mask = F.interpolate(comp_ref['O_mask'],(size,size))[b_idx,c_idx], F.interpolate(color_ref['O_mask'],(size,size))[b_idx,c_idx]
    #             if comp_mask.sum() == 0 or comp_mask.sum() == 1 or rgb_mask.sum() == 0 or rgb_mask.sum() == 1:
    #                 continue

    #             comp_matrix = torch.masked_select(comp_ref['CPM_feature'][b_idx], comp_mask.bool()).reshape(c, -1) # 64, pixel_num_A
    #             comp_matrix_bar = comp_matrix - comp_matrix.mean(1, keepdim=True) # (64, 1)
    #             comp_matrix_norm = torch.norm(comp_matrix_bar, dim=0, keepdim=True)
    #             comp_matrix_ = comp_matrix_bar / comp_matrix_norm

    #             rgb_matrix = torch.masked_select(color_ref['C_feature'][b_idx], rgb_mask.bool()).reshape(c, -1) # 64, pixel_num_B
    #             rgb_matrix_bar = rgb_matrix - rgb_matrix.mean(1, keepdim=True) # 64, pixel_num_B
    #             rgb_matrix_norm = torch.norm(rgb_matrix_bar, dim=0, keepdim=True)
    #             rgb_matrix_ = rgb_matrix_bar / rgb_matrix_norm
               
    #             correlation_matrix = torch.matmul(comp_matrix_.transpose(0,1), rgb_matrix_)
    #             if torch.isnan(correlation_matrix).sum():
    #                 import pdb; pdb.set_trace()
    #             correlation_matrix = F.softmax(correlation_matrix,dim=1)
                
    #             rgb_pixels = torch.masked_select(color_ref['C_img'][b_idx], rgb_mask.bool()).reshape(3,-1)
    #             colorized_matrix = torch.matmul(correlation_matrix, rgb_pixels.transpose(0,1))

    #             canvas[b_idx].masked_scatter_(comp_mask.bool(), colorized_matrix) # 3 128 128

    #     return canvas

    # def forward(self, gray_feature, rgb_feature, rgb_image, gray_label, rgb_label, img_size):
    #     gray_feature = F.interpolate(gray_feature, (img_size,img_size))
    #     rgb_feature = F.interpolate(rgb_feature, (img_size,img_size))
    #     rgb_image = F.interpolate(rgb_image, (img_size,img_size))
    #     gray_label= F.interpolate(gray_label, (img_size,img_size))
    #     rgb_label = F.interpolate(rgb_label, (img_size,img_size))
    #     color_reference = self.do_RC(gray_feature, rgb_feature, rgb_image, gray_label, rgb_label)

    #     return color_reference 

    # def do_RC(self, color_ref, comp_ref):
    #     canvas = torch.ones_like(color_ref['C_feature']) * -1
    #     b, c, h, w = comp_ref['C_feature'].size()
        
    #     for b_idx in range(b):
    #         for c_idx in range(1, 12):
    #             gray_mask, rgb_mask = comp_ref['O_mask'][b_idx,c_idx], color_ref['O_mask'][b_idx,c_idx]
    #             if gray_mask.sum() == 0 or gray_mask.sum() == 1 or rgb_mask.sum() == 0 or rgb_mask.sum() == 1:
    #                 continue
    #             gray_matrix = torch.masked_select(comp_ref['C_feature'][b_idx], gray_mask.bool()).reshape(c, -1) # 64, pixel_num_A
    #             gray_matrix_bar = gray_matrix - gray_matrix.mean(1, keepdim=True) # (64, 1)
    #             gray_matrix_norm = torch.norm(gray_matrix_bar, dim=0, keepdim=True)
    #             gray_matrix_ = gray_matrix_bar / gray_matrix_norm

    #             rgb_matrix = torch.masked_select(color_ref['C_feature'][b_idx], rgb_mask.bool()).reshape(c, -1) # 64, pixel_num_B
    #             rgb_matrix_bar = rgb_matrix - rgb_matrix.mean(1, keepdim=True) # 64, pixel_num_B
    #             rgb_matrix_norm = torch.norm(rgb_matrix_bar, dim=0, keepdim=True)
    #             rgb_matrix_ = rgb_matrix_bar / rgb_matrix_norm
               
    #             correlation_matrix = torch.matmul(gray_matrix_.transpose(0,1), rgb_matrix_)
    #             if torch.isnan(correlation_matrix).sum():
    #                 import pdb; pdb.set_trace()
    #             correlation_matrix = F.softmax(correlation_matrix,dim=1)
                
    #             rgb_pixels = torch.masked_select(color_ref['C_feature'][b_idx], rgb_mask.bool()).reshape(3,-1)
    #             colorized_matrix = torch.matmul(correlation_matrix, rgb_pixels.transpose(0,1)).transpose(0,1)

    #             canvas[b_idx].masked_scatter_(gray_mask.bool(), colorized_matrix) # 3 128 128

    #     return canvas