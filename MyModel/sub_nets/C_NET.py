import torch
import torch.nn as nn
import torch.nn.functional as F

class C_Net(nn.Module):
    def __init__(self):
        super(C_Net, self).__init__()

    def forward(self, gray_feature, rgb_feature, rgb_image, gray_label, rgb_label):
        color_reference = self.do_RC(gray_feature, rgb_feature, rgb_image, gray_label, rgb_label)

        return color_reference 

    def do_RC(self, gray_feature_map, rgb_feature_map, rgb_image, gray_one_hot, rgb_one_hot):
        _, n_ch, _, _ = gray_one_hot.shape
        canvas = torch.ones_like(rgb_image) * -1
        b, c, h, w = gray_feature_map.size()
        for b_idx in range(b):
            for c_idx in range(1, 12):
                gray_mask, rgb_mask = gray_one_hot[b_idx,c_idx], rgb_one_hot[b_idx,c_idx]
                if gray_mask.sum() == 0 or gray_mask.sum() == 1 or rgb_mask.sum() == 0 or rgb_mask.sum() == 1:
                    continue
                gray_matrix = torch.masked_select(gray_feature_map[b_idx], gray_mask.bool()).reshape(c, -1) # 64, pixel_num_A
                gray_matrix_bar = gray_matrix - gray_matrix.mean(1, keepdim=True) # (64, 1)
                gray_matrix_norm = torch.norm(gray_matrix_bar, dim=0, keepdim=True)
                gray_matrix_ = gray_matrix_bar / gray_matrix_norm

                rgb_matrix = torch.masked_select(rgb_feature_map[b_idx], rgb_mask.bool()).reshape(c, -1) # 64, pixel_num_B
                rgb_matrix_bar = rgb_matrix - rgb_matrix.mean(1, keepdim=True) # 64, pixel_num_B
                rgb_matrix_norm = torch.norm(rgb_matrix_bar, dim=0, keepdim=True)
                rgb_matrix_ = rgb_matrix_bar / rgb_matrix_norm
               
                correlation_matrix = torch.matmul(gray_matrix_.transpose(0,1), rgb_matrix_)
                if torch.isnan(correlation_matrix).sum():
                    import pdb; pdb.set_trace()
                correlation_matrix = F.softmax(correlation_matrix,dim=1)
                
                rgb_pixels = torch.masked_select(rgb_image[b_idx], rgb_mask.bool()).reshape(3,-1)
                colorized_matrix = torch.matmul(correlation_matrix, rgb_pixels.transpose(0,1)).transpose(0,1)

                canvas[b_idx].masked_scatter_(gray_mask.bool(), colorized_matrix) # 3 128 128
        return canvas