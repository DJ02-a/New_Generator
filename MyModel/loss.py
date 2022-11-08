from lib.loss_interface import Loss, LossInterface
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from MyModel.utils.util_loss import VGGLoss

class MyModelLoss(LossInterface):
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}

        self.vggloss = VGGLoss(self.args.gpu).cuda(self.args.gpu)
    
    def get_loss_G(self, dict, valid=False):
        L_G = 0.0
        
        # origin_skin_mask = (1 - torch.sum(dict["gray_one_hot"][:,2:10],dim=1,keepdim=True))
        # fake_skin_mask = F.interpolate((1 - dict["move_new_mask"][:,0].unsqueeze(1)),(512,512))
        if self.args.W_adv: # transposed_mask
            L_adv = (-dict["g_pred_fake"]).mean()
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_g_adv"] = round(L_adv.item(), 4)

        if self.args.W_vgg:
            L_vgg = self.vggloss(dict["fake_img"], dict["C_img"])
            # L_vgg = self.vggloss(dict["fake_img"]*(1-dict["fake_B_mask_union"]), dict["C_img"]*(1-dict["fake_B_mask_union"]))
            L_G += self.args.W_vgg * L_vgg
            self.loss_dict["L_vgg"] = round(L_vgg.item(), 4)

        # feat loss
        if self.args.W_feat:
            L_feat = Loss.get_L1_loss(dict["feat_fake"]["3"], dict["feat_real"]["3"])
            L_G += self.args.W_feat * L_feat
            self.loss_dict["L_feat"] = round(L_feat.item(), 4)
        
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(dict["fake_img"], dict["C_img"])
            # L_lpips = Loss.get_lpips_loss(dict["fake_img"]*(1-dict["fake_B_mask_union"]), dict["C_img"]*(1-dict["fake_B_mask_union"]))
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)
            
        if self.args.W_blend:
            L_blend = Loss.get_L1_loss(dict['base_dict']['skin_ref']['C_img_colored'], F.interpolate(dict["C_img"],(128,128)))
            L_G += self.args.W_blend * L_blend
            self.loss_dict["L_blend"] = round(L_blend.item(), 4)
            
        if self.args.W_component:
            L_component = self.get_component_loss(dict)
            L_G += self.args.W_component * L_component
            self.loss_dict["L_component"] = round(L_component.item(), 4)
            
        if valid:
            self.loss_dict["valid_L_G"] += round(L_G.item(), 4)
        else:
            self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_component_loss(self, dict):
        # def get_component_loss(self, fake_img, color_img, fake_mask, gray_mask):
        comp_loss_sum = .0
        count = 0
        crop_size = 128
        for b_idx in range(2):
            # for c_idx in [2,3,4,5,6,9]:
            for component in dict['comp_dict']:
                comp_dict = dict['comp_dict'][component]
                comp_index = comp_dict['index']
                # _gray_mask = gray_mask[b_idx][c_idx]
                    
                if dict["O_mask"][b_idx, comp_index].sum() and dict["fake_O_mask"][b_idx, comp_index].sum():
                    try:
                        count += 1
                        # print(count, "count")

                        real_comp = dict["C_img"][b_idx] * dict["O_mask"][b_idx, comp_index].unsqueeze(0) # 3 H W
                        fake_comp = dict["fake_img"][b_idx] * dict["fake_O_mask"][b_idx, comp_index].unsqueeze(0) # 3 H W
                        
                        # real_comp = F.interpolate(real_comp, (128, 128))
                        # fake_comp = F.interpolate(fake_comp, (128, 128))

                        center_x, center_y = comp_dict['center'][b_idx]
                        # center_x, center_y = comp_dict['center'][b_idx]
                        scale_x, scale_y = comp_dict['scale'][b_idx]
                        shift_x, shift_y = comp_dict['shift'][b_idx]
                        center_x, center_y, shift_x, shift_y = int(center_x), int(center_y), int(shift_x), int(shift_y)
                        half_x, half_y = int(crop_size//2 // scale_x), int(crop_size//2 // scale_y)

                        real_comp_crop     = transforms.functional.crop(F.pad(real_comp,(half_x,half_x,half_y,half_y)), top=half_y + (center_y-crop_size//2), left=half_x + (center_x-crop_size//2), height= crop_size, width= crop_size)
                        fake_comp_crop     = transforms.functional.crop(F.pad(fake_comp,(half_x,half_x,half_y,half_y)), top=half_y + (center_y+shift_y-half_y), left=half_x + (center_x+shift_x-half_x), height= 2*half_y, width= 2*half_x)

                        # real_comp_crop = real_comp[:, :, 4*(center_y-crop_size//2):4*(center_y+crop_size//2), 4*(center_x-crop_size//2):4*(center_x+crop_size//2)]
                        # fake_comp_crop = fake_comp[:, :, 4*(center_y+shift_y-half_y):4*(center_y+shift_y+half_y), 4*(center_x+shift_x-half_x):4*(center_x+shift_x+half_x)]
                        
                        real_comp_canvas = F.interpolate(real_comp_crop.unsqueeze(0), (256, 256))
                        fake_comp_canvas = F.interpolate(fake_comp_crop.unsqueeze(0), (256, 256))
                            
                        # comp_loss_sum += Loss.get_L1_loss(real_comp_crop, fake_comp_crop)
                        comp_loss_sum += Loss.get_L1_loss(real_comp_canvas, fake_comp_canvas)
                    except:
                        import pdb;pdb.set_trace()


        return comp_loss_sum/count
        

    def get_loss_D(self, dict, valid=False):
        L_D = 0.0
        L_real = (F.relu(1 - dict["d_pred_real"])).mean()
        L_fake = (F.relu(1 + dict["d_pred_fake"])).mean()
        L_D = L_real + L_fake
        
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        if valid:
            self.loss_dict["valid_L_D"] += round(L_D.item(), 4)
        else:
            self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
            self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
            self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        