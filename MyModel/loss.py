from lib.loss_interface import Loss, LossInterface
import time
import torch.nn.functional as F
from MyModel.utils.util_loss import VGGLoss

class MyModelLoss(LossInterface):
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}

        self.vggloss = VGGLoss(self.args.gpu).cuda(self.args.gpu)
    
    def get_loss_G(self, dict, valid=False):
        L_G = 0.0
        
        if self.args.W_adv:
            # L_gan = self.ganloss(dict["g_pred_fake"],True,for_discriminator=False)
            L_adv = (-dict["g_pred_fake"]).mean()
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_g_adv"] = round(L_adv.item(), 4)

        if self.args.W_vgg:
            L_vgg = self.vggloss(dict["fake_img"], dict["color_image"])
            L_G += self.args.W_vgg * L_vgg
            self.loss_dict["L_vgg"] = round(L_vgg.item(), 4)

        # feat loss
        if self.args.W_feat:
            L_feat = Loss.get_L1_loss(dict["feat_fake"]["3"], dict["feat_real"]["3"])
            L_G += self.args.W_feat * L_feat
            self.loss_dict["L_feat"] = round(L_feat.item(), 4)
        
        
        
        if valid:
            self.loss_dict["valid_L_G"] += round(L_G.item(), 4)
        else:
            self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

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
        