import torch
import torch.nn.functional as F
from tqdm import tqdm
from lib import utils
from lib.model_interface import ModelInterface
from MyModel.loss import MyModelLoss
from MyModel.nets import Generator
from packages import ProjectedDiscriminator

class MyNetModel(ModelInterface):
    def set_networks(self):
        self.G = Generator().cuda(self.gpu).train()
        
        self.D = ProjectedDiscriminator().cuda(self.gpu).train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)
        
    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.args)

    def go_step(self, global_step):
        # load batch
        color_img, color_gray, color_mask, skin_img, skin_gray, skin_mask, l_brow_img, l_brow_gray, l_brow_mask, \
                r_brow_img, r_brow_gray, r_brow_mask, l_eye_img, l_eye_gray, l_eye_mask, r_eye_img, r_eye_gray, r_eye_mask, \
                    nose_img, nose_gray, nose_mask, mouth_img, mouth_gray, mouth_mask = self.load_next_batch()

        self.dict["G_img"] = color_gray
        self.dict["C_img"] = color_img
        self.dict["O_mask"] = color_mask

        # color_ref, skin_ref, Lbrow_ref, Rbrow_ref, Leye_ref, Reye_ref, nose_ref, mouth_ref
        self.dict["C_imgs"] = [color_img, skin_img, l_brow_img, r_brow_img, l_eye_img, r_eye_img, nose_img, mouth_img]
        self.dict["G_imgs"] = [color_gray, skin_gray, l_brow_gray, r_brow_gray, l_eye_gray, r_eye_gray, nose_gray, mouth_gray]
        self.dict["O_masks"] = [color_mask, skin_mask, l_brow_mask, r_brow_mask, l_eye_mask, r_eye_mask, nose_mask, mouth_mask]
        
        # run G
        self.run_G(self.dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.args, self.G, self.opt_G, loss_G)
        # run D
        self.run_D(self.dict)

        # update G
        loss_D = self.loss_collector.get_loss_D(self.dict)
        utils.update_net(self.args, self.D, self.opt_D, loss_D)

        # for visualisation
        input_grid1 = self.gen_vis_grid(color_img, skin_img, l_brow_img, r_brow_img, 512)
        input_grid2 = self.gen_vis_grid(l_eye_img, r_eye_img, nose_img, mouth_img, 512)
        intermeidate_grid1 = self.gen_vis_grid(torch.ones_like(self.dict['base_dict']['skin_ref']['C_ref'])*-1, self.dict['base_dict']['skin_ref']['C_ref'], self.dict['comp_dict']['Lbrow_ref']['CPM_ref'], self.dict['comp_dict']['Rbrow_ref']['CPM_ref'], 512)
        intermeidate_grid2 = self.gen_vis_grid(self.dict['comp_dict']['Leye_ref']['CPM_ref'], self.dict['comp_dict']['Reye_ref']['CPM_ref'], self.dict['comp_dict']['nose_ref']['CPM_ref'], self.dict['comp_dict']['mouth_ref']['CPM_ref'], 512)
        
        # for visualisation mask
        vis_mask = torch.ones_like(self.dict['base_dict']['fake']['B_mask_union']) * -1
        vis_mask += F.interpolate(self.dict['base_dict']['skin_ref']['O_mask'][:,1].unsqueeze(1)*.5,(256,256)) + torch.sum(self.dict['base_dict']['fake']['O_mask'][:,2:10], dim=1, keepdim=True).clamp(0,1)
        
        self.train_images = [
            input_grid1,
            input_grid2,
            intermeidate_grid1,
            intermeidate_grid2,
            F.interpolate(vis_mask, (512,512)),
            self.dict["fake_img"],
            ]
        
    def gen_vis_grid(self, a,b,c,d,size):
        # for visualisation
        a_intermediate_image = F.interpolate(a, (size//2,size//2))
        b_intermediate_image = F.interpolate(b, (size//2,size//2))
        c_intermediate_image = F.interpolate(c, (size//2,size//2))
        d_intermediate_image = F.interpolate(d, (size//2,size//2))
        intermeidate_grid_1 = torch.cat((a_intermediate_image, b_intermediate_image), dim=-1)
        intermeidate_grid_2 = torch.cat((c_intermediate_image,d_intermediate_image), dim=-1)
        intermeidate_grid = torch.cat((intermeidate_grid_1, intermeidate_grid_2), dim=-2)
        
        return intermeidate_grid

    def run_G(self, dict):

        fake_img, base_dict, comp_dict = self.G(dict["C_imgs"], dict["G_imgs"], dict["O_masks"])
                
        dict["fake_img"] = fake_img
        dict["base_dict"] = base_dict
        dict["comp_dict"] = comp_dict
        
        dict["fake_B_mask_union"] = F.interpolate(base_dict['fake']['B_mask_union'], (512,512)).clamp(0,1)
        dict['fake_O_mask'] = F.interpolate(base_dict['fake']['O_mask'],(512,512)).clamp(0,1)

        # g_pred_fake, feat_fake = self.D(dict["fake_img"], None)
        # feat_real = self.D.get_feature(dict["C_img"])
        g_pred_fake, _ = self.D(dict["fake_img"], None)
        feat_fake = self.D.get_feature(dict["fake_img"]*(1-dict["fake_B_mask_union"]))
        feat_real = self.D.get_feature(dict["C_img"]*(1-dict["fake_B_mask_union"]))
        
        # feat_fake = self.D.get_feature(dict["fake_img"])
        # feat_real = self.D.get_feature(dict["C_img"])
        
        dict["g_pred_fake"] = g_pred_fake
        dict["feat_fake"] = feat_fake
        dict["feat_real"] = feat_real

    def run_D(self, dict):
        d_pred_fake, _  = self.D(dict['fake_img'].detach(), None)
        d_pred_real, _  = self.D(dict['C_img'], None)

        dict["d_pred_fake"] = d_pred_fake
        dict["d_pred_real"] = d_pred_real

    def do_validation(self):
        self.loss_collector.loss_dict["valid_L_G"] = .0
        self.loss_collector.loss_dict["valid_L_D"] = .0
        self.G.eval()
        self.D.eval()
        
        grid1, grid2, grid3, grid4, grid5, grid6 = [], [], [], [], [], []
        pbar = tqdm(self.valid_dataloader, desc='Run validate...')
        for color_img, color_gray, color_mask, skin_img, skin_gray, skin_mask, l_brow_img, l_brow_gray, l_brow_mask, \
                r_brow_img, r_brow_gray, r_brow_mask, l_eye_img, l_eye_gray, l_eye_mask, r_eye_img, r_eye_gray, r_eye_mask, \
                    nose_img, nose_gray, nose_mask, mouth_img, mouth_gray, mouth_mask in pbar:
                        
            color_img, color_gray, color_mask, skin_img, skin_gray, skin_mask, l_brow_img, l_brow_gray, l_brow_mask, \
                r_brow_img, r_brow_gray, r_brow_mask, l_eye_img, l_eye_gray, l_eye_mask, r_eye_img, r_eye_gray, r_eye_mask, \
                    nose_img, nose_gray, nose_mask, mouth_img, mouth_gray, mouth_mask = \
            color_img.to(self.gpu), color_gray.to(self.gpu), color_mask.to(self.gpu), skin_img.to(self.gpu), skin_gray.to(self.gpu), skin_mask.to(self.gpu), l_brow_img.to(self.gpu), l_brow_gray.to(self.gpu), l_brow_mask.to(self.gpu), \
                r_brow_img.to(self.gpu), r_brow_gray.to(self.gpu), r_brow_mask.to(self.gpu), l_eye_img.to(self.gpu), l_eye_gray.to(self.gpu), l_eye_mask.to(self.gpu), r_eye_img.to(self.gpu), r_eye_gray.to(self.gpu), r_eye_mask.to(self.gpu), \
                    nose_img.to(self.gpu), nose_gray.to(self.gpu), nose_mask.to(self.gpu), mouth_img.to(self.gpu), mouth_gray.to(self.gpu), mouth_mask.to(self.gpu)
                    
            self.valid_dict["G_img"] = color_gray
            self.valid_dict["C_img"] = color_img
            self.valid_dict["O_mask"] = color_mask

            # color_ref, skin_ref, Lbrow_ref, Rbrow_ref, Leye_ref, Reye_ref, nose_ref, mouth_ref
            self.valid_dict["C_imgs"] = [color_img, skin_img, l_brow_img, r_brow_img, l_eye_img, r_eye_img, nose_img, mouth_img]
            self.valid_dict["G_imgs"] = [color_gray, skin_gray, l_brow_gray, r_brow_gray, l_eye_gray, r_eye_gray, nose_gray, mouth_gray]
            self.valid_dict["O_masks"] = [color_mask, skin_mask, l_brow_mask, r_brow_mask, l_eye_mask, r_eye_mask, nose_mask, mouth_mask]
            

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.loss_collector.loss_dict["valid_L_G"] += self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                     
                self.run_D(self.valid_dict)
                self.loss_collector.loss_dict["valid_L_D"] += self.loss_collector.get_loss_D(self.valid_dict, valid=True)

                # for visualisation
                input_grid1 = self.gen_vis_grid(color_img, skin_img, l_brow_img, r_brow_img, 512)
                input_grid2 = self.gen_vis_grid(l_eye_img, r_eye_img, nose_img, mouth_img, 512)
                intermeidate_grid1 = self.gen_vis_grid(torch.ones_like(self.valid_dict['base_dict']['skin_ref']['C_ref'])*-1, self.valid_dict['base_dict']['skin_ref']['C_ref'], self.valid_dict['comp_dict']['Lbrow_ref']['CPM_ref'], self.valid_dict['comp_dict']['Rbrow_ref']['CPM_ref'], 512)
                intermeidate_grid2 = self.gen_vis_grid(self.valid_dict['comp_dict']['Leye_ref']['CPM_ref'], self.valid_dict['comp_dict']['Reye_ref']['CPM_ref'], self.valid_dict['comp_dict']['nose_ref']['CPM_ref'], self.valid_dict['comp_dict']['mouth_ref']['CPM_ref'], 512)
        
                # for visualisation mask
                vis_mask = torch.ones_like(self.valid_dict['base_dict']['fake']['B_mask_union']) * -1
                vis_mask += F.interpolate(self.valid_dict['base_dict']['skin_ref']['O_mask'][:,1].unsqueeze(1)*.5,(256,256)) + torch.sum(self.valid_dict['base_dict']['fake']['O_mask'][:,2:10], dim=1, keepdim=True).clamp(0,1)
        
            if len(grid1) < 8: grid1.append(input_grid1)
            if len(grid2) < 8: grid2.append(input_grid2)
            if len(grid3) < 8: grid3.append(intermeidate_grid1)
            if len(grid4) < 8: grid4.append(intermeidate_grid2)
            if len(grid5) < 8: grid5.append(F.interpolate(vis_mask, (512,512)))
            if len(grid6) < 8: grid6.append(self.valid_dict["fake_img"])
            
        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)

        self.loss_collector.val_print_loss()

        self.G.train()
        self.G.c_net.eval()
        self.D.train()


        # save last validated images
        self.valid_images = [
            torch.cat(grid1, dim=0), 
            torch.cat(grid2, dim=0), 
            torch.cat(grid3, dim=0),
            torch.cat(grid4, dim=0), 
            torch.cat(grid5, dim=0), 
            torch.cat(grid6, dim=0), 
        ]

    @property
    def loss_collector(self):
        return self._loss_collector
        