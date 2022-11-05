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
        G_img, C_img, O_mask = self.load_next_batch()

        self.dict["G_img"] = G_img
        self.dict["C_img"] = C_img
        self.dict["O_mask"] = O_mask

        # color_ref, skin_ref, Lbrow_ref, Rbrow_ref, Leye_ref, Reye_ref, nose_ref, mouth_ref

        G_img = F.interpolate(G_img, (128,128))
        C_img = F.interpolate(C_img, (128,128))
        O_mask = F.interpolate(O_mask, (128,128))

        self.dict["G_imgs"] = [G_img, G_img, G_img, G_img, G_img, G_img, G_img, G_img]
        self.dict["C_imgs"] = [C_img, C_img, C_img, C_img, C_img, C_img, C_img, C_img]
        self.dict["O_masks"] = [O_mask, O_mask, O_mask, O_mask, O_mask, O_mask, O_mask, O_mask]
        
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
        Lbrow_intermediate_image = F.interpolate(self.dict['comp_dict']['Lbrow_ref']['CPM_ref'], (256,256))
        Leye_intermediate_image = F.interpolate(self.dict['comp_dict']['Leye_ref']['CPM_ref'], (256,256))
        nose_intermediate_image = F.interpolate(self.dict['comp_dict']['nose_ref']['CPM_ref'], (256,256))
        mouth_intermediate_image = F.interpolate(self.dict['comp_dict']['mouth_ref']['CPM_ref'], (256,256))
        intermeidate_grid_1 = torch.cat((Lbrow_intermediate_image, Leye_intermediate_image), dim=-1)
        intermeidate_grid_2 = torch.cat((nose_intermediate_image,mouth_intermediate_image), dim=-1)
        intermeidate_grid = torch.cat((intermeidate_grid_1, intermeidate_grid_2), dim=-2)
        
        # for visualisation mask
        vis_mask = torch.ones_like(self.dict['base_dict']['fake']['B_mask_union']) * -1
        # vis_mask += (1-self.dict['base_dict']['fake']['B_mask_union'].clamp(0,1)) * self.dict['base_dict']['skin_ref']['O_mask'][:,1].unsqueeze(1)*.5
        # vis_mask += self.dict['base_dict']['fake']['B_mask_union'].clamp(0,1) * torch.sum(self.dict['base_dict']['fake']['O_mask'][:,2:10], dim=1, keepdim=True).clamp(0,1)
        vis_mask += self.dict['base_dict']['skin_ref']['O_mask'][:,1].unsqueeze(1)*.5 + torch.sum(self.dict['base_dict']['fake']['O_mask'][:,2:10], dim=1, keepdim=True).clamp(0,1)
        
        self.train_images = [
            self.dict["C_img"],
            self.dict["G_img"], 
            F.interpolate(self.dict['base_dict']['skin_ref']['C_ref'], (512,512)),
            F.interpolate(self.dict['base_dict']['skin_ref']['C_img_colored'], (512,512)),
            intermeidate_grid,
            F.interpolate(vis_mask, (512,512)),
            self.dict["fake_img"],
            ]
        

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
        # feat_fake = self.D.get_feature(dict["fake_img"]*(1-dict["fake_B_mask_union"]))
        # feat_real = self.D.get_feature(dict["C_img"]*(1-dict["fake_B_mask_union"]))
        
        feat_fake = self.D.get_feature(dict["fake_img"])
        feat_real = self.D.get_feature(dict["C_img"])
        
        dict["g_pred_fake"] = g_pred_fake
        dict["feat_fake"] = feat_fake
        dict["feat_real"] = feat_real

    def run_D(self, dict):
        d_pred_fake, _  = self.D(dict['fake_img'].detach(), None)
        d_pred_real, _  = self.D(dict['C_img'], None)
        # d_pred_fake, _  = self.D(dict['fake_img'].detach()*(1-dict["fake_B_mask_union"]).detach(), None)
        # d_pred_real, _  = self.D(dict['C_img']*(1-dict["fake_B_mask_union"]).detach(), None)

        dict["d_pred_fake"] = d_pred_fake
        dict["d_pred_real"] = d_pred_real

    def do_validation(self):
        self.loss_collector.loss_dict["valid_L_G"] = .0
        self.loss_collector.loss_dict["valid_L_D"] = .0
        self.G.eval()
        self.D.eval()
        
        input_grid, result_grid, intermeidate_grids, color_reference_grid, blend_color_reference_grid = [], [], [], [], []
        pbar = tqdm(self.valid_dataloader, desc='Run validate...')
        for G_img, C_img, O_mask in pbar:
            G_img, C_img, O_mask = G_img.to(self.gpu), C_img.to(self.gpu), O_mask.to(self.gpu)

            self.valid_dict["G_img"] = G_img
            self.valid_dict["C_img"] = C_img
            self.valid_dict["O_mask"] = O_mask

            G_img = F.interpolate(G_img, (128,128))
            C_img = F.interpolate(C_img, (128,128))
            O_mask = F.interpolate(O_mask, (128,128))

            self.valid_dict["G_imgs"] = [G_img, G_img, G_img, G_img, G_img, G_img, G_img, G_img]
            self.valid_dict["C_imgs"] = [C_img, C_img, C_img, C_img, C_img, C_img, C_img, C_img]
            self.valid_dict["O_masks"] = [O_mask, O_mask, O_mask, O_mask, O_mask, O_mask, O_mask, O_mask]

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.loss_collector.loss_dict["valid_L_G"] += self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                     
                self.run_D(self.valid_dict)
                self.loss_collector.loss_dict["valid_L_D"] += self.loss_collector.get_loss_D(self.valid_dict, valid=True)

                # for visualisation
                Lbrow_intermediate_image = F.interpolate(self.valid_dict['comp_dict']['Lbrow_ref']['CPM_ref'], (256,256))
                Leye_intermediate_image = F.interpolate(self.valid_dict['comp_dict']['Leye_ref']['CPM_ref'], (256,256))
                nose_intermediate_image = F.interpolate(self.valid_dict['comp_dict']['nose_ref']['CPM_ref'], (256,256))
                mouth_intermediate_image = F.interpolate(self.valid_dict['comp_dict']['mouth_ref']['CPM_ref'], (256,256))
                intermeidate_grid_1 = torch.cat((Lbrow_intermediate_image, Leye_intermediate_image), dim=-1)
                intermeidate_grid_2 = torch.cat((nose_intermediate_image,mouth_intermediate_image), dim=-1)
                intermeidate_grid = torch.cat((intermeidate_grid_1, intermeidate_grid_2), dim=-2)
# ,
            if len(input_grid) < 8: input_grid.append(self.valid_dict["C_img"])
            if len(intermeidate_grids) < 8: intermeidate_grids.append(intermeidate_grid)
            if len(result_grid) < 8: result_grid.append(self.valid_dict["fake_img"])
            if len(color_reference_grid) < 8: color_reference_grid.append(F.interpolate(self.valid_dict['base_dict']['skin_ref']['C_ref'], (512,512)))
            if len(blend_color_reference_grid) < 8: blend_color_reference_grid.append(F.interpolate(self.valid_dict['base_dict']['skin_ref']['C_img_colored'], (512,512)))
            
        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)

        self.loss_collector.val_print_loss()

        self.G.train()
        self.D.train()


        # save last validated images
        self.valid_images = [
            torch.cat(input_grid, dim=0), 
            torch.cat(color_reference_grid, dim=0), 
            torch.cat(blend_color_reference_grid, dim=0), 
            torch.cat(intermeidate_grids, dim=0), 
            torch.cat(result_grid, dim=0), 
        ]

    @property
    def loss_collector(self):
        return self._loss_collector
        