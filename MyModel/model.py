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
        self.G = Generator(256).cuda(self.gpu).train()
        
        self.D = ProjectedDiscriminator().cuda(self.gpu).train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)
        
    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.args)

    def go_step(self, global_step):
        # load batch

        gray_image, color_image, color_flip_image, gray_one_hot, color_flip_one_hot = self.load_next_batch()

        self.dict["gray_image"] = gray_image
        self.dict["color_image"] = color_image
        self.dict["color_flip_image"] = color_flip_image
        self.dict["gray_one_hot"] = gray_one_hot
        self.dict["color_flip_one_hot"] = color_flip_one_hot
        
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

        # print images
        self.train_images = [
            self.dict["color_image"],
            self.dict["color_flip_image"],
            self.dict["gray_image"],
            # self.dict["transposed_gray"],
            self.dict["color_reference_image"],
            self.dict["fake_img"],
            self.dict["fake_gray"]
            ]
        

    def run_G(self, dict):
        fake_img, color_reference_image, transposed_gray, transposed_mask = self.G(dict["gray_image"], dict["color_image"], dict["color_flip_image"], dict["gray_one_hot"], dict["color_flip_one_hot"])
        # https://holypython.com/python-pil-tutorial/how-to-convert-an-image-to-black-white-in-python-pil/
        fake_gray = (fake_img[:,0] * 0.299 + fake_img[:,1] * 0.587 + fake_img[:,2]* 0.114).unsqueeze(1)
                
        dict["fake_img"] = fake_img
        dict["color_reference_image"] = color_reference_image
        dict["transposed_gray"] = transposed_gray
        dict["transposed_mask"] = transposed_mask
        dict["fake_gray"] = fake_gray
        
        g_pred_fake, feat_fake = self.D(dict["fake_img"], None)
        feat_real = self.D.get_feature(dict["color_image"])
        
        dict["g_pred_fake"] = g_pred_fake
        dict["feat_fake"] = feat_fake
        dict["feat_real"] = feat_real

    def run_D(self, dict):
        d_pred_fake, _  = self.D(dict['fake_img'].detach(), None)
        d_pred_real, _  = self.D(dict['color_image'], None)

        dict["d_pred_fake"] = d_pred_fake
        dict["d_pred_real"] = d_pred_real

    def do_validation(self):
        self.loss_collector.loss_dict["valid_L_G"] = .0
        self.loss_collector.loss_dict["valid_L_D"] = .0
        self.G.eval()
        self.D.eval()
        
        input_grid, result_grid = [], []
        pbar = tqdm(self.valid_dataloader, desc='Run validate...')
        for gray_image, color_image, color_flip_image, gray_one_hot, color_flip_one_hot in pbar:
            gray_image, color_image, color_flip_image, gray_one_hot, color_flip_one_hot = \
                gray_image.to(self.gpu), color_image.to(self.gpu), color_flip_image.to(self.gpu), gray_one_hot.to(self.gpu), color_flip_one_hot.to(self.gpu)


            self.valid_dict["gray_image"] = gray_image
            self.valid_dict["color_image"] = color_image
            self.valid_dict["color_flip_image"] = color_flip_image
            self.valid_dict["gray_one_hot"] = gray_one_hot
            self.valid_dict["color_flip_one_hot"] = color_flip_one_hot

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.loss_collector.loss_dict["valid_L_G"]\
                     += self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                     
                self.run_D(self.valid_dict)
                self.loss_collector.loss_dict["valid_L_D"]\
                     += self.loss_collector.get_loss_D(self.valid_dict, valid=True)

            if len(input_grid) < 8: input_grid.append(self.valid_dict["color_image"])
            if len(result_grid) < 8: result_grid.append(self.valid_dict["fake_img"])
            
        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)

        self.loss_collector.val_print_loss()

        self.G.train()
        self.D.train()


        # save last validated images
        self.valid_images = [
            torch.cat(input_grid, dim=0), 
            torch.cat(result_grid, dim=0), 
        ]

    @property
    def loss_collector(self):
        return self._loss_collector
        