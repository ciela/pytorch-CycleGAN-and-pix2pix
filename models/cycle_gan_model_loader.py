import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModelLoader(BaseModel):
    def name(self):
        return 'CycleGANModelLoader'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        self.image = input['img'].to(self.device)
        self.image_paths = input['paths']

    def forward(self, AtoB):
        with torch.no_grad():
            self.fake_img = self.netG_A(self.image) if AtoB else self.netG_B(self.image)
