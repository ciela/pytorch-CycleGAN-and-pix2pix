import argparse
import torch

import data.unaligned_single_dataset as datausd
import models.cycle_gan_model_loader as mcycle
import util.util as util


def get_opts():
    opts = argparse.ArgumentParser()
    opts.gpu_ids = []
    opts.isTrain = False
    opts.checkpoints_dir = 'checkpoints'
    opts.name = 'bam_h2s_cyclegan'
    opts.resize_or_crop = 'resize_and_crop'
    opts.input_nc = 3
    opts.output_nc = 3
    opts.ngf = 64
    opts.which_model_netG = 'resnet_9blocks'
    opts.norm = 'instance'
    opts.no_dropout = True
    opts.init_type = 'normal'
    opts.which_epoch = '200'
    opts.verbose = False
    opts.which_direction = 'AtoB'
    opts.dataset_mode = 'unaligned'
    opts.dataroot = '/Users/a12201/src/github.com/ciela/data/jsai2018'
    opts.loadSize = 286
    opts.fineSize = 256
    opts.batchSize = 1
    opts.serial_batches = True
    opts.nThreads = 1
    return opts


def main():
    opts = get_opts()

    model = mcycle.CycleGANModelLoader()
    model.initialize(opts)

    dataset = datausd.UnalignedSingleDataset()
    dataset.initialize(opts)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batchSize,
            shuffle=not opts.serial_batches,
            num_workers=int(opts.nThreads))

    for i, data in enumerate(dataloader):
        model.set_input(data)  # いったん AtoB
        model.forward(True)  # いったん AtoB
        img_path = model.get_image_paths()[0]
        fake_img = model.get_fake_img()
        im = util.tensor2im(fake_img)
        print('%04d: process image... %s' % (i, img_path))
        util.save_image(im, img_path + '_fake.jpg')


import tkinter as tk


class CycleGANDemoApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.create_widgets(master)
        
    def create_widgets(self, master):
        # happy -> scary button
        self.run_AtoB = tk.Button(self, text='Happy -> Scary', command=self.run_AtoB)
        self.run_AtoB.pack(side='top')
        # scary -> happy button
        self.run_BtoA = tk.Button(self, text='Scary -> Happy', command=self.run_BtoA)
        self.run_BtoA.pack(side='top')
        # dnd image
        # TODO
        # quit button
        self.quit = tk.Button(self, text="QUIT", command=master.destroy)
        self.quit.pack(side="bottom")

    def display_img(self):
        pass

    def run_AtoB(self):
        print('AtoB')

    def run_BtoA(self):
        print('BtoA')

def tkinter_test():
    root = tk.Tk()
    app = CycleGANDemoApp(master=root)
    app.mainloop()


if __name__ == '__main__':
    tkinter_test()    
    # main()
