import argparse
import torch

import data.unaligned_single_dataset as datausd
import models.cycle_gan_model_loader as mcycle
import util.util as util
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvtf
from PIL import Image


def get_transform_wo_norm(opt):
    transform_list = []
    osize = [opt.loadSize, opt.loadSize]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(opt.fineSize))
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor(),]
    return transforms.Compose(transform_list)


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
    opts.dataroot = '/Users/a12201/data/jsai2018'
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
        print(data['img'].shape)
        model.set_input(data)  # いったん AtoB
        model.forward(True)  # いったん AtoB
        img_path = model.get_image_paths()[0]
        fake_img = model.get_fake_img()
        im = util.tensor2im(fake_img)
        print('%04d: process image... %s' % (i, img_path))
        util.save_image(im, img_path + '_fake.jpg')


class EmotionalStyleTransfer():
    def __init__(self):
        opts = get_opts()
        # model runner
        self.model = mcycle.CycleGANModelLoader()
        self.model.initialize(opts)
        self.transform = get_transform(opts)
        self.transform_wo_norm = get_transform_wo_norm(opts)  # for display
    
    def set_image(self, image, path):
        inputs = {'img': self.transform(image).unsqueeze(0), 'paths': path}
        self.model.set_input(inputs)
        return tvtf.to_pil_image(self.transform_wo_norm(image))

    def run(self, AtoB):
        self.model.forward(AtoB)
        return Image.fromarray(util.tensor2im(self.model.get_fake_img()))


import tkinter as tk
import tkinter.filedialog as tkfd
import PIL.ImageTk, PIL.Image


class CycleGANDemoApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.create_widgets(master)
        self.transfer = EmotionalStyleTransfer()
        
    def create_widgets(self, master):
        filename = '/Users/a12201/data/bam/bam_h2s_cyclegan2/testA/355281.jpg'
        # Image
        pil_image = PIL.Image.open(filename).convert('RGB')
        self.photo_image = PIL.ImageTk.PhotoImage(pil_image)
        self.img_label = tk.Label(self, image=self.photo_image, width=256, height=256)
        self.img_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        # happy -> scary button
        self.run_AtoB = tk.Button(self, text='Happy -> Scary', command=self.run_AtoB)
        self.run_AtoB.grid(row=1, column=0, padx=5, pady=5)
        # scary -> happy button
        self.run_BtoA = tk.Button(self, text='Scary -> Happy', command=self.run_BtoA)
        self.run_BtoA.grid(row=1, column=1, padx=5, pady=5)
        # file dialog
        self.open_file = tk.Button(self, text='Open File', command=self.select_image)
        self.open_file.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        # quit button
        self.quit = tk.Button(self, text="QUIT", command=master.destroy)
        self.quit.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def select_image(self):
        selected_file = tkfd.askopenfilename(
            initialdir = "/Users/a12201/data", title= "Select file", 
            filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
        print(selected_file)
        pil_image = PIL.Image.open(selected_file).convert('RGB')
        data = self.transfer.set_image(pil_image, selected_file)
        self.photo_image = PIL.ImageTk.PhotoImage(data)
        self.img_label.configure(image=self.photo_image)

    def run_AtoB(self):
        print('AtoB')
        pil_image = self.transfer.run(True)
        self.photo_image = PIL.ImageTk.PhotoImage(pil_image)
        self.img_label.configure(image=self.photo_image)

    def run_BtoA(self):
        print('BtoA')
        pil_image = self.transfer.run(False)
        self.photo_image = PIL.ImageTk.PhotoImage(pil_image)
        self.img_label.configure(image=self.photo_image)


def tkinter_test():
    root = tk.Tk()
    root.title('Emotional Style Stransfer Demo App')
    # root.size(1000, 800)
    root.geometry('1000x750')
    app = CycleGANDemoApp(master=root)
    app.mainloop()


if __name__ == '__main__':
    tkinter_test()    
    # main()
