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


class EmotionalStyleTransfer():
    def __init__(self):
        # model runner
        self.model = mcycle.CycleGANModelLoader()
        self.model.initialize(get_opts())
        # dataset TODO: methodize
        dataset = datausd.UnalignedSingleDataset()
        dataset.initialize(opts)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batchSize,
            shuffle=not opts.serial_batches,
            num_workers=int(opts.nThreads))
    
    def set_image(self, image):
        # util.im2tensor 的な何か dataloader から抜いてくる
        self.model.set_input(image)  # TODO: tensor of image

    def run(self, AtoB):
        self.model.forward(AtoB)
        return util.tensor2im(self.model.get_fake_img())


import tkinter as tk
import tkinter.filedialog as tkfd
import PIL.ImageTk, PIL.Image


class CycleGANDemoApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        self.create_widgets(master)
        
    def create_widgets(self, master):
        filename = '/Users/a12201/data/bam/bam_h2s_cyclegan2/testA/355281.jpg'
        # Image
        pil_image = PIL.Image.open(filename).convert('RGB')
        self.photo_image = PIL.ImageTk.PhotoImage(pil_image)
        self.img_label = tk.Label(self, image=self.photo_image, height=500)
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
        self.photo_image = PIL.ImageTk.PhotoImage(pil_image)
        self.img_label.configure(image=self.photo_image)

    def print_entry(self, ev):
        print(ev)
        print(self.contents.get())

    def display_img(self, ev):
        print(ev)
        filename = self.contents.get()
        pil_image = PIL.Image.open(filename).convert('RGB')
        self.photo_image = PIL.ImageTk.PhotoImage(pil_image)
        self.img_label.configure(image=self.photo_image)
        # self.canvas = tk.Canvas(self, width=500, height=500)
        # self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
        # self.canvas.pack(side='top')

    def run_AtoB(self):
        print('AtoB')
        # TODO: get image from filehandler

    def run_BtoA(self):
        print('BtoA')
        # TODO: get image from filehandler


def tkinter_test():
    root = tk.Tk()
    root.title('Emotional Style Stransfer Demo App')
    # root.size(1000, 800)
    root.geometry('1000x600')
    app = CycleGANDemoApp(master=root)
    app.mainloop()


if __name__ == '__main__':
    tkinter_test()    
    # main()
