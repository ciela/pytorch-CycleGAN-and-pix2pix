import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedSingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)

        # dir 配下の画像ファイルをすべて探して paths に格納
        self.paths = make_dataset(self.dir)

        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        img = Image.open(path).convert('RGB')

        img = self.transform(img)
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = tmp.unsqueeze(0)

        return {'img': img, 'paths': path}

    def __len__(self):
        return self.size

    def name(self):
        return 'UnalignedSingleDataset'
