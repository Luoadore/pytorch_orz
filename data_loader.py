import torch.utils.data as data
from PIL import Image
import torch
import numpy as np


class mura_data(data.Dataset):
    def __init__(self, lists, img_root='/root/workspace/data/inbreast/crop/', train=True, transform=None):
        self.img_root = img_root
        self.train = train
        with open(lists) as f:
            self.ids = [x.strip().split() for x in f.readlines()]
        self.transform = transform

    def __getitem__(self, index):
        imgpath = self.img_root+self.ids[index][0]+'.png'
        label = int(self.ids[index][1])
        label = torch.LongTensor([label])
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.ids)