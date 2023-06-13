import os
import random

from matplotlib import pyplot as plt
from torch import tensor
from torch.utils.data import Dataset,DataLoader
import torchvision.datasets
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

from learn.demo2.config import input_shape
from utils.utils import cvtColor
from utils.utils_aug import RandomResizedCrop, ImageNetPolicy, Resize, CenterCrop


class getDataset(Dataset):
    def __init__(self,getDataset,transform=None,relables=False,random=True):
        self.getDataset = getDataset
        self.relables = relables
        self.transform = transform
        self.random = random

        self.resize_crop = RandomResizedCrop(input_shape, scale=(0.6, 1.0))
        self.policy = ImageNetPolicy()

        self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
        self.center_crop = CenterCrop(input_shape)
    def __getitem__(self, index):
        # 随机找第i图，再随机抽取0和1，来判断此次是否找相同lab还是不同lab
        list = []
        for i in range(len(self.getDataset.imgs)):
            list.append(self.getDataset.imgs[i][0])
        datas = list
        listb = []
        for i in range(len(self.getDataset.imgs)):
            listb.append(self.getDataset.imgs[i][1])
        labels = listb

        rand_i = random.choice(range(len(datas)))
        img0 = datas[rand_i]

        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                rand_j = random.choice(range(len(datas)))
                img1 = datas[rand_j]
                if labels[rand_j]==labels[rand_i]:
                    break
        else:
            while True:
                rand_j = random.choice(range(len(datas)))
                img1 = datas[rand_j]

                if labels[rand_j]!=labels[rand_i]:
                    break
        img0 = Image.open(img0)
        img1 = Image.open(img1)
        img0 = cvtColor(img0)
        img1 = cvtColor(img1)
        img0 = self.AutoAugment(img0, random=self.random)
        img1 = self.AutoAugment(img1, random=self.random)
        if self.transform is not None:
             img0 = self.transform(img0)
             # print(img0.shape)
             img1 = self.transform(img1)

        # 标签不相等为1 ， 相等为0
        if self.relables:
            return img0, img1, torch.from_numpy(np.array([int(labels[rand_i] != labels[rand_j])], dtype=np.float32)), \
                   labels[rand_i], labels[rand_j]
        else:
            return img0, img1, torch.from_numpy(np.array([int(labels[rand_i] != labels[rand_j])], dtype=np.float32))

    def __len__(self):
        return len(self.getDataset.imgs)

    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        # ------------------------------------------#
        #   resize并且随即裁剪
        # ------------------------------------------#
        image = self.resize_crop(image)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        # flip = self.rand() < .5
        # if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   随机增强
        # ------------------------------------------#
        image = self.policy(image)
        return image

transform_train = transforms.Compose([transforms.Resize(size=(input_shape)),
                                              transforms.ToTensor()])
def getDataloder(data_dir = "./dataset/train",batch_size = 64):
    train_dataset = torchvision.datasets.ImageFolder(root = data_dir)
    dataset = getDataset(train_dataset,transform=transform_train, random=True)
    train_dataloader = DataLoader(dataset,shuffle=True,batch_size = batch_size)
    return train_dataloader
def getTestDataloder(data_dir = "./dataset/test",batch_size = 64):
    test_dataset = torchvision.datasets.ImageFolder(root = data_dir)
    dataset = getDataset(test_dataset,relables = True,transform=transform_train, random=False)
    test_dataloader = DataLoader(dataset,shuffle=True,batch_size = batch_size)
    # dataiter = iter(test_dataloader)
    return test_dataloader
