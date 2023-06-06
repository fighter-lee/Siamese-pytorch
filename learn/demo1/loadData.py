import os
import random
from torch.utils.data import Dataset,DataLoader
import torchvision.datasets
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils import load_dataset
input_shape     = [96, 96]

class getDataset(Dataset):
    def __init__(self,getDataset,transform=None,relables=False):
        self.getDataset = getDataset
        self.relables = relables
        self.transform = transform
    def __getitem__(self, index):
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
        if self.transform is not None:
             img0 = self.transform(img0)
             # print(img0.shape)
             img1 = self.transform(img1)

        if self.relables:
            return img0, img1, torch.from_numpy(np.array([int(labels[rand_i] != labels[rand_j])], dtype=np.float32)), \
                   labels[rand_i], labels[rand_j]
        else:
            return img0, img1, torch.from_numpy(np.array([int(labels[rand_i] != labels[rand_j])], dtype=np.float32))

    def __len__(self):
        return len(self.getDataset.imgs)

transform_train = transforms.Compose([transforms.Resize(size=(32, 32)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
def getDataloder(data_dir = "./FlickerLogo32/train",batch_size = 64):
    train_dataset = torchvision.datasets.ImageFolder(root = data_dir)
    dataset = getDataset(train_dataset,transform=transform_train)
    train_dataloader = DataLoader(dataset,shuffle=True,batch_size = batch_size)
    return train_dataloader
def getTestDataloder(data_dir = "./FlickerLogo32/test",batch_size = 1):
    test_dataset = torchvision.datasets.ImageFolder(root = data_dir)
    dataset = getDataset(test_dataset,relables = True,transform=transform_train)
    test_dataloader = DataLoader(dataset,shuffle=True,batch_size = batch_size)
    dataiter = iter(test_dataloader)
    return test_dataloader

def getDataloderNew(dataset_path = "D:\workspace\python\Siamese-pytorch-master\datasets",batch_size = 32):
    train_lines, train_labels, val_lines, val_labels = load_dataset(dataset_path, True, 0.9)
    train_dataset = SiameseDataset(input_shape, train_lines, train_labels, True)
    val_dataset = SiameseDataset(input_shape, val_lines, val_labels, False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
               drop_last=True, collate_fn=dataset_collate, sampler=None)
    test_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=None)
    return train_dataloader, test_dataloader