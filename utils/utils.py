import math
import os
import random
from functools import partial
from random import shuffle

import numpy as np
from PIL import Image

from .utils_aug import center_crop, resize


def load_dataset(dataset_path, train_own_data, train_ratio):
    types       = 0
    train_path  = os.path.join(dataset_path, 'images_background')
    lines       = []
    labels      = []
    
    if train_own_data:
        #-------------------------------------------------------------#
        #   自己的数据集，遍历大循环
        #-------------------------------------------------------------#
        for character in os.listdir(train_path):
            #-------------------------------------------------------------#
            #   对每张图片进行遍历
            #-------------------------------------------------------------#
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                lines.append(os.path.join(character_path, image))
                labels.append(types)
            types += 1
    else:
        #-------------------------------------------------------------#
        #   Omniglot数据集，遍历大循环
        #-------------------------------------------------------------#
        for alphabet in os.listdir(train_path):
            alphabet_path = os.path.join(train_path, alphabet)
            #-------------------------------------------------------------#
            #   Omniglot数据集，遍历小循环
            #-------------------------------------------------------------#
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                #-------------------------------------------------------------#
                #   对每张图片进行遍历
                #-------------------------------------------------------------#
                for image in os.listdir(character_path):
                    lines.append(os.path.join(character_path, image))
                    labels.append(types)
                types += 1

    #-------------------------------------------------------------#
    #   将获得的所有图像进行打乱。
    #-------------------------------------------------------------#
    random.seed(1)
    shuffle_index = np.arange(len(lines), dtype=np.int32)
    shuffle(shuffle_index)
    random.seed(None)
    lines    = np.array(lines,dtype=np.object)
    labels   = np.array(labels)
    lines    = lines[shuffle_index]
    labels   = labels[shuffle_index]
    
    #-------------------------------------------------------------#
    #   将训练集和验证集进行划分
    #-------------------------------------------------------------#
    num_train           = int(len(lines)*train_ratio)

    val_lines      = lines[num_train:]
    val_labels     = labels[num_train:]

    train_lines    = lines[:num_train]
    train_labels   = labels[:num_train]
    return train_lines, train_labels, val_lines, val_labels

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x):
    x /= 255.0
    return x

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vit_b_16': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/vit-patch_16.pth',
        'swin_transformer_tiny': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_tiny_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_small': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_small_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_base': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_base_patch4_window7_224_imagenet1k.pth'
    }
    try:
        url = download_urls[backbone]

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        load_state_dict_from_url(url, model_dir)
    except:
        print("There is no pretrained model for " + backbone)
