import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from .utils import get_lr


def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0
    total = 0
    val_loss            = 0
    val_total_accuracy  = 0
    val_total = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()

    for iteration,  (img, label) in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                img  = img.cuda(local_rank)
                label = label.cuda(local_rank)
        img0, img1 = img

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            output1, output2 = model_train(img0, img1)
            loss_contrastive  = loss(output1, output2, label)

            loss_contrastive.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                output1, output2 = model_train(img0, img1)
                loss_contrastive = loss(output1, output2, label)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_contrastive).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            distance = F.pairwise_distance(output1, output2)
            zero_labels = label == 0  # 找出labels中为0的位置
            predict_for_zero_labels = distance[zero_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
            less_than_1_15 = predict_for_zero_labels < 1.15  # 找出浮点类型数小于1.15的位置
            accuracy = less_than_1_15.sum().item()

            one_labels = label == 1  # 找出labels中为1的位置
            predict_for_one_labels = distance[one_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
            greater_than_1_15 = predict_for_one_labels >= 1.15  # 找出浮点类型数大于1.15的位置
            accuracy += greater_than_1_15.sum().item()  # 计算浮点类型数大于1.15的数量

        total_loss      += loss_contrastive.item()
        total_accuracy  += accuracy
        total += label.size()[0]
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc'       : total_accuracy / total,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()


    for iteration,  (img, label) in enumerate(genval):
        if iteration >= epoch_step_val:
            break

        with torch.no_grad():
            if cuda:
                img  = img.cuda(local_rank)
                label = label.cuda(local_rank)

            img0, img1 = img

            optimizer.zero_grad()
            output1, output2 = model_train(img0, img1)
            output  = loss(output1, output2, label)

            distance = F.pairwise_distance(output1, output2)
            zero_labels = label == 0  # 找出labels中为0的位置
            predict_for_zero_labels = distance[zero_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
            less_than_1_15 = predict_for_zero_labels < 1.15  # 找出浮点类型数小于1.15的位置
            accuracy = less_than_1_15.sum().item()

            one_labels = label == 1  # 找出labels中为1的位置
            predict_for_one_labels = distance[one_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
            greater_than_1_15 = predict_for_one_labels > 1.15  # 找出浮点类型数大于1.15的位置
            accuracy += greater_than_1_15.sum().item()  # 计算浮点类型数大于1.15的数量

        val_loss            += output.item()
        val_total_accuracy  += accuracy
        val_total += label.size()[0]
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / val_total
                                })
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def test_image_distance(img_1, img_2, net):
    img_1 = img_1.type(torch.FloatTensor)
    img_2 = img_2.type(torch.FloatTensor)
    output1, output2 = net(img_1, img_2)
    euclidean_distance = F.pairwise_distance(output1, output2).item()
    def normal_distribution(x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)
    similarity = normal_distribution(euclidean_distance, 0, 1) / normal_distribution(0, 0, 1)
    return similarity