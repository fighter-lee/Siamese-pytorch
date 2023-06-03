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

    val_loss            = 0
    val_total_accuracy  = 0
    
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

        # with torch.no_grad():
        #     accuracy = calculate_accuracy(output1, output2, label)

        total_loss      += loss_contrastive.item()
        # total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                # 'acc'       : total_accuracy / (iteration + 1),
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

            # equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targetsdddd)
            # accuracy    = torch.mean(equal.float())

        val_loss            += output.item()
        # val_total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                # 'acc'       : val_total_accuracy / (iteration + 1)
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

def calculate_accuracy(output1, output2, label, threshold=0.5):
    euclidean_distance = F.pairwise_distance(output1, output2)
    predictions = (euclidean_distance < threshold).float()  # 阈值判定
    correct = (predictions == label).float().sum()  # 计算匹配成功的样本对数量
    accuracy = correct / label.shape[0]  # 计算成功率百分比
    return accuracy  # 返回成功率百分比（转换为标量）