import os

import torch

from learn.demo2.config import save_dir, input_shape, getModel
from learn.demo2.myutils import getAccuracy, showImgArray
from loss import ContrastiveLoss
import torch.optim as optim
from loadData import getDataloder, getTestDataloder
from tqdm import tqdm

from utils.callbacks import LossHistory
from utils.utils import get_lr_scheduler, set_optimizer_lr

train_number_epochs = 20
train_batch_size = 64
save_period = 10
dataloader = getDataloder(batch_size=train_batch_size)
testdataloder = getTestDataloder(batch_size=train_batch_size)

net = getModel()
loss_func = ContrastiveLoss()

# ---------------------------------------#
#   判断每一个世代的长度
# ---------------------------------------#
num_train = len(dataloader.dataset)
num_val = len(testdataloder.dataset)
epoch_step = num_train // train_batch_size
epoch_step_val = num_val // train_batch_size

# ---------------------------------------#
#   获得学习率下降的公式
# ---------------------------------------#
optimizer_type = "adam"
lr_decay_type = 'cos'
nbs = 64
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(train_batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(train_batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
momentum = 0.9
weight_decay = 5e-4
optimizer = {
    'adam': optim.Adam(net.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': optim.SGD(net.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
}[optimizer_type]
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, train_number_epochs)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    loss_history = LossHistory(save_dir, net, input_shape=input_shape)
    for epoch in range(train_number_epochs):

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        total = 0
        total_loss = 0
        total_accuracy = 0

        val_total = 0
        val_loss = 0
        val_total_accuracy = 0

        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{train_number_epochs}', postfix=dict, mininterval=0.3)

        for i, data in enumerate(dataloader):
            # img0,img1  (batch, 3, 32, 32)
            # label (batch)
            img_1, img_2, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(
                torch.FloatTensor)
            showImgArray(img_1, img_2, label)
            optimizer.zero_grad()
            output_1, output_2 = net(img_1, img_2)
            loss = loss_func(output_1, output_2, label)
            loss.backward()
            optimizer.step()

            total += label.size()[0]
            accuracy = getAccuracy(output_1, output_2, label)
            total_loss += loss.item()
            total_accuracy += accuracy

            pbar.set_postfix(**{'total_loss': total_loss / (i + 1),
                                'acc': total_accuracy / total,
                                })
            pbar.update(1)

        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{train_number_epochs}', postfix=dict,
                    mininterval=0.3)

        for i, data in enumerate(testdataloder):
            img_1, img_2, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(
                torch.FloatTensor)
            optimizer.zero_grad()
            output_1, output_2 = net(img_1, img_2)
            loss = loss_func(output_1, output_2, label)

            val_total += label.size()[0]
            accuracy = getAccuracy(output_1, output_2, label)
            val_loss += loss.item()
            val_total_accuracy += accuracy

            pbar.set_postfix(**{'val_loss': val_loss / (i + 1),
                                'acc': val_total_accuracy / val_total,
                                })
            pbar.update(1)

        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(train_number_epochs))
        print('Total acc: %.3f || Val acc: %.3f ' % (total_accuracy / total, val_total_accuracy / val_total))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == train_number_epochs:
            torch.save(net.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(net.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(net.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
