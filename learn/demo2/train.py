import os

import torch

from learn.demo2.config import save_dir, input_shape
from learn.demo2.myutils import getAccuracy
from loss import ContrastiveLoss
from net import SiameseNetwork
import torch.optim as optim
from loadData import getDataloder,getTestDataloder
from test import make_test
from tqdm import tqdm

from utils.callbacks import LossHistory

train_number_epochs = 10
train_batch_size = 64
save_period = 10
dataloader = getDataloder(batch_size=train_batch_size)
testdataloder = getTestDataloder(batch_size=train_batch_size)

net = SiameseNetwork()
loss_func = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)

# ---------------------------------------#
#   判断每一个世代的长度
# ---------------------------------------#
num_train = len(dataloader.dataset)
num_val = len(testdataloder.dataset)
epoch_step = num_train // train_batch_size
epoch_step_val = num_val // train_batch_size



if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    loss_history = LossHistory(save_dir, net, input_shape=input_shape)
    for epoch in range(train_number_epochs):

        total =0
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
            img_1, img_2, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(torch.FloatTensor)
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
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{train_number_epochs}', postfix=dict, mininterval=0.3)

        for i, data in enumerate(testdataloder):
            img_1, img_2, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(torch.FloatTensor)
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


