import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.version import cuda

from loss import ContrastiveLoss
from net import SiameseNetwork
import torch.optim as optim
from loadData import getDataloder,getTestDataloder, getDataloderNew
from test import make_test
train_number_epochs = 5
train_batch_size = 64
# dataloader = getDataloder()
# testdataloder = getTestDataloder()
dataloader, testdataloder = getDataloderNew()

net = SiameseNetwork()
loss_func = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)


accuracy = .0

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    def showImg(image_1, image_2, label):
        for i in range(10):
            image_1_1 = image_1[i].permute(1, 2, 0)
            plt.subplot(1, 2, 1)
            plt.imshow(image_1_1)

            plt.subplot(1, 2, 2)
            image_2_2 = image_2[i].permute(1, 2, 0)
            plt.imshow(image_2_2)
            plt.text(-12, -12, 'lab:%.3f' % label[i].item(), ha='center', va='bottom', fontsize=11)
            plt.show()
            time.sleep(2)

    for epoch in range(train_number_epochs):
        for i, (img, label) in enumerate(dataloader):
            # img0,img1  (batch, 3, 32, 32)
            # label (batch)
            img_1, img_2 = img
            # img_1, img_2, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(torch.FloatTensor)
            optimizer.zero_grad()
            output_1, output_2 = net(img_1, img_2)
            loss = loss_func(output_1, output_2, label)
            # showImg(img_1, img_2, label)
            loss.backward()
            optimizer.step()
            if i % 1 == 0:
                total = len(dataloader)
                print("\rEpoch: %d, cur_epoch_progress: %d/%d, loss: %f, accuracy: %f" % (
                epoch, i, total, loss.item(), accuracy), end="")
                if i % 20 == 0:
                    accuracy = make_test(net, testdataloder)

    # torch.save(net, 'model.pth')
    torch.save(net.state_dict(), 'model.pth')

