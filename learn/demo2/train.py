import os

import torch
from loss import ContrastiveLoss
from net import SiameseNetwork
import torch.optim as optim
from loadData import getDataloder,getTestDataloder
from test import make_test
train_number_epochs = 50
train_batch_size = 64
dataloader = getDataloder()
testdataloder = getTestDataloder()

net = SiameseNetwork()
loss_func = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    accuracy = .0
    for epoch in range(train_number_epochs):
        for i, data in enumerate(dataloader):
            # img0,img1  (batch, 3, 32, 32)
            # label (batch)
            img_1, img_2, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor), data[2].type(torch.FloatTensor)
            optimizer.zero_grad()
            output_1, output_2 = net(img_1, img_2)
            loss = loss_func(output_1, output_2, label)
            loss.backward()
            optimizer.step()
            if i%1==0:
                total = len(dataloader)
                print("\rEpoch: %d, cur_epoch_progress: %d/%d, loss: %f, accuracy: %f" % (epoch, i, total, loss.item(), accuracy), end="")
                if i%20==0:
                    accuracy = make_test(net)

    torch.save(net.state_dict(), 'model.pth')

