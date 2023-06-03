import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

train_transform = transforms.Compose([
    # transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
    transforms.Resize(28),
    transforms.ToTensor()
])
# test_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

train_data = torchvision.datasets.FashionMNIST(root='/data/FashionMNIST', train=True, download=True,
                                               transform=train_transform)
test_data = torchvision.datasets.FashionMNIST(root='/data/FashionMNIST', train=False, download=True,
                                              transform=train_transform)
# print(test_data[0])

batch_size = 256
num_workers = 0
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# # 如果是自己的数据集需要自己构建dataset
# class FashionMNISTDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.transform = transform
#         self.images = df.iloc[:, 1:].values.astype(np.uint8)
#         self.labels = df.iloc[:, 0].values
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image = self.images[idx].reshape(28, 28, 1)
#         label = int(self.labels[idx])
#         if self.transform is not None:
#             image = self.transform(image)
#         else:
#             image = torch.tensor(image / 255., dtype=torch.float)
#         label = torch.tensor(label, dtype=torch.long)
#         return image, label
#
#
# train = pd.read_csv("../FashionMNIST/fashion-mnist_train.csv")
# # train.head(10)
# test = pd.read_csv("../FashionMNIST/fashion-mnist_test.csv")
# # test.head(10)
# print(len(test))
# train_iter = FashionMNISTDataset(train, data_transform)
# print(train_iter)
# test_iter = FashionMNISTDataset(test, data_transform)


# print(train_iter)


def show_images(imgs, num_rows, num_cols, targets, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for ax, img, target in zip(axes, imgs, targets):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL
            ax.imshow(img)
        # 设置坐标轴不可见
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(hspace=0.35)
        ax.set_title('{}'.format(target))
    return axes


# 将dataloader转换成迭代器才可以使用next方法
# X, y = next(iter(train_iter))
# show_images(X.squeeze(), 3, 8, targets=y)
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128 * 8 * 8, 512)
        self.drop1 = nn.Dropout()
        self.fc6 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # print(" x shape ",x.size())
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model = model.to(device)
# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    train_loss = 0
    train_loss_list = []
    for data, label in train_iter:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_iter.dataset)
    train_loss_list.append(train_loss)
    print('Epoch:{}\tTraining Loss:{:.6f}'.format(epoch + 1, train_loss))


def val():
    model.eval()
    gt_labels = []
    pred_labels = []
    acc_list = []
    with torch.no_grad():
        for data, label in test_iter:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    acc_list.append(acc)
    print(gt_labels, pred_labels)
    print('Accuracy: {:6f}'.format(acc))


epochs = 2
for epoch in range(epochs):
    train(epoch)
    val()

torch.save(model, "mymodel.pth")

# 写成csv
model = torch.load("mymodel.pth")
model = model.to(device)
id = 0
preds_list = []
with torch.no_grad():
    for x, y in test_iter:
        batch_pred = list(model(x.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1
# print(preds_list)

with open('result_ljh.csv', 'w') as f:
    f.write('Id,Category\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
