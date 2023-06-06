import torch
import numpy as np
import torch.nn.functional as F
from loadData import getTestDataloder
import math


def make_test(net,test_dir = 'images_background_2/test'):
    test_dataloader = getTestDataloder()
    dataiter = iter(test_dataloader)

    THRESHOLD = 1.15
    correct_pre = 0
    for i in range(100):
        x0, x1, label2, label0, label1 = next(dataiter)
        output1, output2 = net(x0.type(torch.FloatTensor), x1.type(torch.FloatTensor))
        euclidean_distance = F.pairwise_distance(output1, output2)
        if label2 == 0 and euclidean_distance.cpu()[0].detach().numpy() < THRESHOLD:
            correct_pre += 1
        if label2 == 1 and euclidean_distance.cpu()[0].detach().numpy() >= THRESHOLD:
            correct_pre += 1
        accuracy = correct_pre / (i + 1)
        #print(test_image_distance(x0,x1,net))
    return accuracy


# 图片相似度
def test_image_distance(img_1, img_2, net):
    img_1 = img_1.type(torch.FloatTensor)
    img_2 = img_2.type(torch.FloatTensor)
    output1, output2 = net(img_1, img_2)
    euclidean_distance = F.pairwise_distance(output1, output2).item()
    def normal_distribution(x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)
    similarity = normal_distribution(euclidean_distance, 0, 1) / normal_distribution(0, 0, 1)
    return similarity