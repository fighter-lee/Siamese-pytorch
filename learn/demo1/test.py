import torch
import numpy as np
import torch.nn.functional as F
import math


def make_test(net,testdataloder):

    THRESHOLD = 1.15
    correct_pre = 0
    for i, (img, label) in enumerate(testdataloder):

        x0, x1 = img
        output1, output2 = net(x0, x1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        for i in range(len(label)):
            labItem = label[i]
            euclidean_distance_item = euclidean_distance[i]
            if labItem == 0 and euclidean_distance_item.item() < THRESHOLD:
                correct_pre += 1
            if labItem == 1 and euclidean_distance_item.item() >= THRESHOLD:
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
