
import torch.nn.functional as F

def getAccuracy(output1, output2, label):
    distance = F.pairwise_distance(output1, output2)
    zero_labels = label == 0  # 找出labels中为0的位置
    predict_for_zero_labels = distance[zero_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
    less_than_1_15 = predict_for_zero_labels < 1.15  # 找出浮点类型数小于1.15的位置
    accuracy = less_than_1_15.sum().item()

    one_labels = label == 1  # 找出labels中为1的位置
    predict_for_one_labels = distance[one_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
    greater_than_1_15 = predict_for_one_labels > 1.15  # 找出浮点类型数大于1.15的位置
    accuracy += greater_than_1_15.sum().item()  # 计算浮点类型数大于1.15的数量
    return accuracy