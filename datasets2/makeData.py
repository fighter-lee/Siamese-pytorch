
import os
import random
import shutil

if __name__ == '__main__':
    dataset_dir = 'Icons-50'
    train_dir = '../learn/demo2/dataset/train'
    test_dir = '../learn/demo2/dataset/test'
    type_suff = '-2'

    # 遍历每个类型的图片文件夹
    for type_dir in os.listdir(dataset_dir):
        type_path = os.path.join(dataset_dir, type_dir)
        if not os.path.isdir(type_path):
            continue

        # 获取该类型图片的列表
        img_list = os.listdir(type_path)

        # 计算训练集和测试集的数量
        train_num = int(len(img_list) * 0.2)
        test_num = int((len(img_list) - train_num) * 0.1)

        # 随机选择训练集和测试集
        train_list = random.sample(img_list, train_num)
        test_list = random.sample(list(set(img_list) - set(train_list)), test_num)

        # 获取目标文件夹路径
        dataset_type_dir = os.path.join(dataset_dir, type_dir)
        train_type_dir = os.path.join(train_dir, type_dir+type_suff)
        test_type_dir = os.path.join(test_dir, type_dir+type_suff)

        # 如果目标文件夹不存在，就创建它
        if not os.path.exists(dataset_type_dir):
            os.makedirs(dataset_type_dir)
        if not os.path.exists(train_type_dir):
            os.makedirs(train_type_dir)
        if not os.path.exists(test_type_dir):
            os.makedirs(test_type_dir)

        # 复制训练集和测试集到相应的目录中
        for img_name in train_list:
            img_path = os.path.join(type_path, img_name)
            shutil.copy(img_path, os.path.join(train_type_dir, img_name))

        for img_name in test_list:
            img_path = os.path.join(type_path, img_name)
            shutil.copy(img_path, os.path.join(test_type_dir, img_name))

