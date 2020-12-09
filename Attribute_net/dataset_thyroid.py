import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
import torch

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
#        img_txt_file = open(os.path.join(self.root, 'images.txt'))
#        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
#        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
#        img_name_list = [] #用来存放所有的图片名  ***.img
#        for line in img_txt_file:
#            img_name_list.append(line[:-1].split(' ')[-1])  # line[:-1]去除文本最后一个字符（换行符）后剩下的部分 以空格为分隔符保留最后一段
#
#        label_list = [] #存放图片序号对应的类别 改为了从0开始
#        for line in label_txt_file:
#            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
#
#        train_test_list = []  #取对用的分类用数字 0或1
#        for line in train_val_file:
#            train_test_list.append(int(line[:-1].split(' ')[-1]))
#
#        #把训练集和测试集的文件名分别存在对应list  1就放在训练集 0是测试集
#
#        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i] #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
#        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
#
#



        traindir='/data1/gyt/data/caption_data_dmgk/train/'

        testdir='/data1/gyt/data/caption_data_dmgk/test/'

        train_list=os.listdir(traindir)

        test_list=os.listdir(testdir)









        #训练集  读取图片与对应类别
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'train', train_file)) for train_file in
                              train_list[:data_len]]
            self.train_label = [int(train_file.split('.')[0].split('_')[1]) for train_file in
                              train_list[:data_len]]
        #测试集  读取图片与对应类别
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'test', test_file)) for test_file in
                             test_list[:data_len]]
            self.test_label = [int(test_file.split('.')[0].split('_')[1]) for test_file in
                              test_list[:data_len]]


    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            #img = transforms.Resize((224, 224), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            #img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            #img = transforms.Resize((224, 224), Image.BILINEAR)(img)   #重置图像分辨率
            img = transforms.CenterCrop(INPUT_SIZE)(img)                #功能：依据给定的size从中心裁剪
            img = transforms.ToTensor()(img)                            #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)       #对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class Thyroid():
    def __init__(self, root, is_train=True, feature_num=None, data_len=None):
        #root文件夹下必须有annotations.txt,train_test_split.txt
        #annotation里边写了image_name，label;     train_test_split.txt写了image_name,is_train;
        #feature_num是第几个feature,和文件中的feature在字符串中所在的位置一样
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'annotations.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = [] #用来存放所有的图片名  ***.jpg
        label_list = []  # 存放图片序号对应的类别 改为了从0开始
        train_test_list = []  # 取对用的分类用数字 0或1
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[0])  # line[:-1]去除文本最后一个字符（换行符）后剩下的部分 以空格为分隔符保留最后一段
            label_list.append(int(line[:-1].split(' ')[-1][feature_num]))
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))    #1是训练，0是测试

#        #把训练集和测试集的文件名分别存在对应list  1就放在训练集 0是测试集
#
        train_list = [x for i, x in zip(train_test_list, img_name_list) if i] #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        train_label_list = [x for i,x in zip(train_test_list,label_list) if i]
        test_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        test_label_list = [x for i,x in zip(train_test_list,label_list) if not i]

#
        #训练集  读取图片与对应类别
        if self.is_train:
            self.train_list=train_list
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'all', train_file)) for train_file in
                              train_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        #测试集  读取图片与对应类别
        if not self.is_train:
            self.test_list=test_list
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'all', test_file)) for test_file in
                             test_list[:data_len]]
            self.test_label = test_label_list[:data_len]


    def __getitem__(self, index):
        if self.is_train:
            #img_name=self.train_list[index]
            #print('----------------------',img_name)
            img, target = self.train_img[index], self.train_label[index]
            #print(type(img))
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)   #
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((224, 224), Image.BILINEAR)(img)
            #img = transforms.RandomCrop(INPUT_SIZE)(img)
            #img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            #img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((224, 224), Image.BILINEAR)(img)   #重置图像分辨率
            #img = transforms.CenterCrop(INPUT_SIZE)(img)                #功能：依据给定的size从中心裁剪
            img = transforms.ToTensor()(img)                            #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            #img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)       #对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    f_num=3
    dataset = Thyroid(root='./thyroid_data_20191031/',is_train=True,feature_num=f_num)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                                   shuffle=True, num_workers=0, drop_last=False)
    #dataset = (root='../data/new_thyroid_data_20190923/thyroid_data_20191031/all/',is_train=true,feature_num=f_num)
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in train_dataloader:
        #img, label = data[0].cuda(), data[1].cuda()
        print(data[0].size(), data[1])
    dataset = Thyroid(root='./thyroid_data_20191031/', is_train=False,feature_num=f_num)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
