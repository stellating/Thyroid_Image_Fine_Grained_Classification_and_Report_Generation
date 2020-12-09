# coding:utf-8
"""
利用resnet50提取图片的语义信息
并保存层results.pth
"""
from config import Config
import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image
import numpy as np
from lib.network import *
from torch import nn
import time

t.set_grad_enabled(False)
opt = Config()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class CaptionDataset(data.Dataset):

    def __init__(self, caption_data_path):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            normalize
        ])

        data = t.load(caption_data_path)
        self.ix2id = data['ix2id']
        self.ix_label = data['label']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                     for _ in self.ix2id]
        self.labels = [self.ix_label[_] for _ in self.ix_label]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        label = self.labels[index]
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class Caption_Dataset(data.Dataset):

    def __init__(self, caption_data_path):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            normalize
        ])

        data = t.load(caption_data_path)
        self.ix2id = data['ix2id']
        #self.ix_label = data['label']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                     for _ in self.ix2id]
        #self.labels = [self.ix_label[_] for _ in self.ix_label]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        #label = self.labels[index]
        img = self.transforms(img)
        return img, index

    def __len__(self):
        return len(self.imgs)

def get_dataloader(opt):
    train_dataset = CaptionDataset(opt.train_data_path)
    train_dataloader = data.DataLoader(train_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 #num_workers=opt.num_workers,
                                 )
    test_dataset= CaptionDataset(opt.test_data_path)
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=False)
    return train_dataloader,test_dataloader

def get_forward_dataloader(opt):
    dataset = Caption_Dataset(opt.data_path)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 #num_workers=opt.num_workers,
                                 )
    return dataloader

def train(opt):
    opt.batch_size = 32
    train_dataloader,test_dataloader = get_dataloader(opt)
    net = resnet50()
    if t.cuda.is_available():
        net = nn.DataParallel(net)
        net.cuda()

    opt_func  = t.optim.Adam(net.parameters(), lr=0.000001)
    loss_func = nn.CrossEntropyLoss()

    for epoch_index in range(10):
        st = time.time()

        t.set_grad_enabled(True)
        net.train()
        for train_batch_index, (img_batch, label_batch) in enumerate(train_dataloader):
            if t.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            _,predict = net(img_batch)
            loss = loss_func(predict, label_batch)

            net.zero_grad()
            loss.backward()
            opt_func.step()

        print('(LR:%f) Time of a epoch:%.4fs' % (opt_func.param_groups[0]['lr'], time.time() - st))

        t.set_grad_enabled(False)
        net.eval()
        total_loss = []
        total_acc = 0
        total_sample = 0


        for test_batch_index, (img_batch, label_batch) in enumerate(test_dataloader):
            if t.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            _,predict = net(img_batch)
            loss = loss_func(predict, label_batch)

            predict = predict.argmax(dim=1)
            acc = (predict == label_batch).sum()

            total_loss.append(loss)
            total_acc += acc
            total_sample += img_batch.size(0)

        net.train()

        mean_acc = total_acc.item() * 1.0 / total_sample
        mean_loss = sum(total_loss) / total_loss.__len__()

        print('[Test] epoch[%d/%d] acc:%.4f%% loss:%.4f\n'
              % (epoch_index, 10, mean_acc * 100, mean_loss.item()))
    t.save(net.state_dict(), './params.pth')




# 模型
# resnet50 = tv.models.resnet50(pretrained=True)
# del resnet50.fc
# resnet50.fc = lambda x: x
# resnet50.cuda()

def extract_feature(dataloader):
    feature_model_path=''
    # # 数据
    batch_size = 32
    # dataloader = get_forward_dataloader(opt)
    results = t.Tensor(len(dataloader.dataset), 14, 14, 2048).fill_(0)
    # batch_size = opt.batch_size

    #模型
    model = t.load(feature_model_path)
    # print(model)
    #model.forward = lambda x:resnet50.new_forward(model.x)
    #del model.fc
    #model.fc = lambda x:x
    model.cuda()
    alldata={}

    # 前向传播，计算分数
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        # 确保序号没有对应错
        imgs = imgs.cuda()
        features, _ = model(imgs)
        results[i:i + 1] = features.data.cpu()

    # 200000*2048 20万张图片，每张图片2048维的feature
    # t.save(results, 'results.pth')
    return results

if __name__ == '__main__':
    #train(opt)
    extract_feature()

