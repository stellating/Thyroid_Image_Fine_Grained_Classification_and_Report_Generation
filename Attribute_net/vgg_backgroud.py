from __future__ import print_function, division

import os
import argparse
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn import DataParallel
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
import datasets_crop
from utils import init_log,progress_bar

parser = argparse.ArgumentParser()
# # Data input settings
parser.add_argument('--featurenum', type=int, default=0,help='train feature')
parser.add_argument('--batchsize',type=int, default=16,help='batchsize of train epoach')
parser.add_argument('--resume',type=bool, default=False,help='batchsize of train epoach')
parser.add_argument('--learning_rate',type=float, default=0.000001,help='batchsize of train epoach')
parser.add_argument('--num_classes',type=int, default=5,help='classes number of features')
parser.add_argument('--SAVE_FREQ',type=int, default=1,help='save frequant of epoach')
parser.add_argument('--save_dir',type=str, default="./checkpoints/",help='save dir of checkpoints')
args = parser.parse_args()
# read dataset
# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
print("Starting reading dataset......")

train_dataloader = torch.utils.data.DataLoader(datasets_crop.FeatureDataset('./data/', 'thyroid_1018_', 'TRAIN', args.featurenum,transform=transforms.Compose([normalize])),
        batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(datasets_crop.FeatureDataset('./data/', 'thyroid_1018_', 'TEST', args.featurenum,transform=transforms.Compose([normalize])),
        batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
print("Dataset reading done!")
# define model
model = torchvision.models.resnet50(pretrained=True)
#提取fc层中固定的参数
fc_features = model.fc.in_features
print("Revise num_classes as: "+str(args.num_classes))
#修改类别为9
model.fc = nn.Linear(fc_features, args.num_classes)
for p in model.parameters():
    p.requires_grad = True
if torch.cuda.is_available():
    print("Cuda available!")
    model.cuda()
start_epoch=0
save_dir = os.path.join(args.save_dir,'f'+str(args.featurenum)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
print("Save checkpoints in: "+save_dir)
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

if args.resume:
    print("Resume model from: "+args.resume)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch']

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_func = nn.CrossEntropyLoss()

Loss_list = []
Accuracy_list = []

for epoch in range(start_epoch,500):
    _print('epoch {}'.format(epoch))
    # training-----------------------------
    model.train()
    for i,data in enumerate(train_dataloader):
        img,label = data[0].cuda(), data[1].cuda()
        #img,label=img.cuda(),label.cuda()
        batch_size=img.size(0)
        optimizer.zero_grad()

        out = model(img)
        # print(out, label)
        label = label.squeeze()
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()
        progress_bar(i, len(train_dataloader), 'train')
    if epoch % args.SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        model.eval()
        for i, data in enumerate(train_dataloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                out=model(img)
                label = label.squeeze()
                loss = loss_func(out, label)
                #pred = torch.max(out, 1)[1]
                _, pred = torch.max(out, 1)
                total+=batch_size
                train_correct += torch.sum(pred == label).data[0]
                #train_acc += train_correct.data[0]
                train_loss += loss.item() * batch_size
                progress_bar(i, len(train_dataloader), 'eval train set')
        train_acc = float(train_correct)/total
        train_loss = train_loss/total
        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

        # evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                concat_logits= model(img)
                # calculate loss
                label = label.squeeze()
                concat_loss = loss_func(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict == label).data[0]
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(test_dataloader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

        # save model
        net_state_dict = model.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    print('finishing training')

