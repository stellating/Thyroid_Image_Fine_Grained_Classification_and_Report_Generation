import torch
import torch.utils.data as Data
import torchvision
from lib.network import ResNet50
from torch import nn
from torch.cuda import amp
import torchvision.transforms as transforms
import time
import argparse
from scipy.misc import imread, imresize
import numpy as np
import datasets_crop
import math

parser = argparse.ArgumentParser()
# # Data input settings
parser.add_argument('--featurenum', type=int, default=3,help='train feature')
parser.add_argument('--resume',type=bool, default=False,help='batchsize of train epoach')
parser.add_argument('--num_classes',type=int, default=6,help='classes number of features')
parser.add_argument('--image_path',type=str, default="./thyroid_data_20191031/all/1_1.jpg",help='image path')
args = parser.parse_args()
# read dataset
# Custom dataloaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = imread(args.image_path)
if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)
img = imresize(img, (256, 256))
img = img.transpose(2, 0, 1)
img = img / 255.
img = torch.FloatTensor(img).to(device)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])
image = transform(img)  # (3, 256, 256)

target_w=256
target_h=256
net = ResNet50(target_w,target_h,args.num_classes)
if torch.cuda.is_available():
    # net = nn.DataParallel(net)
    print(net)
    net.cuda()
if torch.cuda.is_available():
    img_batch = img.cuda()

_,predict = net(img_batch)
predict = predict.argmax(dim=1)


