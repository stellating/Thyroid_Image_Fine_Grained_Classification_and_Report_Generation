import torch
import torch.utils.data as Data
import torchvision
from lib.network import ResNet50
from torch import nn
from torch.cuda import amp
import torchvision.transforms as transforms
import time
import argparse
import datasets_crop
import math

parser = argparse.ArgumentParser()
# # Data input settings
parser.add_argument('--featurenum', type=int, default=3,help='train feature')
parser.add_argument('--batchsize',type=int, default=16,help='batchsize of train epoach')
parser.add_argument('--resume',type=bool, default=False,help='batchsize of train epoach')
parser.add_argument('--learning_rate',type=float, default=0.0000001,help='batchsize of train epoach')
parser.add_argument('--num_classes',type=int, default=6,help='classes number of features')
parser.add_argument('--SAVE_FREQ',type=int, default=1,help='save frequant of epoach')
parser.add_argument('--save_dir',type=str, default="./checkpoints/",help='save dir of checkpoints')
args = parser.parse_args()
# read dataset
# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

print("Starting reading dataset......")

train_loader = torch.utils.data.DataLoader(datasets_crop.FeatureDataset('./data/', 'thyroid_1018_', 'TRAIN', args.featurenum,transform=transforms.Compose([normalize])),
        batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(datasets_crop.FeatureDataset('./data/', 'thyroid_1018_', 'TEST', args.featurenum,transform=transforms.Compose([normalize])),
        batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
print("Dataset reading done!")

train_batch_num = len(train_loader)
test_batch_num = len(test_loader)
target_w=256
target_h=256
net = ResNet50(target_w,target_h,args.num_classes)
if torch.cuda.is_available():
    # net = nn.DataParallel(net)
    print(net)
    net.cuda()

# +++++++++++++++++++++++++++++++
scaler = amp.GradScaler()
# +++++++++++++++++++++++++++++++

opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
loss_func = nn.CrossEntropyLoss()

for epoch_index in range(50):
    st = time.time()

    torch.set_grad_enabled(True)
    net.train()
    total_train_loss = []
    total_train_acc = 0
    total_train_sample = 0
    for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            label_batch = label_batch.cuda()
            # print(img_batch.shape,label_batch.shape)

        # ++++++++++++++++++++++++++++++++++++++++++++++
        # predict = net(img_batch)
        # loss = loss_func(predict, label_batch)
        with amp.autocast():
            # y = net(torch.randn(1, 3, target_w, target_h).cuda())
            _,predict = net(img_batch)
            label_batch=label_batch.squeeze()
            loss = loss_func(predict, label_batch)
        # ++++++++++++++++++++++++++++++++++++++++++++++

        net.zero_grad()
        # ++++++++++++++++++++++++++++++++++++++++++++++
        # loss.backward()
        # opt.step()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # ++++++++++++++++++++++++++++++++++++++++++++++
        predict = predict.argmax(dim=1)
        acc = (predict == label_batch).sum()

        total_train_loss.append(loss)
        total_train_acc += acc
        total_train_sample += img_batch.size(0)

        mean_acc = total_train_acc.item() * 1.0 / total_train_sample
        mean_loss = sum(total_train_loss) / total_train_loss.__len__()

        print('[Train] epoch[%d/%d] acc:%.4f%% loss:%.4f'
              % (epoch_index, 50, mean_acc * 100, mean_loss.item()))

    print('(LR:%f) Time of a epoch:%.4fs' % (opt.param_groups[0]['lr'], time.time()-st))

    torch.set_grad_enabled(False)
    net.eval()
    total_loss = []
    total_acc = 0
    total_sample = 0

    for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            label_batch = label_batch.cuda()

        _,predict = net(img_batch)
        label_batch = label_batch.squeeze()
        loss = loss_func(predict, label_batch)

        predict = predict.argmax(dim=1)
        acc = (predict == label_batch).sum()

        total_loss.append(loss)
        total_acc += acc
        total_sample += img_batch.size(0)

    mean_acc = total_acc.item() * 1.0 / total_sample
    mean_loss = sum(total_loss) / total_loss.__len__()

    print('[Test] epoch[%d/%d] acc:%.4f%% loss:%.4f\n'
          % (epoch_index, 50, mean_acc * 100, mean_loss.item()))

weight_path = 'weights/net_f3_2nl_GAP_'+str(args.featurenum)+'.pth'
print('Save Net weights to', weight_path)
net.cpu()
torch.save(net.state_dict(), weight_path)
