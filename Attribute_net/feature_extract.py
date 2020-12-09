# coding:utf-8
"""
利用resnet50提取图片的语义信息
并保存层results.pth
"""
# 模型
# resnet50 = tv.models.resnet50(pretrained=True)
# del resnet50.fc
# resnet50.fc = lambda x: x
# resnet50.cuda()
import datasets_crop
import torch
import torchvision.transforms as transforms
import tqdm
from lib.network import ResNet50
from torch import nn
def extract_feature():
    # # 数据
    batchsize=16
    featurenum=13
    num_classes = 2
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # dataloader = get_forward_dataloader(opt)
    train_loader = torch.utils.data.DataLoader(
        datasets_crop.FeatureDataset('./data/', 'thyroid_1018_', 'TRAIN', featurenum,
                                     transform=transforms.Compose([normalize])),
        batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets_crop.FeatureDataset('./data/', 'thyroid_1018_', 'TEST', featurenum,
                                     transform=transforms.Compose([normalize])),
        batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True)
    results = torch.Tensor(len(train_loader.dataset), 2048).fill_(0)

    #模型
    target_w = 256
    target_h = 256

    net = ResNet50(target_w,target_h,num_classes)
    if torch.cuda.is_available():
        # net = nn.DataParallel(net)
        net.cuda()

    net.load_state_dict(torch.load('weights/net_f3_2nl_GAP_'+str(featurenum)+'.pth'))
    # net.load_state_dict(torch.load('weights/net_f3_2nl_GAP.pth'))
    #model.forward = lambda x:resnet50.new_forward(model.x)
    #del model.fc
    #model.fc = lambda x:x
    # model.cuda()

    # 前向传播，计算分数
    for ii, (imgs, targets) in tqdm.tqdm(enumerate(train_loader)):
        imgs = imgs.cuda()
        features, _ = net(imgs)
        features = features.view(features.shape[0],-1)
        results[ii * batchsize:(ii + 1) * batchsize] = features.data.cpu()

    # 200000*2048 20万张图片，每张图片2048维的feature
    torch.save(results, 'results_f'+str(featurenum)+'.pth')

def reExtract_features():
    pth_list = ['results_f3.pth','results_f4.pth','results_f5.pth','results_f6.pth','results_f7.pth','results_f8.pth','results_f9.pth','results_f10.pth','results_f11.pth']
    pth_feature_list=[]
    for i in pth_list:
        feature=torch.load(i)
        pth_feature_list.append(feature)
    root_pth='results_f3.pth'
    features = torch.load(root_pth)
    img_num = features.shape[0]
    pth_num = len(pth_list)
    results = torch.Tensor(img_num, pth_num, 2048).fill_(0)
    for i in range(img_num):
        for j,pth_features in enumerate(pth_feature_list):
            print(pth_features.shape)
            results[i,j,:]=pth_features[j,:]
    print(results.shape)
    torch.save(results, 'results_f5.pth')



if __name__ == '__main__':
    #train(opt)
    # extract_feature()
    # results = torch.load('results_f3.pth')
    # print(results.shape)
    reExtract_features()

