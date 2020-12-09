import torch
import torchvision.transforms as transforms
from extract_feature_model import load_checkpoint
from datasets import CaptionDataset
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

def cal_feat(images, feat, feat_num):
    model_path = 'C:/Users/DELL/Downloads/pytorch-book-master/chapter10-image_caption/mycode_background/result/' + feat + '_009.ckpt'
    model = load_checkpoint(model_path, feature_num=feat_num)
    model = model.to(device)
    model.eval()

    # feature0 = torch.Tensor(len(images), 2048).fill_(0)

    # Move to device, if available
    imgs_origin = images.to(device)
    feature = model(imgs_origin)
    feature0 = feature.data.cpu()

    return feature0