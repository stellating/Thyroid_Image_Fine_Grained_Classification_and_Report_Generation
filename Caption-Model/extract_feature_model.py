import torch
import torch.nn as nn
import torchvision
import os
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True


class load_checkpoint(nn.Module):
    """
    Encoder.
    """

    def __init__(self, model_path, feature_num=5, encoded_image_size=1):
        super(load_checkpoint, self).__init__()
        self.enc_image_size = encoded_image_size

        # define model
        resnet = torchvision.models.resnet50(pretrained=False)
        # 提取fc层中固定的参数
        fc_features = resnet.fc.in_features
        # 修改类别为9
        resnet.fc = nn.Linear(fc_features, feature_num)
        #model_path = 'C:/Users/DELL/Downloads/pytorch-book-master/chapter10-image_caption/mycode_background/result/f3_20200516_191915/3_009.ckpt'

        checkpoint = torch.load(model_path)#, map_location=lambda storage, loc: storage.cuda())
        # print(checkpoint)

        # for key in checkpoint['net_state_dict']:
        #     print(key)
        resnet.load_state_dict(checkpoint['net_state_dict'])

        # resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

if __name__ == '__main__':
    model_path = 'C:/Users/DELL/Downloads/pytorch-book-master/chapter10-image_caption/mycode_background/result/f3_20200516_191915/3_009.ckpt'
    load=load_checkpoint(model_path, feature_num=5)
    img = Variable(torch.rand(1, 3, 224, 224))
    result=load(img.cuda())
    dic_feature={}
    dic_feature['image'] = result
    result=result.squeeze(1)
    print(dic_feature)