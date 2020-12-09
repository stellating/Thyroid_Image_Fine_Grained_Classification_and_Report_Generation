import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision.transforms as transforms
import random
import numpy as np


class FeatureDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, attribute, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.crop_imgs = self.h['crop_images']

        # Load features (completely into memory)
        with open(os.path.join(data_folder, self.split + '_FEATURES_' + data_name + '.json'), 'r') as j:
            self.features = json.load(j)
        with open(os.path.join(data_folder, self.split + '_FGLABELS_' + data_name + '.json'), 'r') as j:
            self.fglabels = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        self.dataset_size = len(self.imgs)

        self.attribute = attribute


    def __getitem__(self, i):
        # print(i)
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i] / 255.)

        crop_img = torch.FloatTensor(self.crop_imgs[i] / 255.)

        if self.transform is not None:
            img = self.transform(img)
            crop_img = self.transform(crop_img)

        feature = self.features[i]

        fglabel = self.fglabels[i]

        # print(feature,feature[self.attribute])

        label = torch.LongTensor([int(feature[self.attribute])])

        # label = torch.LongTensor([int(fglabel)])
        # print(crop_img.shape,label)

        return img, label

    def __len__(self):
        return self.dataset_size

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset('./caption_data_crop_triplet_TEST/', 'thyroid_5_cap_per_img_0_min_word_freq', 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # print(len(train_loader))
    with open(os.path.join('../Image_text_matching/caption_data_crop/', 'FEATURE_' + 'thyroid_10_cap_per_img_0_min_word_freq' + '.json'), 'r') as j:
        feature_map = json.load(j)
    with open('../Image_text_matching/caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json', 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    for i, (img_idx, img0, img1, img2, crop_img0, crop_img1, crop_img2, feature0, feature1, feature2, caption0, caption1, caption2, caplen0, caplen1, caplen2, Feat0, Feat1, Feat2, cover_label0, cover_label1, cover_label2) in enumerate(train_loader):
        count=0
        if i==len(train_loader)-6:
            continue

        # print(feature0)
        # print(feature1)
        # print(feature2)

        # print('Feat:',Feat.shape)
        # print('cover_label:',cover_label)
        # print('caps:',caps.shape)
        # print('caplens:',caplen)
        # print('rand_2_captions:',rand_2_captions.shape)
        # print('rand_2_caplens:',rand_2_caplens)
        # print('rand_2_cover_label:',rand_2_cover_label)

        # lengths = []
        # for i1 in Feat0:
            # lengths.append(i1.shape[0])
        # print('caption:',caps)
        # for j in caps:
        #     jj=j.numpy()
        #     print(str([rev_word_map[ind] for ind in jj]))

        # print('all_captions:',all_captions)
        # print(rand_2_captions[0][1])
        # print('rand_2_caps:')
        # for j in rand_2_captions:
        #     for ji in j:
        #         jj=ji.numpy()
        #         print(str([rev_word_map[ind] for ind in jj]))

        lengths = []
        feature_batch_size = feature0.size(0)
        Feat = torch.LongTensor(feature_batch_size, len(feature_map['features'] * 3)).fill_(0)

        # cover_label: (batch_size, 1, 2)
        cover_label = torch.IntTensor(feature_batch_size, len(caption0[0])).fill_(0)
        cover_good = []

        for j in range(feature_batch_size):
            flag = 0
            Feature = torch.LongTensor(len(feature_map['features'] * 3))
            count = 0
            for ii, f in enumerate(feature_map['features']):

                Feature[ii * 3] = f
                Feature[ii * 3 + 1] = feature_map[str(f)][int(feature0[j, ii])]
                Feature[ii * 3 + 2] = torch.LongTensor([0])
            lengths.append(len(Feature))
            Feat[j] = Feature

            # print(cover_label0)


