import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision.transforms as transforms


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
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

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load features (completely into memory)
        with open(os.path.join(data_folder, self.split + '_FEATURES_' + data_name + '.json'), 'r') as j:
            self.features = json.load(j)

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_idx = i//self.cpi
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        crop_img = torch.FloatTensor(self.crop_imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
            crop_img = self.transform(crop_img)

        # print(self.features[0][0],self.captions[0])
        feat=[]
        for f in self.features[0][i//self.cpi]:
            feat.append(int(f))
        feature = torch.LongTensor(feat)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            all_captions = torch.LongTensor(self.captions[((i//self.cpi)*self.cpi):(((i//self.cpi)*self.cpi)+self.cpi)])
            return img_idx, img, crop_img, feature, caption, caplen, all_captions
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img_idx, img, crop_img, feature, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset('./caption_data_crop/', 'thyroid_10_cap_per_img_0_min_word_freq', 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    print(len(train_loader))
    with open(os.path.join('./caption_data_crop/', 'FEATURE_' + 'thyroid_10_cap_per_img_0_min_word_freq' + '.json'), 'r') as j:
        feature_map = json.load(j)
    with open('./caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json', 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    for i, (img_idx, imgs, crop_imgs, feature, caps, caplens,all_captions) in enumerate(train_loader):
        count=0
        cover_label=[]
        print('caption:',caps)
        for j in caps:
            jj=j.numpy()
            print(str([rev_word_map[ind] for ind in jj]))

        print('all_captions:',all_captions)
        for j in all_captions:
            for ji in j:
                jj=ji.numpy()
                print(str([rev_word_map[ind] for ind in jj]))

        lengths = []
        feature_batch_size = feature.size(0)
        Feat = torch.LongTensor(feature_batch_size, len(feature_map['features'] * 3)).fill_(0)

        for j in range(feature_batch_size):
            Feature = torch.LongTensor(len(feature_map['features'] * 3))
            for ii, f in enumerate(feature_map['features']):
                # if f in caps:
                #     count=count+1
                if feature_map[str(f)][int(feature[j,ii])] in caps:
                    count=count+1
                    cover_label.append(feature_map[str(f)][int(feature[j,ii])])
                Feature[ii * 3] = f
                Feature[ii * 3 + 1] = feature_map[str(f)][int(feature[j, ii])]
                Feature[ii * 3 + 2] = torch.LongTensor([0])
            lengths.append(len(Feature))
            Feat[j] = Feature
        print('Feat:',Feat)
        for i in Feat:
            ii=i.numpy()
            print(str([rev_word_map[ind] for ind in ii]))
        print('count:',count)
        print(cover_label)

        cover_word=str([rev_word_map[ind] for ind in cover_label])
        print(cover_word)


