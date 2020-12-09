import os
import numpy as np
import h5py
import json
import torch
from torch import nn
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder,output_folder,
                       ):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k','thyroid'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)


    # Read image paths and captions for each image
    train_image_paths = []
    train_image_feature = []
    train_image_bboxs = []
    train_image_fglabels = []
    val_image_paths = []
    val_image_feature = []
    val_image_bboxs = []
    val_image_fglabels = []
    test_image_paths = []
    test_image_feature = []
    test_image_bboxs = []
    test_image_fglabels = []

    for img in data['images']:
        path = os.path.join(image_folder, img['imgid'].zfill(6)+'.'+img['imgpath'].split('.')[-1])
        # print(path)

        bbox = img['bbox']
        feature = img['feature']
        fg_label = img['fg_label']
        # print('feature:',feature)

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_bboxs.append(bbox)
            train_image_feature.append(feature)
            train_image_fglabels.append(fg_label)

        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_bboxs.append(bbox)
            val_image_feature.append(feature)
            val_image_fglabels.append(fg_label)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_bboxs.append(bbox)
            test_image_feature.append(feature)
            test_image_fglabels.append(fg_label)

    # Sanity check
    assert len(train_image_paths) == len(train_image_feature)
    assert len(val_image_paths) == len(val_image_feature)
    assert len(test_image_paths) == len(test_image_feature)
    assert len(train_image_bboxs) == len(train_image_paths)
    assert len(val_image_bboxs) == len(val_image_paths)
    assert len(test_image_bboxs) == len(test_image_paths)
    assert len(train_image_fglabels) == len(train_image_feature)
    assert len(train_image_fglabels) == len(train_image_paths)
    print(len(train_image_paths),len(val_image_paths),len(test_image_paths))

    # # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0
    #
    # rev_word_map = {v: k for k, v in word_map.items()}
    #
    # Create a base/root name for all output files
    base_filename = dataset+'_1018_'
    #
    # # Save word map to a JSON
    # with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
    #     json.dump(word_map, j)



    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imbbox, imfeature, imfglabel, split in [(train_image_paths, train_image_bboxs, train_image_feature, train_image_fglabels, 'TRAIN'),
                                   (val_image_paths, val_image_bboxs, val_image_feature, val_image_fglabels, 'VAL'),
                                   (test_image_paths, test_image_bboxs, test_image_feature, test_image_fglabels, 'TEST')]:

        # print('imfeature:',imfeature)
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:

            # Create dataset inside HDF5 file to store images
            if split=='TRAIN':
                len_img=2011   #440,219
            elif split=='VAL':
                len_img=671    #140,69
            else:
                len_img=672   #311,259

            images = h.create_dataset('images', (len_img, 3, 256, 256), dtype='uint8')

            crop_images = h.create_dataset('crop_images',(len_img, 3, 32, 32), dtype='uint8')

            print("\nReading %s images, storing to file...\n" % split)

            features = []
            fglabels = []
            img_count=0

            for i, path in enumerate(tqdm(impaths)):
                if i==1711:
                    continue
                features.append(imfeature[i])
                fglabels.append(imfglabel[i])
                # Read images
                img = imread(impaths[i])
                xy = imbbox[i].split(',')
                x1 = int(float(xy[0]) * img.shape[1])
                y1 = int(float(xy[1]) * img.shape[0])
                x2 = int(float(xy[2]) * img.shape[1])
                y2 = int(float(xy[3]) * img.shape[0])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                crop_img = img[y1:y2 + 1, x1:x2 + 1]
                img = imresize(img, (256, 256))
                crop_img = imresize(crop_img, (32, 32))
                img = img.transpose(2, 0, 1)
                crop_img = crop_img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert crop_img.shape == (3, 32, 32)
                assert np.max(img) <= 255
                assert np.max(crop_img) <= 255

                # Save image and Features to HDF5 file
                images[img_count] = img
                crop_images[img_count] = crop_img
                img_count = img_count+1

            # Sanity check
            assert images.shape[0] == len(features)

            with open(os.path.join(output_folder, split + '_FEATURES_' + base_filename + '.json'),'w') as j:
                json.dump(features, j)

            with open(os.path.join(output_folder, split + '_FGLABELS_' + base_filename + '.json'),'w') as j:
                json.dump(fglabels, j)

if __name__ == '__main__':
    # load_feature(
    #     word_map_file='../Image_text_matching/caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json',
    #     feature_file='feature_CN.json',
    #     output_json='caption_data_crop_triplet/FEATURE_thyroid_10_cap_per_img_0_min_word_freq.json')
    # Create input files (along with word map)
    create_input_files(dataset='thyroid',
                       karpathy_json_path='./fine_grain_thyroid_gyt_20201018.json',
                       image_folder='../data/Image/',
                       output_folder='./data')


