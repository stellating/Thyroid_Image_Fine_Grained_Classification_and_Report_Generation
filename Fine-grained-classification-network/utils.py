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


def create_input_files(dataset, karpathy_json_path, feature_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
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

    with open(feature_json_path, 'r') as j:
        features = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_bboxs = []
    train_image_feature = []
    val_image_paths = []
    val_image_captions = []
    val_image_bboxs = []
    val_image_feature = []
    test_image_paths = []
    test_image_captions = []
    test_image_bboxs = []
    test_image_feature = []
    word_freq = Counter()

    for key in features:
        word_freq.update(features[key])
    print(word_freq)

    for img in data['images']:
        captions = []
        for c in img['tokens']:
            # Update word frequency
            word_freq.update(c)
            if len(c) <= max_len:
                captions.append(c)
            # print(word_freq)

        if len(captions) == 0:
            continue


        #path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
          #  image_folder, img['filename'])

        path = os.path.join(image_folder, img['imgid'].zfill(6)+'.'+img['imgpath'].split('.')[-1])
        # print(path)

        bbox = img['bbox']
        feature = img['feature']

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_bboxs.append(bbox)
            train_image_feature.append(feature)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_bboxs.append(bbox)
            val_image_feature.append(feature)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_bboxs.append(bbox)
            test_image_feature.append(feature)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)
    assert len(train_image_bboxs) == len(train_image_paths)
    assert len(val_image_bboxs) == len(val_image_paths)
    assert len(test_image_bboxs) == len(test_image_paths)
    print(len(train_image_paths),len(val_image_paths),len(test_image_paths))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imbbox, imfeature, split in [(train_image_paths, train_image_captions, train_image_bboxs, train_image_feature, 'TRAIN'),
                                   (val_image_paths, val_image_captions, val_image_bboxs, val_image_feature, 'VAL'),
                                   (test_image_paths, test_image_captions, test_image_bboxs, test_image_feature, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            crop_images = h.create_dataset('crop_images',(len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []
            features = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                features.append(imfeature)

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
                crop_img = imresize(crop_img, (256, 256))
                img = img.transpose(2, 0, 1)
                crop_img = crop_img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert crop_img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                assert np.max(crop_img) <= 255

                # Save image to HDF5 file
                images[i] = img
                crop_images[i] = crop_img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)


            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            assert images.shape[0] == len(features)

            with open(os.path.join(output_folder, split + '_FEATURES_' + base_filename + '.json'),'w') as j:
                json.dump(features, j)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

def load_feature(word_map_file, feature_file, output_json):
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    with open(feature_file, 'r') as j:
        feature_dic = json.load(j)

    print(word_map['<unk>'])
    new_dic={}
    for key in feature_dic:
        if key!='features':
            new_dic[word_map.get(key, word_map['<unk>'])] = [word_map.get(word, word_map['<unk>']) for word in feature_dic[key]]
        else:
            new_dic['features']=[word_map.get(word, word_map['<unk>']) for word in feature_dic[key]]
    print(new_dic)

    with open(output_json, 'w') as j:
        json.dump(new_dic, j)
    return new_dic


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, encoder_crop, text_encoder, decoder, encoder_optimizer, encoder_crop_optimizer,
                    decoder_optimizer, bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'encoder_crop':encoder_crop,
             'text_encoder':text_encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'encoder_crop_optimizer':encoder_crop_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = './checkpoint/GYT-1209-CROP-checkpoint_' + data_name + '.pth.tar'
    filebestname = './checkpoint/BEST-GYT-1209-CROP-alpha-checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, filebestname)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        print(dist)
        dist = dist + dist.t()
        print(dist)
        dist.addmm_(1, -2, inputs, inputs.t())
        print(dist)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        print(dist)


        # For each anchor, find the hardest positive and negative
        print(targets,n)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        print(mask)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        print(dist_ap)
        print(dist_an)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

if __name__ == '__main__':
    criterion = TripletLoss(margin=0.4)
    # scores = torch.tensor([186,12,13,14,15,0,0,0,0,0,0,0,0,0,0,0,0])
    # targets = torch.tensor([186,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    scores = torch.rand(3,3)
    targets = torch.rand(3,3)
    print(scores)
    print(targets)
    loss = criterion(scores, targets)
