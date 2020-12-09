import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.enabled = False

def caption_image_beam_search(rev_word_map, encoder, encoder_crop, encoder_text, decoder, feature, word_map, feature_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread('uploads/image.png')
    crop_img = imread('static/crop_img.png')
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    if len(crop_img.shape) == 2:
        crop_img = crop_img[:,:,np.newaxis]
        crop_img = np.concatenate([crop_img,crop_img,crop_img],axis=2)
    img = cv2.resize(img,(256, 256))
    crop_img = cv2.resize(crop_img,(256,256))
    img = img.transpose(2, 0, 1)
    crop_img = crop_img.transpose(2,0,1)
    img = img / 255.
    crop_img = crop_img / 255.
    img = torch.FloatTensor(img).to(device)
    crop_img = torch.FloatTensor(crop_img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    crop_image = transform(crop_img)

    lengths = []

    Feature = torch.cuda.LongTensor(len(feature_map['features'] * 3))
    for i, f in enumerate(feature_map['features']):
        Feature[i * 3] = f
        Feature[i * 3 + 1] = feature_map[str(f)][int(feature[i])]
        Feature[i * 3 + 2] = torch.LongTensor([0])
    lengths.append(len(Feature))
    Feat=Feature.unsqueeze(0)
    # print(Feat.size(),Feat,lengths)
    # print(Feat)
    feature_enc, feature_lens = encoder_text(Feat, lengths)
    words = [rev_word_map[int(ind.data.cpu())] for ind in Feature]
    print(words)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    crop_image = crop_image.unsqueeze(0)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_crop_out = encoder_crop(crop_image)
    enc_image_size = encoder_crop_out.size(1)
    encoder_dim = encoder_out.size(3)
    encoder_crop_dim = encoder_crop_out.size(3)
    encoder_feature_dim = feature_enc.size(2)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    encoder_crop_out = encoder_crop_out.view(1,-1,encoder_crop_dim)
    entire_encoder_out = torch.cat([encoder_crop_out,encoder_out],1)
    num_pixels = entire_encoder_out.size(1)
    crop_num_pixels = encoder_crop_out.size(1)
    feature_num_pixels = feature_enc.size(1)

    # We'll treat the problem as having a batch size of k
    entire_encoder_out = entire_encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    encoder_crop_out = encoder_crop_out.expand(k,crop_num_pixels,encoder_crop_dim)
    feature_enc = feature_enc.expand(k,feature_num_pixels,encoder_feature_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1

    # print(entire_encoder_out.size())

    h_lang, c_lang = decoder.init_hidden_state(entire_encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        # print(k_prev_words)

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        # print(encoder_crop_out.size(),h_lang.size())
        awe, alpha = decoder.attention(encoder_crop_out, h_lang)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h_lang))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        # print(embeddings.size(),awe.size())

        att_input = torch.cat([embeddings, awe], dim=1)
        h_att, c_att = decoder.att_lstm(att_input, (h_lang, c_lang))
        att = decoder.attention1(h_att, entire_encoder_out, feature_enc)
        lang_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = decoder.lang_lstm(lang_input, (h_att, c_att))  # (s, decoder_dim)

        scores = decoder.fc(h_lang)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        # print('complete_inds:',complete_inds)
        # print('incomplete_inds:',incomplete_inds)

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h_lang = h_lang[prev_word_inds[incomplete_inds]]
        c_lang = c_lang[prev_word_inds[incomplete_inds]]
        encoder_crop_out = encoder_crop_out[prev_word_inds[incomplete_inds]]
        entire_encoder_out = entire_encoder_out[prev_word_inds[incomplete_inds]]
        feature_enc = feature_enc[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    # i = complete_seqs_scores.index(max(complete_seqs_scores))
    # seq = complete_seqs
    # alphas = complete_seqs_alpha[i]

    i = np.argsort(complete_seqs_scores)
    seq = [complete_seqs[ind] for ind in i]

    return seq


def visualize_att(image_path, seq, alphas, rev_word_map, result_file, smooth=True):
    """
    Visualizes caption.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param img: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    seq=seq[0]
    words = [rev_word_map[ind] for ind in seq]
    f=open(result_file,'a')
    f.write(''.join(words))
    f.write('\n')
    f.close()

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    image_dir = '/home/swf/gyt/data/images/'
    # parser.add_argument('--img', '-i', default=[image_dir+'000000.jpg',image_dir+'000001.jpg',image_dir+'000002.jpg',image_dir+'000003.jpg',image_dir+'000004.jpg'],help='path to image')
    parser.add_argument('--img', '-i',
                        default=[image_dir + '000005.jpg', image_dir + '000006.jpg', image_dir + '000007.jpg',
                                 image_dir + '000008.jpg', image_dir + '000009.jpg'], help='path to image')

    # parser.add_argument('--img', '-i',
    #                     default=[image_dir + '004837.bmp', image_dir + '004838.bmp', image_dir + '004839.bmp'], help='path to image')

    crop_image_dir = '/home/swf/gyt/data/crop_image_nlp'
    parser.add_argument('--crop_img', '-ci',
                        default=[crop_image_dir + '/000005.jpg', crop_image_dir + '/000006.jpg',
                                 crop_image_dir + '/000007.jpg', crop_image_dir + '/000008.jpg',
                                 crop_image_dir + '/000009.jpg'],
                        help='path to crop image')
    parser.add_argument('--feature', '-f', default=['100131222132300000011504000', '000000000000011112200000002',
                                                    '000000000000011112200000001', '100131212132300000001504000',
                                                    '100131213132300000001504000'], help='feature of image')

    parser.add_argument('--model', '-m', default='/home/swf/gyt/a-PyTorch-Tutorial-to-Image-Captioning_train/checkpoint/BEST-GYT-0911-CROP-alpha-checkpoint_thyroid_10_cap_per_img_0_min_word_freq.pth.tar',help='path to model')
    parser.add_argument('--word_map', '-wm', default='/home/swf/gyt/a-PyTorch-Tutorial-to-Image-Captioning_train/caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json',help='path to word map JSON')
    parser.add_argument('--feature_map', '-fm',
                        default='/home/swf/gyt/a-PyTorch-Tutorial-to-Image-Captioning_train/caption_data_crop/FEATURE_thyroid_10_cap_per_img_0_min_word_freq.json',
                        help='path to word map JSON')

    parser.add_argument('--beam_size', '-b', default=1, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--result_file','-rf',default='/home/swf/gyt/a-PyTorch-Tutorial-to-Image-Captioning_train/results.txt',help='path to result file')
    args = parser.parse_args()

    f = open(args.result_file, 'a')
    f.write(str(args))
    f.write('\n')
    f.close()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    encoder_crop = checkpoint['encoder_crop']
    encoder_crop = encoder_crop.to(device)
    encoder_crop.eval()
    encoder_text = checkpoint['text_encoder']
    # encoder_text = encoder_text.to(device)
    encoder_text.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    # Read feature map
    with open(args.feature_map, 'r') as j:
        feature_map = json.load(j)



    for i in range(len(args.img)):
        img=args.img[i]
        crop_img = args.crop_img[i]
        feat = []
        for f in args.feature[i]:
            feat.append(int(f))
        feature = torch.LongTensor(feat)
        # print(feature)

        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(rev_word_map, args.result_file, encoder, encoder_crop, encoder_text, decoder, img, crop_img, feature, word_map, feature_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)

        # Visualize caption and attention of best sequence
        visualize_att(crop_img, seq, alphas, rev_word_map, args.result_file, args.smooth)
