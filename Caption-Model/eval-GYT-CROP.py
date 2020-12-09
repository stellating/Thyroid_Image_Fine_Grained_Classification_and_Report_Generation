import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets_crop import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Parameters
data_folder = './caption_data_crop/'  # folder with data files saved by create_input_files.py
data_name = 'thyroid_10_cap_per_img_0_min_word_freq'  # base name shared by data files
checkpoint = './checkpoint/BEST-GYT-0911-CROP-alpha-checkpoint_thyroid_10_cap_per_img_0_min_word_freq.pth.tar'  # model checkpoint
word_map_file = './caption_data_crop/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
feat_map_file = './caption_data_crop/FEATURE_thyroid_10_cap_per_img_0_min_word_freq.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
encoder_crop = checkpoint['encoder_crop']
# encoder_crop = encoder_crop.to(device)
encoder_crop.eval()
encoder_text = checkpoint['text_encoder']
# encoder_text = encoder_text.to(device)
encoder_text.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Read feature map
with open(feat_map_file, 'r') as j:
    feature_map = json.load(j)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()


    # For each image
    for i, (img_idx, image, crop_img, feature, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        lengths = []
        feature_batch_size = feature.size(0)
        Feat = torch.LongTensor(feature_batch_size, len(feature_map['features'] * 3)).fill_(0)

        for j in range(feature_batch_size):
            Feature = torch.LongTensor(len(feature_map['features'] * 3))
            for i, f in enumerate(feature_map['features']):
                Feature[i * 3] = f
                Feature[i * 3 + 1] = feature_map[str(f)][int(feature[j, i])]
                Feature[i * 3 + 2] = torch.LongTensor([0])
            lengths.append(len(Feature))
            Feat[j] = Feature
        # print(Feat)

        feature_enc, feature_lens = encoder_text(Feat, lengths)

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        crop_img = crop_img.to(device)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)


        encoder_crop_out = encoder_crop(crop_img)
        enc_image_size = encoder_crop_out.size(1)

        encoder_dim = encoder_out.size(3)
        encoder_crop_dim = encoder_crop_out.size(3)
        encoder_feature_dim =feature_enc.size(2)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        encoder_crop_out = encoder_crop_out.view(1,-1,encoder_dim)
        entire_encoder_out = torch.cat([encoder_out,encoder_crop_out],1)
        num_pixels = entire_encoder_out.size(1)
        crop_num_pixels = encoder_crop_out.size(1)
        feature_num_pixels = feature_enc.size(1)

        # We'll treat the problem as having a batch size of k
        entire_encoder_out = entire_encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        encoder_crop_out = encoder_crop_out.expand(k, crop_num_pixels, encoder_crop_dim)
        feature_enc = feature_enc.expand(k,feature_num_pixels,encoder_feature_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        # h_att, c_att = decoder.init_hidden_state(encoder_out)
        h_lang,c_lang = decoder.init_hidden_state(entire_encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_crop_out, h_lang)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h_lang))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            # print(embeddings.size(),awe.size())

            att_input=torch.cat([embeddings, awe], dim=1)
            h_att, c_att=decoder.att_lstm(att_input,(h_lang,c_lang))
            # print(h_att.size(),encoder_out.size(),p_conv_feats.size())
            att=decoder.attention1(h_att,entire_encoder_out,feature_enc)
            lang_input= torch.cat([att,h_att],1)

            h_lang, c_lang = decoder.lang_lstm(lang_input,(h_att, c_att))  # (s, decoder_dim)


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
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # print(complete_inds)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h_lang = h_lang[prev_word_inds[incomplete_inds]]
            c_lang = c_lang[prev_word_inds[incomplete_inds]]
            entire_encoder_out = entire_encoder_out[prev_word_inds[incomplete_inds]]
            encoder_crop_out = encoder_crop_out[prev_word_inds[incomplete_inds]]
            feature_enc = feature_enc[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = np.argsort(complete_seqs_scores)
        seq = [complete_seqs[ind] for ind in i]

        # i = complete_seqs_scores.index(max(complete_seqs_scores))
        # seq = complete_seqs[i]


        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        # hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        img_hypos = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                seq))

        # Hypotheses
        hypotheses.append(img_hypos)

        assert len(references) == len(hypotheses)
        # print('references:')
        # print(references)
        # print('hypotheses:')
        # print(hypotheses)
    with open('output_ref.json', 'w') as j:
        json.dump(references, j)
    with open('output_hypo.json', 'w') as j:
        json.dump(hypotheses, j)


    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)#,weights=(1.0, 0, 0, 0))

    return bleu4


if __name__ == '__main__':
    beam_size = 5
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
