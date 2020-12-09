import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
# from att_SCAN_model import Encoder, Encoder_feats, DecoderWithAttention
from att_crop_model import Encoder, Encoder_crop, DecoderWithAttention
from SCAN.SCAN_model import EncoderText,ContrastiveLoss
from datasets_crop import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
best_bleu4=0.
def main():
    """
    Training and validation.
    """

    # Hyper Parameters
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--data_folder', default='./caption_data_crop/',
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='thyroid_10_cap_per_img_0_min_word_freq',
                        help=' base name shared by data files')

    # Model parameters
    parser.add_argument('--emb_dim', default=512,
                        help=' dimension of word embeddings')
    parser.add_argument('--attention_dim', default=512,
                        help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', default=512,
                        help=' dimension of decoder RNN')
    parser.add_argument('--dropout', default=0.5,
                        help='dropout')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='sets device for model and PyTorch tensors')

    # Training parameters
    parser.add_argument('--start_epoch', default=0,
                        help='start_epoch')
    parser.add_argument('--epochs', default=120,
                        help=' number of epochs to train for (if early stopping is not triggered)')
    parser.add_argument('--epochs_since_improvement', default=0,
                        help='keeps track of number of epochs since there\'s been an improvement in validation BLEU')
    parser.add_argument('--batch_size', default=16,
                        help='batch_size')
    parser.add_argument('--workers', default=0,
                        help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--encoder_lr', default=1e-4 ,
                        help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', default=4e-4,
                        help='learning rate for decoder')
    parser.add_argument('--grad_clip', default=5,
                        help='clip gradients at an absolute value of')
    parser.add_argument('--alpha_c', default=1,
                        help='regularization parameter for \'doubly stochastic attention\', as in the paper')
    # parser.add_argument('--best_bleu4', default=0,
    #                     help='BLEU-4 score right now')
    parser.add_argument('--print_freq', default=100,
                        help='print training/validation stats every __ batches')
    parser.add_argument('--fine_tune_encoder', default=True,
                        help='fine-tune encoder?')
    parser.add_argument('--checkpoint', default=None,
                        help='path to checkpoint, None if none')

    #loss parameters
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--embed_dim', default=2048,
                        help='Rank loss embedding dimension.')
    parser.add_argument('--num_layers', default=1,
                        help='num layers.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')

    opt = parser.parse_args()
    print(opt)

    global word_map, feature_map, best_bleu4

    # Read word map
    word_map_file = os.path.join(opt.data_folder, 'WORDMAP_' + opt.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    print(len(word_map))

    # Read feature map
    with open(os.path.join(opt.data_folder, 'FEATURE_' + opt.data_name + '.json'), 'r') as j:
        feature_map = json.load(j)

    # Initialize / load checkpoint
    if opt.checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=opt.attention_dim,
                                       embed_dim=opt.emb_dim,
                                       decoder_dim=opt.decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=opt.dropout)
        text_encoder = EncoderText(vocab_size=len(word_map),
                                   word_dim=300,
                                   embed_size=opt.embed_dim,
                                   num_layers=opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=opt.decoder_lr)
        encoder = Encoder()
        encoder_crop = Encoder_crop()
        encoder.fine_tune(opt.fine_tune_encoder)
        # encoder_feats = Encoder_feats()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=opt.encoder_lr) if opt.fine_tune_encoder else None
        encoder_crop_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_crop.parameters()),
                                                  lr=opt.encoder_lr) if opt.fine_tune_encoder else None

    else:
        checkpoint = torch.load(opt.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_crop = checkpoint['encoder_crop']
        text_encoder = checkpoint['text_encoder']
        # encoder_feats = checkpoint['encoder_feats']
        encoder_optimizer = checkpoint['encoder_optimizer']
        encoder_crop_optimizer = checkpoint['encoder_crop_optimizer']
        if opt.fine_tune_encoder is True and opt.encoder_optimizer is None:
            encoder.fine_tune(opt.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=opt.encoder_lr)
            encoder_crop_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_crop.parameters()),
                                                      lr=opt.encoder_lr) if opt.fine_tune_encoder else None

    # Move to GPU, if available
    decoder = decoder.to(opt.device)
    encoder = encoder.to(opt.device)
    encoder_crop = encoder_crop.to(opt.device)
    # encoder_feats = encoder_feats.to(opt.device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(opt.device)
    criterion_similarity = ContrastiveLoss(opt=opt,
                                           margin=opt.margin,
                                           max_violation=opt.max_violation)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(opt.data_folder, opt.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(opt.data_folder, opt.data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)

    # Epochs
    for epoch in range(opt.start_epoch, opt.epochs):
        print(epoch)

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if opt.epochs_since_improvement == 20:
            break
        if opt.epochs_since_improvement > 0 and opt.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if opt.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(opt=opt,
              train_loader=train_loader,
              encoder=encoder,
              encoder_crop=encoder_crop,
              encoder_text=text_encoder,
              decoder=decoder,
              criterion=criterion,
              criterion_similarity = criterion_similarity,
              encoder_optimizer=encoder_optimizer,
              encoder_crop_optimizer=encoder_crop_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(opt=opt,
                                val_loader=val_loader,
                                encoder=encoder,
                                encoder_crop=encoder_crop,
                                encoder_text=text_encoder,
                                decoder=decoder,
                                criterion=criterion,
                                criterion_similarity = criterion_similarity,
                                )

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(opt.data_name, epoch, epochs_since_improvement, encoder, encoder_crop, text_encoder, decoder, encoder_optimizer, encoder_crop_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(opt,train_loader, encoder, encoder_crop, encoder_text, decoder, criterion, criterion_similarity, encoder_optimizer, encoder_crop_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    encoder_crop.train()
    encoder_text.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses1 = AverageMeter()  # loss (per word decoded)
    losses2 = AverageMeter()


    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (img_idx, imgs, crop_imgs, feature, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # print(i)

        # Move to GPU, if available
        imgs_origin = imgs.to(opt.device)
        imgs_crop_origin = crop_imgs.to(opt.device)
        caps = caps.to(opt.device)
        caplens = caplens.to(opt.device)

        # Forward prop.
        imgs = encoder(imgs_origin)
        imgs_crop = encoder_crop(imgs_crop_origin)

        # feature = '100122214132100000001202000'
        lengths=[]
        feature_batch_size=feature.size(0)
        Feat = torch.LongTensor(feature_batch_size, len(feature_map['features']*3)).fill_(0)

        for j in range(feature_batch_size):
            Feature=torch.LongTensor(len(feature_map['features']*3))
            for ii, f in enumerate(feature_map['features']):
                Feature[ii*3]=f
                Feature[ii*3+1]=feature_map[str(f)][int(feature[j,ii])]
                Feature[ii*3+2]=torch.LongTensor([0])
            lengths.append(len(Feature))
            Feat[j]=Feature
        # print(Feat)

        feature_enc, feature_lens = encoder_text(Feat, lengths)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, imgs_crop, feature_enc, caps, caplens)
        #scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, p_feats, caps, caplens)
        #scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        # p_feats=p_feats.type(torch.cuda.FloatTensor)
        # cap_emb=cap_emb.type(torch.cuda.FloatTensor)
        # cap_lens=cap_lens.type(torch.cuda.IntTensor)
        criterion_loss = criterion(scores, targets)
        criterion_similarity_loss = criterion_similarity(imgs_crop, feature_enc, feature_lens).to(device)
        loss = criterion(scores, targets)+criterion_similarity(imgs_crop, feature_enc, feature_lens).to(device)

        # Add doubly stochastic attention regularization
        loss += opt.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
            encoder_crop_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if opt.grad_clip is not None:
            clip_gradient(decoder_optimizer, opt.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, opt.grad_clip)
                clip_gradient(encoder_crop_optimizer, opt.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
            encoder_crop_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses1.update(criterion_loss.item(), sum(decode_lengths))
        losses2.update(criterion_similarity_loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss_sim {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss1=losses1,
                                                                          loss2=losses2,
                                                                          top5=top5accs))


def validate(opt,val_loader, encoder, encoder_crop, encoder_text, decoder, criterion, criterion_similarity):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()
    if encoder_crop is not None:
        encoder_crop.eval()
    if encoder_text is not None:
        encoder_text.eval()

    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (img_idx, imgs, crop_img, feature, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs_origin = imgs.to(opt.device)
            imgs_crop_origin = crop_img.to(opt.device)
            caps = caps.to(opt.device)
            caplens = caplens.to(opt.device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs_origin)
            if encoder_crop is not None:
                imgs_crop = encoder_crop(imgs_crop_origin)
            # p_feats = torch.rand(32, 196, 2048).cuda()

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

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, imgs_crop, feature_enc, caps, caplens)
            # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss1=criterion(scores, targets)
            loss2=criterion_similarity(imgs_crop, feature_enc, feature_lens).to(device)

            # Calculate loss
            loss = criterion(scores, targets)+criterion_similarity(imgs_crop, feature_enc, feature_lens).to(device)

            # Add doubly stochastic attention regularization
            loss += opt.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses1.update(loss1.item(), sum(decode_lengths))
            losses2.update(loss2.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % opt.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Loss_sim {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss1=losses1, loss2 = losses2, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss1.avg:.3f}, {loss2.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss1=losses1,
                loss2=losses2,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
