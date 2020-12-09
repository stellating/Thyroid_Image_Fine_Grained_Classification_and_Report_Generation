import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
from extract_feature import cal_feat

os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

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


class Encoder_feats(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder_feats, self).__init__()

        '''
        # define model
        resnet = torchvision.models.resnet50(pretrained=False)
        fc_features = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_features, 5)
        model_path = 'C:/Users/DELL/Downloads/pytorch-book-master/chapter10-image_caption/mycode_background/result/f3_20200516_191915/3_009.ckpt'

        checkpoint = torch.load(model_path)
        # for key in checkpoint['net_state_dict']:
        #     print(key)
        resnet.load_state_dict(checkpoint['net_state_dict'])
        # print(model)
        # model.eval()
        # img = torch.rand(4, 3, 224, 224)
        # result = model(img)
        # print(result)

        # resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        '''
        self.feat = ['3', '6', '7', '10']
        self.feat_num = [5, 2, 2, 4]

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        #out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = torch.Tensor(images.size(0), len(self.feat), 2048).fill_(0)
        # print(out.size())
        for c,f in enumerate(self.feat):
            f_n = self.feat_num[c]
            feature = cal_feat(images, f, f_n)
            for feat_idx,feat in enumerate(feature):
                out[feat_idx,c]=feat
            # out[:,c,:] = feature
        # print(out.size())
        out = out.view(len(images), 2048, len(self.feat)//2, -1)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # print(out.size())
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


    # def fine_tune(self, fine_tune=False):
    #     """
    #     Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
    #
    #     :param fine_tune: Allow?
    #     """
    #     for p in self.resnet.parameters():
    #         p.requires_grad = False
    #     # If fine-tuning, only fine-tune convolutional blocks 2 through 4
    #     for c in list(self.resnet.children())[5:]:
    #         for p in c.parameters():
    #             p.requires_grad = fine_tune


class Attention1(nn.Module):
    def __init__(self, rnn_size, att_hid_size):
        super(Attention1, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats):
        # print('att_feats:',att_feats.size())  #32,196,2048
        # print('att_hid_size:',self.att_hid_size)  #512
        # print('rnn_size:',self.rnn_size)    #512
        # print('att_feats:',att_feats.numel())

        # The p_att_feats here is already projected
        batch_size = h.size(0)   #31
        #batch_size = att_feats.size(0)
        # print('batch_size:',batch_size)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        # print('att_size:',att_size)
        # print('p_att_feats:',p_att_feats.size())

        att = p_att_feats.contiguous().view(-1, att_size, self.att_hid_size)
        # print('h:',h.size())
        # print('att:',att.size())

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        att = att.type(torch.cuda.FloatTensor)
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        att_feats_ = att_feats.contiguous().view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention2(nn.Module):
    def __init__(self, rnn_size, att_hid_size):
        super(Attention2, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats, mask):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        hAflat = self.alpha_net(dot)  # (batch * att_size) * 1
        hAflat = hAflat.view(-1, att_size)  # batch * att_size
        hAflat.masked_fill_(mask, self.min_value)

        weight = F.softmax(hAflat, dim=1)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # print(encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

        self.att_lstm = nn.LSTMCell(encoder_dim + decoder_dim, decoder_dim)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(decoder_dim * 2, decoder_dim)  # h^1_t, \hat v
        self.attention1 = Attention1(decoder_dim, attention_dim)
        self.attention2 = Attention2(decoder_dim, attention_dim)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        # print(mean_encoder_out.size())

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, p_conv_feats, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_feats_dim = p_conv_feats.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        p_conv_feats = p_conv_feats.view(batch_size,-1,encoder_feats_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # print('caption_lengths_init:',caption_lengths,'batch_size_init:',batch_size)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        # (batch_size, decoder_dim)
        h_lang, c_lang = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            # print('decode_lengths:',decode_lengths)
            # print('t',t)
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h_lang[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h_lang[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
            att_lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding],dim=1)
            h_att, c_att = self.att_lstm(att_lstm_input, (h_lang[:batch_size_t], h_lang[:batch_size_t]))
            att = self.attention1(h_att, encoder_out[:batch_size_t], p_conv_feats[:batch_size_t])
            #att2 = self.attention2(h_att, encoder_out, p_pool_feats, att_mask[:, 1:])
            lang_lstm_input = torch.cat([att, h_att], 1)

            # ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(state[0][1]))
            h_lang, c_lang = self.lang_lstm(lang_lstm_input, (h_att[:batch_size_t], c_att[:batch_size_t]))
            preds = self.fc(self.dropout(h_lang))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

if __name__ == '__main__':
    model = torch.load('entire_model.pth')
    # Remove linear and pool layers (since we're not doing classification)
    modules = list(model.children())
    print(modules)