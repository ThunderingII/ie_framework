import numpy as np
import torch

from torch import nn

import model.crf as crf
import bert_pytorch.model.bert as bert
import model.transformer as transformer
from config.config_cls import NerModelConfig

torch.manual_seed(2019)


# gelu activation function
def gelu(x):
    return x * torch.sigmoid(1.702 * x)


activate_fn_map = {'gelu': gelu, 'sigmoid': nn.functional.sigmoid,
                   'relu': nn.functional.relu, }


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, activate_fn='gelu', bias=True):
        super(Linear, self).__init__()
        self.activate_fn = activate_fn
        self.linear = nn.Linear(dim_in, dim_out, bias)

    def forward(self, feats):
        if self.activate_fn and self.activate_fn in activate_fn_map:
            return activate_fn_map[self.activate_fn](self.linear(feats))
        else:
            return self.linear(feats)


class ConvBlock(nn.Module):
    def __init__(self, in_s, out_s, dil=1, win=3):
        super(ConvBlock, self).__init__()
        pad = dil * (win - 1) // 2
        self.conv1 = nn.Conv1d(in_s, out_s, win, 1, pad, dil)
        self.conv2 = nn.Conv1d(in_s, out_s, win, 1, pad, dil)
        self.use_x = in_s == out_s

    def forward(self, x_in):
        """
        :param x_in: [N, C, L]
        :return: [batch_size, embed_size, seq_len]
        """
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        sigma = torch.sigmoid(x2)
        x_conv = x1 * sigma
        if self.use_x:
            x_conv += x_in
        return x_conv


class Ner(nn.Module):
    def __init__(self, config: NerModelConfig):
        super().__init__()
        self.em = EmbeddingLayer(config)
        self.fe = FeatureExtraction(config)
        self.tg = TaggingLayer(config)

    def forward(self, x, tags, segment_info=None, *x_o):
        embedding, mask = self.em(x, segment_info, *x_o)
        feats, mask = self.fe(embedding, mask)
        out = self.tg(feats, mask, tags)
        return out

    def predict(self, x):
        embedding, mask = self.em(x)
        feats, mask = self.fe(embedding, mask)
        return self.tg.predict(feats, mask)


class FeatureExtraction(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.mode_type = param['mode_type']
        self.dropout = nn.Dropout(param['dropout'])
        self.hidden_dim = param['hidden_size']
        self.e2f = Linear(param['e_size'], self.hidden_dim)
        batch_size = param['batch_size']
        if self.mode_type == 'b':
            self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim // 2,
                                num_layers=1, bidirectional=True)
            self.lstm_hidden = [
                torch.randn(2, batch_size, self.hidden_dim // 2),
                torch.randn(2, batch_size, self.hidden_dim // 2)]
        elif self.mode_type == 't':
            self.te = nn.Sequential(
                [transformer.TransformerEncode(self.hidden_dim,
                                               param['head_num'],
                                               param['tf_dropout']) for _ in
                 range(param['tf_size'])])
        elif self.mode_type == 'cnn':
            self.conv_1 = ConvBlock(self.hidden_dim, self.hidden_dim)
            self.conv_2 = ConvBlock(self.hidden_dim, self.hidden_dim, dil=2)
            self.conv_4 = ConvBlock(self.hidden_dim, self.hidden_dim, dil=4)
        self.hidden2tag = Linear(param['hidden_size'], param['num_tags'],
                                 activate_fn=None)

    def forward(self, embedding, mask):
        embedding = self.dropout(self.e2f(embedding))
        if self.mode_type == 'c':
            # only use crf
            outputs = embedding.transpose(0, 1)
            mask = mask.transpose(0, 1)
        elif self.mode_type == 'b':
            # after transpose seq_len,batch_size * embed_size
            embedding = embedding.transpose(0, 1)
            # use bi-lstm
            len_w = torch.sum(mask, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, len_w)
            # input of shape (seq_len, batch, input_size),
            # output size: seq_len, batch, num_directions * hidden_size
            if self.lstm_hidden[0].device != embedding.device:
                self.lstm_hidden[0] = self.lstm_hidden[0].to(embedding.device)
                self.lstm_hidden[1] = self.lstm_hidden[1].to(embedding.device)
            lstm_out, self.hidden = self.lstm(packed, self.lstm_hidden)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            mask = mask.transpose(0, 1)
        elif self.mode_type == 't':
            # use transformer
            tf_output = self.te(embedding, mask)
            # transpose to seq_len * batch_size * embed_size
            outputs = tf_output.transpose(0, 1)
        elif self.mode_type == 'cnn':
            # [N,L,C] change to [batch_size, embed_size, seq_len]
            outputs = self.conv_1(embedding.transpose(1, 2))
            outputs = self.conv_2(outputs)
            # change to [seq_len, batch_size, embed_size]
            outputs = self.conv_4(outputs).permute(2, 0, 1)
            mask = mask.transpose(0, 1)

        # after liner layer: seq_len, batch, tag_size
        return self.hidden2tag(self.dropout(outputs)), mask


class EmbeddingLayer(nn.Module):
    def __init__(self, config: NerModelConfig):
        super().__init__()
        self.config = config
        if config.em_type in ['bert', 'mix']:
            # self.bert = bert.BERT(param['vocab_size'], param['bert_size'],
            #                       param['n_layers'], param['attn_heads'],
            #                       param['dropout'])
            if config.freeze_embedding_bert:
                for m in self.bert.modules():
                    for p in m.parameters():
                        p.requires_grad = False
        if config.em_type in ['glove', 'mix']:
            self.em = nn.Embedding(config.vocab_size,
                                   config.glove_size)
        self.of = nn.Embedding(config.vocab_size,
                               config.feature_size)

    def load_embedding(self, weight, is_of=False, freeze=True):
        if not is_of and self.em_type in ['pem', 'mix']:
            self.em.from_pretrained(weight, freeze)
        if is_of:
            self.of.from_pretrained(weight, freeze)

    def forward(self, x, segment_info=None, *x_o):
        mask = (x > 0)
        em_r = None
        if self.config.em_type == 'glove':
            em_r = self.em(x)
        elif self.config.em_type == 'bert':
            segment_info = segment_info if segment_info else torch.ones(
                x.size(), dtype=torch.long).to(x.device)
            em_r = self.bert(x, segment_info)
        elif self.config.em_type == 'mix':
            if self.config.mix_type == 'add':
                segment_info = segment_info if segment_info else torch.ones(
                    x.size(), dtype=torch.long).to(x.device)
                em_r = self.bert(x, segment_info) + self.em(x)
            else:
                em_r = torch.cat((self.bert(x, segment_info), self.em(x)), -1)
        em_r = torch.cat((em_r, self.of(x)), -1)
        return em_r if len(x_o) == 0 else torch.cat((em_r, *x_o), -1), mask


class TaggingLayer(nn.Module):
    def __init__(self, config: NerModelConfig):
        super().__init__()
        self.config = config
        if config.labelling_type in ['crf', 'lc']:
            self.crf = crf.CRF(config.num_tags, False)
        elif config.labelling_type == 'ce':
            self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feats, mask, tags):
        if self.config.labelling_type == 'c':
            return self.crf.crf_log_loss(feats, tags.transpose(0, 1), mask)
        elif self.config.labelling_type == 'lc':
            return self.crf.get_labelwise_loss(feats, tags.transpose(0, 1),
                                               mask)
        elif self.config.labelling_type == 'ce':
            seq_len, batch_size, _ = feats.size()
            # change to [batch_size, seq_len, embed_size]
            feats, mask_x = feats.transpose(0, 1), mask.transpose(0, 1)
            feats = feats.contiguous().view(-1, self.config.num_tags)
            tags = tags.contiguous().view(-1)
            loss = self.cross_entropy(feats, tags).view(batch_size,
                                                        seq_len) * mask_x.float()
            return torch.sum(loss, -1) / mask.float().sum()

    def predict(self, feats, mask):
        if self.config.labelling_type in ['c', 'lc']:
            return self.crf(feats, mask)
        elif self.config.labelling_type == 'ce':
            # change to [batch_size, seq_len, embed_size]
            len_w = mask.transpose(0, 1).int().sum()
            feats = feats.transpose(0, 1)
            res = []
            feats_np = torch.argmax(feats, -1).cpu().numpy()
            for i, f_bs in enumerate(feats_np):
                res.append(f_bs[:len_w[i]])
            return np.zeros(feats.size()[0]), res
