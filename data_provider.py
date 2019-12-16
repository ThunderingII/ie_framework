import re
import torch
import numpy as np
import torch.utils.data as tud

from utils import base_util as bu


class BBNDatasetCombine(tud.Dataset):
    def __init__(self, file, ss=5, il={0, 1, 2, 3, 4}, has_tag=True, tt='bio'):
        self.data = []
        with bu.timer('load data'):
            with open(file) as f:
                si = 0
                for line in f:
                    line = line.strip()
                    if ss and si % ss not in il:
                        si += 1
                        continue
                    tags = []
                    tokens = []
                    if has_tag:
                        sts = re.split('\s+', line)
                        for st in sts:
                            if st[-1] in {'a', 'b', 'c'}:
                                ts = st[:-2].split('_')
                                if tt == 'bio':
                                    tags.extend(
                                        ['B-' + st[-1]] + ['I-' + st[-1]] * (
                                                len(ts) - 1))
                                else:
                                    tags.extend([st[-1]] * len(ts))
                            else:
                                if '/' in st:
                                    st = st.strip()[:-2]
                                ts = st.split('_')
                                tags.extend(['o'] * len(ts))
                            tokens.extend(ts)
                    else:
                        tokens = line.split('_')
                        tags = ['o'] * len(tokens)
                    self.data.append((tokens, tags, len(tokens)))
                    si += 1

    def __getitem__(self, index):
        # tokens, tags, len(tokens)
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collect_fn(tag_to_ix, vc, device, batch, rw=False, MAX_LEN=512):
    """
    Used to process data when generate a batch
    :param model: model to load embedding
    :param bert_dim: bert dim
    :param tag_to_ix: map, tag to id
    :param word_to_ix: map, word to id
    :param rw: if true return tokens
    :param batch: input data
    :return:
    origin_tags, words_batch, bert_ids_batch,
    len_w_batch, tags_batch,sentences_batch
    """
    batch_size = len(batch)
    len_w_batch = np.zeros((batch_size,), dtype=np.int64)
    size = 1
    max_len = 0
    for i in range(batch_size):
        len_w = batch[i][2] if batch[i][2] <= MAX_LEN else MAX_LEN
        max_len = max(len_w, max_len)
        len_w_batch[i] = len_w

    # sort the array by length, sort the len_w by length
    # ([10,2,5],[0,1,2]) after sort ([10,5,2],[0,2,1])
    rz = zip(len_w_batch, range(len(len_w_batch)))
    r = sorted(rz, key=lambda item: item[0], reverse=True)
    _, index_list = zip(*r)

    origin_data = [None] * batch_size
    words_batch = np.zeros((batch_size, max_len), dtype=np.int64)
    tags_batch = np.zeros((batch_size, max_len), dtype=np.int64)

    for ni, bi in enumerate(index_list):
        # tokens, tags, len(tokens)
        words, tags, len_w = batch[bi]
        len_w = min(len_w, MAX_LEN)
        for i in range(len_w):
            words_batch[ni][i] = vc.stoi[words[i]] if words[i] in vc.stoi \
                else vc.unk_index
            tags_batch[ni][i] = tag_to_ix[tags[i]]
        size += len_w
        if rw:
            origin_data[ni] = words
        else:
            origin_data[ni] = tags
        len_w_batch[ni] = len_w

    ods, w, l, t = (origin_data, words_batch, len_w_batch, tags_batch)

    # get gpu tensor
    return ods, torch.from_numpy(w).to(device), torch.from_numpy(l).to(
        device), torch.from_numpy(t).to(device)
