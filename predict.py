import argparse
import functools
import pickle
from pathlib import Path

import torch
import torch.utils.data as tud

from utils import base_util as bu
import data_provider
import tqdm

from ner_bert_glove_crf_pytorch import Bert_CRF
from bert_pytorch.dataset.vocab import WordVocab

torch.manual_seed(2019)
log = bu.get_logger(__name__)

idx_to_tag = None
tag_to_ix = None
params = None
model = None

vc = WordVocab.load_vocab('bert_pytorch/vocab_no.pkl')


def load_model(args):
    global idx_to_tag
    global tag_to_ix
    global params
    global model
    global log

    if model:
        return

    log.info('begin predict')
    with Path(args.model_path).open('rb') as mp:
        if args.gpu_index < 0:
            ml = 'cpu'
        else:
            ml = None
        best_state_dict = torch.load(mp, map_location=ml)
    with Path(args.config_path).open('rb') as mp:
        params, tag_to_ix = pickle.load(mp)
    idx_to_tag = {tag_to_ix[key]: key for key in tag_to_ix}

    if args.gpu_index > -1:
        device = torch.device(f'cuda:{args.gpu_index}')
    else:
        device = torch.device('cpu')

    pdict = torch.load('word2vec/w2v.no.pkl_19', map_location='cpu')
    model = Bert_CRF(tag_to_ix, params, device, pdict['embeddings.weight'])
    model.to(device)
    model.load_state_dict(best_state_dict, strict=False)
    model.eval()


def predict(args, data_in):
    global idx_to_tag
    load_model(args)

    with bu.timer('load data'):
        dataset = data_provider.BBNDatasetCombine(data_in, has_tag=False)

    # tag_to_ix, word_to_ix, device, batch
    collate_fn = functools.partial(data_provider.collect_fn, tag_to_ix,
                                   vc, model.device, rw=True)
    log.warn(f"{'-'*25} process {len(data_in)} {'-'*25}")
    return evaluate(collate_fn, model, dataset, idx_to_tag)


def evaluate(collate_fn, model, dataset_in=None, idx_to_tag=None):
    with torch.no_grad():
        # change batch_size to 1
        batch_size = 1
        dataset_ = dataset_in
        data_loader = tud.DataLoader(dataset_, batch_size, shuffle=False,
                                     collate_fn=collate_fn, drop_last=False)
        ans = []
        finish_size = 0
        # i, w, wi, l, t, _
        for ods, wi, l, t in tqdm.tqdm(data_loader):
            if (finish_size + 1) % 1000 == 0:
                print(f'finish {finish_size + 1}')
            finish_size += batch_size
            score, p = model(wi, l)
            for bi in range(len(p)):
                last_tag = ''
                ans_t = []
                for i in range(len(p[bi])):
                    tn = idx_to_tag[p[bi][i]]

                    # tn = tn if tn == 'o' else tn.split('-')[1]
                    if tn == last_tag:
                        ans_t.append('_' + ods[bi][i])
                    else:
                        ans_t.append('/' + last_tag + '  ')
                        ans_t.append(ods[bi][i])
                    last_tag = tn
                ans_t.append('/' + last_tag)
                # print(ans_t)
                ans.append(''.join(ans_t[1:]))
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="input dir or file",
                        type=str, required=True)
    parser.add_argument('--output', help="input dir or file",
                        type=str, required=False, default='result/out81_io_w2v19.txt')
    parser.add_argument('--gpu_index', help="gpu index must>-1,if use gpu",
                        type=int, default=0)
    parser.add_argument('--config_path', help="the model path",
                        type=str,
                        default='./result/model/81_io_w2v19_config.pkl')
    parser.add_argument('--model_path', help="the config model path",
                        type=str,
                        default='./result/model/81_io_w2v19_torch.pkl')
    args = parser.parse_args()
    ans = predict(args, args.input)
    with open(args.output, 'w', encoding='utf-8') as f:
        for a in ans:
            f.write(a + '\n')


if __name__ == '__main__':
    main()
