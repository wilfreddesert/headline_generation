from transformers import BertTokenizer
from razdel import sentenize
from models.model_builder import AbsSummarizer
import torch
import numpy as np
import pandas as pd
import tqdm
import json
import pickle
import os


DEVICE = 'cuda'
CHECKPOINT_PATH = r'C:\Users\leshanbog\Documents\model\model_step_15000.pt'
MODEL_PATH = r'C:\Users\leshanbog\Documents\model\bert\rubert_cased_L-12_H-768_A-12_pt'
DATASET_PATH = r'C:\Users\leshanbog\Documents\dataset\ru_tg_1101_0510.jsonl'
CHUNK_SIZE = 1024

class BertData:
    def __init__(self, bert_model, lower, max_src_tokens, max_tgt_tokens):
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=lower, do_basic_tokenize=False)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused1] '
        self.tgt_eos = ' [unused2]'
        self.tgt_sent_split = ' [unused3] '
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt):
        src_txt = [' '.join(s) for s in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_tokens = self.tokenizer.tokenize(text)[:self.max_src_tokens]
        src_tokens.insert(0, self.cls_token)
        src_tokens.append(self.sep_token)
        src_indices = self.tokenizer.convert_tokens_to_ids(src_tokens)

        _segs = [-1] + [i for i, t in enumerate(src_indices) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        return src_indices, segments_ids
    
    
def doc2bert(text):
    src = [s.text.lower().split() for s in sentenize(text)]
    src_indices, segments_ids = bert_data.preprocess(src, '')
    return { "src": src_indices, "segs": segments_ids }

def doc2vec(text, model, mode='MeanSum'):
    doc_bert = doc2bert(text)
    
    src = torch.tensor([doc_bert['src']])
    segs = torch.tensor([doc_bert['segs']])
    mask_src = ~(src == 0)
    
    output = model.bert(src.to(DEVICE), segs.to(DEVICE), mask_src.to(DEVICE))
    
    if mode == 'FirstCLS':
        return output[0][0]
    elif mode == 'MeanSum':
        return output[0].mean(0)
    else:
        raise Exception('Wrong mode')
        
        
checkpoint = torch.load(CHECKPOINT_PATH,
                        map_location=lambda storage, loc: storage)


args = lambda a: b

args.model_path = MODEL_PATH
args.large = False
args.temp_dir = 'temp'
args.finetune_bert = False
args.encoder = 'bert'
args.max_pos = 256
args.dec_layers = 6
args.share_emb = False
args.dec_hidden_size = 768
args.dec_heads = 8
args.dec_ff_size = 2048
args.dec_dropout = 0.2
args.use_bert_emb = False

bert_data = BertData(args.model_path, True, 510, 128)

BertSumAbs = AbsSummarizer(args, DEVICE, checkpoint)
BertSumAbs.eval()


data = pd.read_json(DATASET_PATH, encoding='utf-8', lines=True, chunksize=CHUNK_SIZE)


for el in tqdm.tqdm(data, total=450000 // CHUNK_SIZE):
    with open('vectors.npy', 'ab') as fvecs, open('text.jsonl', 'a', encoding='utf-8') as ft:
        for j in range(CHINK_SIZE):
            text = el.iloc[j]["text"].lower().replace('\xa0', ' ').replace('\n', ' ').strip()
            title = el.iloc[j]["title"].lower()

            if not text or not title or text.count(' ') < 8 or title.count(' ') < 3:
                continue

            ft.write(json.dumps({"text": text, "title": title}) + "\n")

            vec = doc2vec(text + " " + title, BertSumAbs, mode="FirstCLS")
            vec = vec.cpu().detach().numpy()
            np.save(fvecs, vec)