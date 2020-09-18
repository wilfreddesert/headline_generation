from transformers import BertTokenizer
import tqdm
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import csv
import json
from razdel import sentenize
from models.model_builder import AbsSummarizer
import torch
import numpy as np
import pandas as pd
import sys

checkpoint = torch.load(sys.argv[1], map_location=lambda storage, loc: storage)
markup_path = "/data/alolbuhtijarov/datasets/ru_threads_target.tsv"
clustering_data_path = "/data/alolbuhtijarov/datasets/ru_clustering_data.jsonl"
embed_mode = "FirstCLS"


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
    
    output = model.bert(src, segs, mask_src)
    
    if mode == 'FirstCLS':
        return output[0][0]
    elif mode == 'MeanSum':
        return output[0].mean(0)
    else:
        raise Exception('Wrong mode')



def read_markup(file_name):
    with open(file_name, "r") as r:
        reader = csv.reader(r, delimiter='\t', quotechar='"')
        header = next(reader)
        for row in reader:
            assert len(header) == len(row)
            record = dict(zip(header, row))
            yield record


from sklearn.metrics import classification_report

def calc_metrics(gold_markup, url2label, url2record, output_dict=False):
    not_found_count = 0
    for first_url, second_url in list(gold_markup.keys()):
        not_found_in_labels = first_url not in url2label or second_url not in url2label
        not_found_in_records = first_url not in url2record or second_url not in url2record
        if not_found_in_labels or not_found_in_records:
            not_found_count += 1
            gold_markuo.pop((first_url, second_url))

    targets = []
    predictions = []
    for (first_url, second_url), target in gold_markup.items():
        prediction = int(url2label[first_url] == url2label[second_url])
        first = url2record.get(first_url)
        second = url2record.get(second_url)
        targets.append(target)
        predictions.append(prediction)
    return classification_report(targets, predictions, output_dict=output_dict)

def get_quality(dist_threshold, print_result=False):
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=dist_threshold,
                                           linkage="single",
                                           affinity="cosine")

    clustering_model.fit(embeds)
    labels = clustering_model.labels_
    
    id2url = dict()
    for i, (url, _) in enumerate(url2record.items()):
        id2url[i] = url

    url2label = dict()
    for i, label in enumerate(labels):
        url2label[id2url[i]] = label
        
    if print_result:
        print(calc_metrics(markup, url2label, url2record))
        return
    metrics = calc_metrics(markup, url2label, url2record, output_dict=True)
    return metrics['macro avg']['f1-score']


args = lambda a: b

args.model_path = '/data/alolbuhtijarov/rubert_cased_L-12_H-768_A-12_pt'
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


model = AbsSummarizer(args, 'cpu', checkpoint)
model.eval()

markup = defaultdict(dict)
for record in read_markup(markup_path):
    first_url = record["INPUT:first_url"]
    second_url = record["INPUT:second_url"]
    quality = int(record["OUTPUT:quality"] == "OK")
    markup[(first_url, second_url)] = quality


url2record = dict()
filename2url = dict()
with open(clustering_data_path, "r") as r:
    for line in r:
        record = json.loads(line)
        url2record[record["url"]] = record
        filename2url[record["file_name"]] = record["url"]


embeds = np.zeros((len(url2record), 768))

for i, (url, record) in tqdm.tqdm(enumerate(url2record.items()), total=embeds.shape[0]):
    text = record["title"] + ' ' + record["text"]
    text = text.lower().replace('\xa0', ' ')
    embeds[i] = doc2vec(text, model, mode=embed_mode)


domain = np.logspace(-5, 0, 70)
quals = [get_quality(dist) for dist in tqdm.tqdm(domain, total=70)]

closer_domain = np.linspace(domain[np.argmax(quals)-5], domain[np.argmax(quals)+5], 70)
closer_quals = [get_quality(dist) for dist in tqdm.tqdm(closer_domain, total=70)]


best_dist = closer_domain[np.argmax(closer_quals)]

print('Best distance =', best_dist)
get_quality(best_dist, print_result=True)

