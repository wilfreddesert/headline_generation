import os
from os import path
from functools import reduce
import re
import pandas as pd
from rouge import Rouge
import nltk
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import random
import sys
import tqdm
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu

<<<<<<< HEAD:src/eval_results.py
gold_path = sys.argv[1]
cand_path = sys.argv[2]
if len(sys.argv) > 3:
    text_path = sys.argv[3]
else:
    text_path = None
=======
ITERATION = sys.argv[1]
if len(sys.argv) > 2:
    PREFIX = sys.argv[2]

>>>>>>> 621de5db6d90470f5c1031c4288611005268b411:eval_results.py

def calc_duplicate_n_grams_rate(documents):
    all_ngrams_count = Counter()
    duplicate_ngrams_count = Counter()
    for doc in documents:
        words = doc.split(" ")
        for n in range(1, 5):
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            unique_ngrams = set(ngrams)
            all_ngrams_count[n] += len(ngrams)
            duplicate_ngrams_count[n] += len(ngrams) - len(unique_ngrams)
    return {n: duplicate_ngrams_count[n]/all_ngrams_count[n] if all_ngrams_count[n] else 0.0
            for n in range(1, 5)}


def calc_metrics(refs, hyps, data, metric="all", meteor_jar=None):
    metrics = dict()
    metrics["count"] = len(hyps)
    metrics["text_example"] = data[-1]
    metrics["ref_example"] = refs[-1]
    metrics["hyp_example"] = hyps[-1]
    many_refs = [[r] if r is not list else r for r in refs]
    if metric in ("bleu", "all"):
        metrics["bleu"] = corpus_bleu(many_refs, hyps)
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        metrics.update(scores)
    if metric in ("meteor", "all") and meteor_jar is not None and os.path.exists(meteor_jar):
        meteor = Meteor(meteor_jar, language="ru")
        metrics["meteor"] = meteor.compute_score(hyps, many_refs)
    if metric in ("duplicate_ngrams", "all"):
        metrics["duplicate_ngrams"] = dict()
        metrics["duplicate_ngrams"].update(calc_duplicate_n_grams_rate(hyps))
    return metrics


def print_metrics(refs, hyps, data, metric="all", meteor_jar=None):
    metrics = calc_metrics(refs, hyps, data, metric, meteor_jar)

    print("-------------METRICS-------------")
    print("Count:\t", metrics["count"])
    print("Text:\t", metrics["text_example"])
    print("Ref:\t", metrics["ref_example"])
    print("Hyp:\t", metrics["hyp_example"])

    if "bleu" in metrics:
        print("BLEU:     \t{:3.2f}".format(metrics["bleu"] * 100.0))
    if "rouge-1" in metrics:
<<<<<<< HEAD:src/eval_results.py
        print("ROUGE-1-F:\t{:3.1f}".format(metrics["rouge-1"]['f'] * 100.0))
        print("ROUGE-2-F:\t{:3.1f}".format(metrics["rouge-2"]['f'] * 100.0))
        print("ROUGE-L-F:\t{:3.1f}".format(metrics["rouge-l"]['f'] * 100.0))
        print('ROUGE-mean\t{:3.1f}'.format((metrics["rouge-1"]['f'] + \
                                            metrics["rouge-2"]['f'] + \
                                            metrics["rouge-l"]['f']) * 100.0 / 3))
=======
        print("ROUGE-1-F:\t{:3.2f}".format(metrics["rouge-1"]['f'] * 100.0))
        print("ROUGE-2-F:\t{:3.2f}".format(metrics["rouge-2"]['f'] * 100.0))
        print("ROUGE-L-F:\t{:3.2f}".format(metrics["rouge-l"]['f'] * 100.0))
>>>>>>> 621de5db6d90470f5c1031c4288611005268b411:eval_results.py
    if "meteor" in metrics:
        print("METEOR:   \t{:3.2f}".format(metrics["meteor"] * 100.0))
    if "duplicate_ngrams" in metrics:
        print("Dup 1-grams:\t{:3.2f}".format(metrics["duplicate_ngrams"][1] * 100.0))
        print("Dup 2-grams:\t{:3.2f}".format(metrics["duplicate_ngrams"][2] * 100.0))
        print("Dup 3-grams:\t{:3.2f}".format(metrics["duplicate_ngrams"][3] * 100.0))
<<<<<<< HEAD:src/eval_results.py


with open(gold_path, 'r') as f:
=======



with open('results/' + PREFIX + '.' + ITERATION + '.gold', 'r') as f:
>>>>>>> 621de5db6d90470f5c1031c4288611005268b411:eval_results.py
    gold = f.readlines()

<<<<<<< HEAD:src/eval_results.py
gold  = [' '.join(wordpunct_tokenize(el.strip())) for el in gold]


with open(cand_path, 'r') as f:
    cand = f.readlines()
=======

with open('results/' + PREFIX + '.'  + ITERATION + '.candidate', 'r') as f:
    cand = f.readlines()

cand = [el.strip().lower() for el in cand]


data = pd.read_csv('results/' + PREFIX + '.' + ITERATION + '.raw_src', sep='\n', names=['text'])
data = [el.replace(' ##', '').replace('[CLS]', '').replace('[SEP]', '') for el in data.text.values]

assert(len(data) == len(gold) and len(gold) == len(cand))
>>>>>>> 621de5db6d90470f5c1031c4288611005268b411:eval_results.py

cand = [' '.join(wordpunct_tokenize(el.strip())) for el in cand]

if text_path is None:
    text = ['' for el in range(len(gold))]
else:
    with open(text_path, 'r') as f:
        text = f.readlines()

print_metrics(gold, cand, text)

