
Most of the code is taken from authors of BertSumAbs: https://github.com/nlpyang/PreSumm


Scripts for mBART and dataset splits are available here: https://github.com/IlyaGusev/summarus

Below are the scripts for BertSumAbs

### Trained models

BertSumAbs checkpoint: https://yadi.sk/d/8TNDcukhEFeekQ


### Data Preprocessing

As a pretrained BERT we use RuBERT from DeepPavlov: http://docs.deeppavlov.ai/en/master/features/models/bert.html

```
python3 convert_to_presumm.py --config-path readers/configs/ria_reader_config.json --file-path ~/dataset/ria/ria.shuffled.train.json --save-path ~/dataset/ria_bert/train.bert.pt --bert-path ~/models/rubert_cased_L-12_H-768_A-12_pt

python3 convert_to_presumm.py --config-path readers/configs/ria_reader_config.json --file-path ~/dataset/ria/ria.shuffled.val.json --save-path ~/dataset/ria_bert/test.bert.pt --bert-path ~/models/rubert_cased_L-12_H-768_A-12_pt
```

### Model Training

```
python3 train.py -task abs -mode train -bert_data_path ~/dataset/ria_bert/ -visible_gpus 0 -dec_dropout 0.2 -model_path ~/models/rubert_cased_L-12_H-768_A-12_pt -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 5000 -batch_size 8 -train_steps 200000 -report_every 400 -accum_count 95 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 256 -log_file ../logs/abs_bert_ria
```

### Predicting

```
python3 train.py -task abs -mode validate -batch_size 12 -visible_gpus 0 -test_batch_size 12 -bert_data_path ~/dataset/ria_bert/ -log_file ../logs/val_abs_bert_ria -model_path ~/models/rubert_cased_L-12_H-768_A-12_pt/ -sep_optim true -use_interval true -max_pos 256 -max_length 90 -alpha 0.95 -min_length 8 -result_path ../results/
```

### Evaluating

```
python3 src/eval_results.py results/ria_40k.40000.gold results/ria_40k.40000.candidate
```
