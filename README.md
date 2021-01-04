
Paper: [Advances of Transformer-Based Models for News Headline Generation](https://arxiv.org/abs/2007.05044)

Most of the code is taken from authors of BertSumAbs: https://github.com/nlpyang/PreSumm

Scripts for mBART and dataset splits are available here: https://github.com/IlyaGusev/summarus

Below are the scripts for BertSumAbs

### Trained models

BertSumAbs checkpoint: https://yadi.sk/d/2jcjmdEXp0EX-Q


### Data Preprocessing

As a pretrained BERT we use RuBERT from DeepPavlov: http://docs.deeppavlov.ai/en/master/features/models/bert.html

```
python convert_to_presumm.py --config-path readers/configs/ria_reader_config.json --file-path ./raw_data/ria_test.json --save-path ./processed_data/output.pt --bert-path ./rubert_cased_L-12_H-768_A-12_pt
```
### Model Training

```
python run.py -mode train -data_path ./processed_data/output.pt -visible_gpus -1 -dec_dropout 0.2 -model_path ./rubert_cased_L-12_H-768_A-12_pt -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 5000 -batch_size 8 -train_steps 200000 -report_every 400 -accum_count 95 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 256 -log_file ../logs/abs_bert_ria
```

### Predicting

```
python run.py -mode test -batch_size 12 -test_from ./rubert_cased_L-12_H-768_A-12_pt/model_step_40000.pt -visible_gpus -1 -test_batch_size 12 -data_path ./processed_data/output.pt -log_file ../logs/val_abs_bert_ria -model_path ./rubert_cased_L-12_H-768_A-12_pt -sep_optim true -use_interval true -max_pos 256 -max_length 90 -alpha 0.95 -min_length 8 -result_path ../check_data
```

### Evaluating

```
python3 src/eval_results.py results/ria_40k.40000.gold results/ria_40k.40000.candidate
```
