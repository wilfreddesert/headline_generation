RAW_RIA_PATH='/data/alolbuhtijarov/dataset/lenta'
BERT_RIA_PATH='/data/alolbuhtijarov/dataset/bert_lenta'
BERT_MODEL_PATH='/data/alolbuhtijarov/model/rubert_cased_L-12_H-768_A-12_pt'
TRAINED_MODEL_PATH='/data/alolbuhtijarov/model/presum'



python3 convert_to_presumm.py --config-path readers/configs/lenta_reader_config.json --file-path "$RAW_RIA_PATH"/lenta-ru-news.val.csv --save-path "$BERT_RIA_PATH"/test.bert.pt --bert-path "$BERT_MODEL_PATH" --max-src-tokens 200 --max-tgt-tokens 100;

# python3 convert_to_presumm.py --config-path readers/configs/ria_reader_config.json --file-path "$RAW_RIA_PATH"/ria.shuffled.train.json --save-path "$BERT_RIA_PATH"/train.bert.pt --bert-path "$BERT_MODEL_PATH" --max-src-tokens 200 --max-tgt-tokens 100;

#python3 train.py -task abs -mode train -bert_data_path "$BERT_RIA_PATH"/ -visible_gpus 5 -dec_dropout 0.2 -model_path "$TRAINED_MODEL_PATH"/ -sep_optim true -lr_bert 0.001 -lr_dec 0.2 -save_checkpoint_steps 4000 -batch_size 12 -train_steps 100000 -report_every 200 -accum_count 70 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 256 -log_file ../logs/abs_bert_ria -train_from /data/alolbuhtijarov/model/presum/model_step_4000.pt;


python3 train.py -task abs -mode validate -batch_size 12 -visible_gpus 6 -test_batch_size 12 -bert_data_path "$BERT_RIA_PATH"/ -log_file ../logs/val_abs_bert_lenta -model_path "$TRAINED_MODEL_PATH"/ -sep_optim true -use_interval true -max_pos 256 -max_length 90 -alpha 0.95 -min_length 8 -result_path ../results/;
