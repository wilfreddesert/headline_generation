RAW_RIA_PATH='/data/alolbuhtijarov/dataset/ria/'
JSON_RIA_PATH='/data/alolbuhtijarov/dataset/ria_json/'
BERT_RIA_PATH='/data/alolbuhtijarov/dataset/ria_bert/'

python3 preprocess.py -mode ria_to_lines -raw_path "$RAW_RIA_PATH" -save_path "$JSON_RIA_PATH" -use_bert_basic_tokenizer false;


python3 preprocess.py -mode format_to_bert -raw_path "$JSON_RIA_PATH" -save_path "$BERT_RIA_PATH"  -lower -n_cpus 1 -log_file ../logs/preprocess.log;

mv /data/alolbuhtijarov/dataset/ria_bert/ria.shuffled.train.bert.pt /data/alolbuhtijarov/dataset/ria_bert/train.bert.pt;

mv /data/alolbuhtijarov/dataset/ria_bert/ria.shuffled.test.bert.pt /data/alolbuhtijarov/dataset/ria_bert/test.bert.pt;

python3 train.py -task abs -mode train -bert_data_path "$BERT_RIA_PATH" -visible_gpus 3 -dec_dropout 0.2 -model_path /data/alolbuhtijarov/model/presum -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 8000 -batch_size 8  -train_steps 190000 -dec_layers 4 -dec_heads 6 -report_every 300  -accum_count 35 -use_bert_emb true -use_interval true -warmup_steps_bert 2000  -warmup_steps_dec 1000 -max_pos 512 -log_file ../logs/abs_bert_ria -train_from /data//alolbuhtijarov/model/presum/model_step_88000.pt;


python3 train.py -task abs -mode validate -batch_size 10 -visible_gpus 3 -test_batch_size 10 -bert_data_path "$BERT_RIA_PATH" -log_file ../logs/val_abs_bert_ria -model_path /data/alolbuhtijarov/model/presum -sep_optim true -use_interval true -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../results/
