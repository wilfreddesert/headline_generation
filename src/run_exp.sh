RAW_RIA_PATH='/data/alolbuhtijarov/dataset/ria/'

python3 preprocess.py -mode ria_to_lines -raw_path $(RAW_RIA_PATH) -save_path ../json_ria -use_bert_basic_tokenizer false;

python3 preprocess.py -mode format_to_bert -raw_path ../json_ria/ -save_path ../bert_data  -lower -n_cpus 1 -log_file ../logs/preprocess.log;

python3 train.py -task abs -mode train -bert_data_path ../bert_data/ -dec_dropout 0.2 -visible_gpus 0 -model_path /data/alolbuhtijarov/model/presum -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -log_file ../logs/abs_bert_ria;

