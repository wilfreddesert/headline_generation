
Paper: [Advances of Transformer-Based Models for News Headline Generation](https://arxiv.org/abs/2007.05044)


### Dependencies

Note that there are `Pipfile` and `Pipfile.lock` in `src`. 

As long as you have `pipenv` installed, you can just run `pipenv install` and set the environment up.

After that, you can just run `pipenv shell`. 

### RuBERT
Download RuBERT from DeepPavlov: http://docs.deeppavlov.ai/en/master/features/models/bert.html and extract the archive to `src`

### Checkpoint

BertSumAbs checkpoint: https://yadi.sk/d/2jcjmdEXp0EX-Q

If you want to use the model, download the checkpoint using the link above and put this file in the `rubert_cased_L-12_H-768_A-12_pt` folder.

### Config

You can easily change any parameters and hyperparameters by modifying `config.yaml`

### Data Preprocessing

```
python preprocess_input.py --config_path config.yaml
```
### Model Training

```
python run.py --mode train --config_path ./config.yaml  
```

### Predicting

```
python run.py --mode test --config_path ./config.yaml  
```