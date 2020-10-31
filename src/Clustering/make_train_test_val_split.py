import sys
import random
import numpy as np

dir_path = sys.argv[1]
txt_name = sys.argv[2]
vec_name = sys.argv[3]
train_frac = float(sys.argv[4])
val_frac = float(sys.argv[5])


# TODO:

with open(dir_path + '/' + txt_name, 'r', encoding='utf-8') as f_txt,\
     open(dir_path + '/' + vec_name, 'rb') as f_vec:
    txt = f_txt.readline()
    vec = np.load(vec)
    
    p = random.random()
    