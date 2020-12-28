import sys
import random
import os
import numpy as np

dir_path = sys.argv[1]
txt_name = sys.argv[2]
vec_name = sys.argv[3]
train_frac = float(sys.argv[4])
val_frac = float(sys.argv[5])


with open(dir_path + "\\" + txt_name, "r", encoding="utf-8") as f_txt, open(
    dir_path + "\\" + vec_name, "rb"
) as f_vec:
    file_size = os.fstat(f_vec.fileno()).st_size
    total = 0
    while True:
        with open(
            dir_path + "/train_texts.jsonl", "a", encoding="utf-8"
        ) as f_train_txt, open(dir_path + "/train_vec.npy", "ab") as f_train_vec, open(
            dir_path + "/test_texts.jsonl", "a", encoding="utf-8"
        ) as f_test_txt, open(
            dir_path + "/test_vec.npy", "ab"
        ) as f_test_vec, open(
            dir_path + "/val_texts.jsonl", "a", encoding="utf-8"
        ) as f_val_txt, open(
            dir_path + "/val_vec.npy", "ab"
        ) as f_val_vec:
            i = 0
            while f_vec.tell() < file_size and i < 10000:
                i += 1
                txt = f_txt.readline()
                vec = np.load(f_vec)
                p = random.random()

                if p < train_frac:
                    f_train_txt.write(txt)
                    np.save(f_train_vec, vec)
                elif p < train_frac + val_frac:
                    f_val_txt.write(txt)
                    np.save(f_val_vec, vec)
                else:
                    f_test_txt.write(txt)
                    np.save(f_test_vec, vec)

            total += i
            print(f"Documents proccessed: {total}")

            if f_vec.tell() >= file_size:
                break
