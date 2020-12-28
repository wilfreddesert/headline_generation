import os

models = ("35000", "40000", "50000", "55000")
print(models)

for name in models:
    path = (
        "/data/alolbuhtijarov/rubert_cased_L-12_H-768_A-12_pt/tg_model_no_mask/model_step_"
        + name
        + ".pt"
    )
    file_res = name + "_no_mask_res"
    os.system("python3 evaluate_clustering.py " + path + " MeanSum > " + file_res)
