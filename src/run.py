#!/usr/bin/env python
from __future__ import division

import argparse
import os

import yaml
from recordtype import recordtype

from others.logging import init_logger
from train_abstractive import test_abs, train_abs

model_flags = [
    "hidden_size",
    "ff_size",
    "heads",
    "emb_size",
    "enc_layers",
    "enc_hidden_size",
    "enc_ff_size",
    "dec_layers",
    "dec_hidden_size",
    "dec_ff_size",
    "encoder",
    "ff_actv",
    "use_interval",
]


def str2bool(value):
    if value.lower() in {"yes", "true", "t", "y", "1"}:
        return True
    if value.lower() in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", required=True)

    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["train"]["gpu_ranks"] = [
        int(i) for i in range(len(config["train"]["visible_gpus"].split(",")))
    ]
    config["train"]["world_size"] = len(config["train"]["gpu_ranks"])
    os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["visible_gpus"]

    init_logger(config["train"]["log_file"])
    DEVICE = "cpu" if config["train"]["visible_gpus"] == "-1" else "cuda"
    DEVICE_ID = 0 if DEVICE == "cuda" else -1
    kwargs_train = config["train"]
    kwargs_test = {**kwargs_train, **config["test"]}
    TrainArgs = recordtype("TrainArgs", kwargs_train)
    TestArgs = recordtype("TestArgs", kwargs_test)
    train_args = TrainArgs(**kwargs_train)
    test_args = TestArgs(**kwargs_test)
    if args.mode == "train":
        train_abs(args=train_args, device_id=DEVICE_ID)
    elif args.mode == "test":
        cp = test_args.test_from
        try:
            STEP = int(cp.split(".")[-2].split("_")[-1])
        except ValueError:
            STEP = 0
        test_abs(args=test_args, device_id=DEVICE_ID, pt=cp, step=STEP)
    else:
        raise ValueError("Unknown argument")
