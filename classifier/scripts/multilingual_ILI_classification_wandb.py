# -*- coding: utf-8 -*-
"""
The script to run model once the best hyperparameters are identified using classification_wandb.py
Update required in final_configs.json 
"""

import os
import time
from pathlib import Path
import argparse
import json
import yaml
import logging as log

from dotenv import load_dotenv
# import pprint

import numpy as np
import torch

from reader import CustomDataset 
from wrapper_wandb import mlm_evaluation

# args
parser = argparse.ArgumentParser(description="Twitter ILI infection detection")
parser.add_argument("--data_file", type=str, help="File name including directory where data resides.")
parser.add_argument("--params_file", type=str, help="File where parameters to run the model are provided.")
parser.add_argument("--config_file", type=str, help="File where sweep config for wandb are provided.")
parser.add_argument("--output_dir", type=str, help="Directory where output are to be stored")
parser.add_argument("--wandb_tmpdir", type=str, help="Directory where wandb data and artifact versions are to be stored.")
args = parser.parse_args()

DATA_FILE = Path(args.data_file)
PARAMS_FILE = Path(args.params_file)
CONFIG_FILE = Path(args.config_file)
OUT_PATH = Path(args.output_dir)
WANDB_DATA_PATH = Path(args.wandb_tmpdir)
# LANG = args.language

load_dotenv()

# ensure outpath exists if not create
OUT_PATH.mkdir(parents=True, exist_ok=True)

WANDB_DATA_PATH = str(WANDB_DATA_PATH.joinpath(OUT_PATH.stem))
os.environ["WANDB_DIR"] = os.environ["WANDB_CACHE_DIR"] = os.environ["WANDB_CONFIG_DIR"] = str(OUT_PATH)
os.environ["WANDB_ARTIFACT_DIR"] = os.environ["WANDB_ARTIFACT_LOCATION"] = os.environ["WANDB_DATA_DIR"] = WANDB_DATA_PATH 
print(f"wandb artifacts and data saved in \n{WANDB_DATA_PATH } \nand rest in {OUT_PATH}\n")

# specified in trainer.py
# os.environ["TRANSFORMERS_CACHE"] = str(OUT_PATH.joinpath(".cache"))
# os.environ["HF_DATASETS_CACHE"] = str(OUT_PATH.joinpath(".cache"))
# os.environ['HF_HOME'] = str(OUT_PATH.joinpath(".cache"))
# os.environ["WANDB_CACHE_DIR"] = str(OUT_PATH.joinpath(".cache"))

# read data and params
data_path = DATA_FILE.parent
tweets = CustomDataset(DATA_FILE, data_path)
print(f"Number of tweets in data: {tweets.__len__()}")
print(f"Distribution of classes in all data {tweets.labels.value_counts()}")

with open (PARAMS_FILE, "r") as f:
    params = json.load(f)
params['MAX_LEN'] = None if params['MAX_LEN']=='None' else params['MAX_LEN']
params['TARGET_NAMES'] = np.unique(tweets.labels).tolist()
print(f"Configuration setup: {params}")   

with open (CONFIG_FILE, "r") as f:
    sweep_config = yaml.safe_load(f)
print(f"Configuration setup for gridsearch: {sweep_config['parameters']}")

for split in params['SPLITS']:

    # cache dir
    cache_path = PARAMS_FILE.parent.joinpath('.cache')
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Cache in {cache_path}")

    # data split index info
    dirname = f"testset{'_'.join([str(i) for i in split])}"
    split_path = OUT_PATH.joinpath(dirname)
    print(f"Reading data split index from: {split_path}\n")
    with open(split_path.joinpath('split_idx.json'), 'r') as f:
        split_idx = json.load(f) 
        
    # determine languages for which to get split index
    if params['LANG']=='all':
        languages = [i for i in split_idx]
    else:
        languages = [i for i in split_idx if i in params['LANG'].split(',')]
    
    # train on all languages and then on each language
    lang_split_idx = {i:split_idx[i] for i in languages}
    print(f"Training data used for {params['LANG']} languages")
    mlm_evaluation(lang_split_idx, tweets, params, sweep_config, split_path, dirname, params['LANG'], cache_path)
    
    for lang_to_train in languages:
        #if lang_to_train=='fr' or lang_to_train=='it':
        print(f"\nTraining data used for {lang_to_train} language")
        lang_split_idx = {}
        lang_split_idx[lang_to_train] = split_idx[lang_to_train]
        mlm_evaluation(lang_split_idx, tweets, params, sweep_config, split_path, dirname, lang_to_train, cache_path)