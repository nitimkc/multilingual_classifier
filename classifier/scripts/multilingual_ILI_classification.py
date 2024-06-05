# -*- coding: utf-8 -*-
"""
The script to run model once the best hyperparameters are identified using classification_wandb.py
Update required in final_configs.json 
"""

import time
from pathlib import Path 
import argparse
import re
import json
import logging as log

import pandas as pd
import numpy as np

from reader import CustomDataset
from wrapper import mlm_evaluation, getsplit

# args
parser = argparse.ArgumentParser(description="Twitter ILI infection detection")
parser.add_argument("--data_file", type=str, help="File name including directory where data resides.")
parser.add_argument("--params_file", type=str, help="File where parameters to run the model are provided.")
parser.add_argument("--output_dir", type=str, help="Directory where output are to be stores. One folder per language.")
parser.add_argument("--temp_model_dir", type=str, help="Directory to temporary save models.")
parser.add_argument("--split_index_filename", type=str, help="Json file name that contains split index to split data for each langauge")
parser.add_argument("--language_evaluation", type=str, help="Boolean for whether to obtain evaluation groupbed by language")
parser.add_argument("--language_evaluation_column", type=str, help="Columns to use for language evaluation")
args = parser.parse_args()

DATA_FILE = Path(args.data_file)
PARAMS_FILE = Path(args.params_file)
OUT_PATH = Path(args.output_dir)
MODEL_PATH = Path(args.temp_model_dir)
SPLIT_IDX_FILENAME = args.split_index_filename
LANG_EVAL = args.language_evaluation
COL_TO_EVAL = args.language_evaluation_column

# read data
data_path = DATA_FILE.parent
tweets = CustomDataset(DATA_FILE, data_path)
print(f"Number of tweets in data: {tweets.__len__()}")
print(f"Distribution of classes in all data {tweets.labels.value_counts()}")

# hyperparameters
params = pd.read_csv(PARAMS_FILE, sep='\t')
target_names = np.unique(tweets.labels).tolist()
# params['split'].apply(ast.literal_eval)
print(f"Configuration setup read from {PARAMS_FILE}")   

# where MLMs are cached
cache_path = PARAMS_FILE.parent.joinpath('.cache')
cache_path.mkdir(parents=True, exist_ok=True)
print(f"Cache in {cache_path}") 

SPLITS = params['split'].unique()
for split in SPLITS:
    
    # read data split index
    dirname = f"testset{split.replace(',','_')}"
    split_path = OUT_PATH.joinpath(dirname)
    print(f"Reading data split index from: {split_path}")
    with open(split_path.joinpath(f'{SPLIT_IDX_FILENAME}.json'), 'r') as f:
        split_idx = json.load(f) 
    
    # save results
    translation_type = SPLIT_IDX_FILENAME.split('split_idx_', 1)[-1]
    all_runs = [i.stem for i in split_path.glob("*") if (i.is_dir()) & (f'{translation_type}_predictions' in i.stem)]
    if len(all_runs)==0:
        n_last_run = 0
    else: 
        run_num = [re.search('(\d+)$', i) for i in all_runs]
        n_last_run = max([int(i.group(0)) for i in run_num])
    save_path = split_path.joinpath(f'{translation_type}_predictions_run{n_last_run+1}')
    save_path.mkdir(parents=True, exist_ok=True) 
    print(f"Results saved in {save_path}")

    # for each language in params
    for lang, lang_params in params.groupby('lang'):
        print(f"\nTrain using data from languages: {lang}")
        print(f"\nOriginal or Translations: {translation_type}")
        training_params = lang_params.to_dict(orient='records')

        # if working with translations, all is not required
        if (lang=='all') & (translation_type !='original'):  
            pass

        else:
            # when learning curve analysis is not required
            if 'learningcurve' not in translation_type:
                lang_split_idx = {}
                lang_split_idx[lang] = split_idx[lang]
                lang_eval = LANG_EVAL
                for config in training_params:
                    config['target_names'] = target_names
                    print(config)
                    mlm_evaluation(lang_split_idx, tweets, config, split_path, MODEL_PATH, cache_path, save_path, lang, LANG_EVAL, COL_TO_EVAL)
            else:
                # if learning curve required and language to analyse matches
                if lang==translation_type.split('_')[0]:
                    for each_split, each_splitidx  in split_idx.items():
                        print(f"\nTrain using split index assigned to {each_split} ")
                        for config in training_params:
                            config['target_names'] = target_names
                            print(config)
                            mlm_evaluation(each_splitidx, tweets, config, split_path, MODEL_PATH, cache_path, save_path, lang, LANG_EVAL, COL_TO_EVAL, lc=True)

# # using data encoded on tweets for languages used in training
# target_names = sorted(test_df['final_annotation'].unique().tolist()) # because english lang has only three labels
# if 'learningcurve' in translation_type:
#     train_df, valid_df, test_df = getsplit(lang_split_idx, tweets, split_path.joinpath(f"{dirname}_{lang}.csv"), lc=True)
# else:
#     train_df, valid_df, test_df = getsplit(lang_split_idx, tweets, split_path.joinpath(f"{dirname}_{lang}.csv"))
# training_params = lang_params.to_dict(orient='records')
# for config in training_params:
#     config['target_names'] = target_names
#     # max length is varying
#     mlm_evaluation_encode_perlang(train_df, valid_df, test_df, config, lang, lang_eval, split_path, cache_path)