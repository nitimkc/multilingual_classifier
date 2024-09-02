# -*- coding: utf-8 -*-
"""
The script for wrapper function to run multilingual_ILI_classification.py
"""

import time
import re
import sys
import shutil
import logging as log

import pandas as pd

from reader import EncodedDataset
from preprocessor import DataProcessor
from trainer import hyperparams, ModelTrainer
from evaluater import PredictionEvaluater

def getsplitidx(lang_split_idx, key='train_idx'):
    idx_list = [v[key] for k,v in lang_split_idx.items()]
    idx_list = [i for eachlist in idx_list for i in eachlist]
    return idx_list

def getsplit(lang_split_idx, tweets, savepath, save=False, lc=False):
    # used for encode using each language only
    
    # get train, valid and test split for selected languages
    if lc:
        train_idx = lang_split_idx['train_idx']
        valid_idx = lang_split_idx['valid_idx']
        test_idx = lang_split_idx['test_idx']
    else:
        train_idx = getsplitidx(lang_split_idx, key='train_idx')
        valid_idx = getsplitidx(lang_split_idx, key='valid_idx')
        test_idx = getsplitidx(lang_split_idx, key='test_idx')
    print(f"Distribution of data in train, validation and test splits: {len(train_idx)}, {len(valid_idx)}, {len(test_idx)}")
    
    # save test set for evaluation later
    train_df = tweets.data.iloc[train_idx]
    valid_df = tweets.data.iloc[valid_idx]
    test_df = tweets.data.iloc[test_idx]
    if save:
        test_df.rename_axis('index').to_csv(savepath)
    return train_df, valid_df, test_df

def mlm_evaluation(lang_split_idx, tweets, config, split_path, model_path, cache_path, save_path, 
                   lang, lang_eval, col_to_eval, num_seed, hyperparams=hyperparams, save=False, lc=False):

    # obtain datasplit index
    if lc:
        train_idx = lang_split_idx['train_idx']
        valid_idx = lang_split_idx['valid_idx']
        test_idx = lang_split_idx['test_idx']
    else:    
        train_idx = getsplitidx(lang_split_idx, key='train_idx')
        valid_idx = getsplitidx(lang_split_idx, key='valid_idx')
        test_idx = getsplitidx(lang_split_idx, key='test_idx')
    print(f"\nDistribution of data in train, validation and test splits: {len(train_idx)}, {len(valid_idx)}, {len(test_idx)}")

    test_df = tweets.data.iloc[test_idx]
    if save:
        test_df.rename_axis('index').to_csv(split_path.joinpath(f"{split_path.stem}_{lang}.csv"))

    # ensure all parameters for training exists
    CONFIG = {k.upper():v for k,v in config.items()}
    if not all(k in CONFIG.keys() for k in hyperparams):
        sys.exit(f"provide all required hyperparams: {hyperparams}. Received only {CONFIG.keys()}")
    
    start_time = time.time()
    try:
        # encode the data using the model checkpoint
        print(f"Working with {CONFIG['MODEL_CHECKPOINT']}")
        processor = DataProcessor(CONFIG, return_type_ids=True)
        
        # encode data using all languages 
        # for encoding data using that language only >>> refer to mlm_evaluation_encode_perlang function below 
        feature_encodings, label_encodings = processor.encoded_data(tweets.data['tweet'], tweets.data['final_annotation'])
    
        # obtain encoded train, valid and test data as dataset object
        encoded_dataset = EncodedDataset(feature_encodings, label_encodings)
        train_dataset, valid_dataset, test_dataset = encoded_dataset.splitdata([train_idx, valid_idx, test_idx])
        print(f"Distribution of data splits for {lang} language is {train_dataset.shape}, {valid_dataset.shape}, {test_dataset.shape}")

        trainer = ModelTrainer(CONFIG, train_dataset, valid_dataset, test_dataset, processor.tokenizer(), model_path, cache_path, num_seed, lang)
        model, prediction_set = trainer.train_eval(get_pred=True)
        print(f"\n{trainer.model_name} trained on {trainer._lang_to_train} languages")
        
        # delete wandb and model folder related to this model checkpoint
        print(f"free space by deleting: {cache_path.parent.joinpath('models')}")
        shutil.rmtree(cache_path.parent.joinpath('models'), ignore_errors=True)
            
        # evaluate on test set and save
        evaluater = PredictionEvaluater(prediction_set, CONFIG['TARGET_NAMES'], save_path, f"{trainer.model_name}_{lang}")
        evaluater.evaluation_report(test_df, lang, lang_eval, col_to_eval)
     
    except Exception as error:
        print("An error occurred:", error)
        log.exception('Failed')
        pass 
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time to finetune {trainer.model_name} on {trainer._lang_to_train} language(s) is {execution_time}")

# def mlm_evaluation_encode_perlang(train_df, valid_df, test_df, config, lang, lang_eval, col_to_eval,
#                                   split_path, model_path, cache_path, save_path, hyperparams=hyperparams, 
#                                   save=False):
#     # in this case the dimension of the encoded data varies on max length 
#     # the encoding is done using data from that language only oopose to all languages above

#     # ensure all parameters for trianing exists
#     CONFIG = {k.upper():v for k,v in config.items()}
#     if not all(k in CONFIG.keys() for k in hyperparams):
#         sys.exit(f"provide all required hyperparams: {hyperparams}. Received only {CONFIG.keys()}")

#     # combine data for preprocessing
#     alldata = pd.concat([train_df, valid_df, test_df])
#     original_index = alldata.index
#     alldata.reset_index(inplace=True)
    
#     start_time = time.time()
#     try:
#         # process data into encoded features and labels and then to dataset object
#         print(f"\nWorking with {CONFIG['MODEL_CHECKPOINT']}")
#         processor = DataProcessor(CONFIG, return_type_ids=True)
#         feature_encodings, label_encodings = processor.encoded_data(alldata['tweet'], alldata['final_annotation'])
#         encoded_dataset = EncodedDataset(feature_encodings, label_encodings)
    
#         # for now unable to use hugging face trainer without validation set
#         # refer to Training arguments and then change if required to train on train+valid data
        
#         # train with data splits and configurations provided
#         n, n_train, n_valid, n_test = alldata.shape[0], train_df.shape[0], valid_df.shape[0], test_df.shape[0]
#         train_dataset, valid_dataset, test_dataset = encoded_dataset.splitdata([range(n_train), 
#                                                                                 range(n_train, n_train+n_valid), 
#                                                                                 range(n_train+n_valid, n)
#                                                                                ])
#         print(f"Distribution of data splits for {lang} language is {train_dataset.shape}, {valid_dataset.shape}, {test_dataset.shape}")
#         trainer = ModelTrainer(CONFIG, train_dataset, valid_dataset, test_dataset, processor.tokenizer(), split_path, model_path, cache_path, lang)
#         model, prediction_set = trainer.train_eval(get_pred=True)
#         print(f"\n{trainer.model_name} trained on {trainer._lang_to_train} languages")
            
#         # evaluate on test set
#         evaluater = PredictionEvaluater(prediction_set, CONFIG['TARGET_NAMES'], split_path, f"{trainer.model_name}_{lang}")
#         evaluater.evaluation_report(test_df, lang_eval, col_to_eval)
     
#     except Exception as error:
#         print("An error occurred:", error)
#         log.exception('Failed')
#         pass 
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"\nTotal execution time to finetune {CONFIG['MODEL_CHECKPOINT']} on {lang} language(s) is {execution_time}")    