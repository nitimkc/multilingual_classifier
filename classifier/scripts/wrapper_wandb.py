# -*- coding: utf-8 -*-
"""
The script for wrapper function to run multilingual_ILI_classification_wandb.py
"""

import time
import shutil
import logging as log
import wandb 

from reader import EncodedDataset
from preprocessor import DataProcessor
from trainer_wandb import ModelTrainer

def getsplit(lang_split_idx, key='train_idx'):
    idx_list = [v[key] for k,v in lang_split_idx.items()]
    # print(len(idx_list))
    idx_list = [i for eachlist in idx_list for i in eachlist]
    return idx_list
    
def mlm_evaluation(lang_split_idx, tweets, params, sweep_config, split_path, dirname, lang_to_train, cache_path, 
                   hyperparams=None, save=False, n_searches=5):

    # get train, valid and test split for selected languages
    train_idx = getsplit(lang_split_idx, key='train_idx')
    valid_idx = getsplit(lang_split_idx, key='valid_idx')
    test_idx = getsplit(lang_split_idx, key='test_idx')
    print(f"Distribution of data in train, validation and test splits: {len(train_idx)}, {len(valid_idx)}, {len(test_idx)}")

    # save test set for evaluation later
    if save:
        test_df = tweets.data.iloc[test_idx]
        test_df.rename_axis('index').to_csv(split_path.joinpath(f"{dirname}_{lang_to_train}.csv"))

    for model_checkpoint in params['MODEL_CHECKPOINT']:
        # print(model_checkpoint)
        model_params = {'MODEL_CHECKPOINT':model_checkpoint, 
                        'MAX_LEN':params['MAX_LEN'],
                        'TARGET_NAMES':params['TARGET_NAMES'],
                       } 
        
        # convert data to encoded features and labels
        start_time = time.time()
        processor = DataProcessor(model_params, return_type_ids=True)
        feature_encodings, label_encodings = processor.encoded_data(tweets.data['tweet'], tweets.data['final_annotation'])

        # obtain encoded train, valid and test data as dataset object
        encoded_dataset = EncodedDataset(feature_encodings, label_encodings)
        train_dataset, valid_dataset, test_dataset = encoded_dataset.splitdata([train_idx, valid_idx, test_idx])
        
        try:     
            # train with hyperparameter provided    
            trainer = ModelTrainer(model_checkpoint, train_dataset, valid_dataset, test_dataset, processor.tokenizer(), 
                                   split_path, cache_path, model_params['TARGET_NAMES'], lang_to_train)
            sweep_id = wandb.sweep(sweep=sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
            wandb.agent(sweep_id, trainer.train_eval, count=params['N_SEARCHES'])
            wandb.finish()  
            print(f"COMPLETED - {trainer.model_name} trained on {lang_to_train} languages")

            # delete wandb folder related to this model checkpoint
            print(f"free space by deleting: {split_path.parent.joinpath('wandb')}")
            shutil.rmtree(split_path.parent.joinpath('wandb'), ignore_errors=True)
            
        except Exception as error:
            print("An error occurred:", error)
            log.exception('Failed')
            pass 
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total execution time to finetune {trainer.model_name} on {lang_to_train} language(s) is {execution_time}\n")