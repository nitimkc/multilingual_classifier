# -*- coding: utf-8 -*-
"""
1. Read annotated multilingual ILI data using CustomDataset.
2. Convert encoded features and labels to dataset objects for integration with transformers model training.

""" 

import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset

    
class CustomDataset(object):
    def __init__(self, file_name, savepath=None):
        
        self._file_name = file_name
        if savepath is not None:
            self._savepath = Path(savepath)
            self._savepath.mkdir(parents=True, exist_ok=True)
        
        self.data =  pd.read_csv(self._file_name)
        # self.data =  self.data.convert_dtypes()
        self.tweets = self.data['tweet']
        self.labels = self.data['final_annotation']

    def __len__(self):    
        if len(self.tweets) != len(self.labels):
            raise sys.exit(f"Number of tweets({len(self.tweets)}) and its labels({len(self.labels)}) do not match.")
        else:
            return len(self.labels)
        
    def __getitem__(self, idx):
        tweet = self.tweets.iloc[idx] 
        label = self.labels.iloc[idx] 
        return tweet, label
    
    def getsplitidx(self, test_split=0.2, valid_split=None, group='lang', stratify_label='final_annotation'):
        
        # group day by language and then perform stratified split by categories and save indices as json
        lang_split_idx = {}
        for grp, grp_df in self.data.groupby(group): 
            train, test = train_test_split(grp_df, test_size=test_split, stratify=grp_df[stratify_label])
            if valid_split is not None:
                train, valid = train_test_split(train, test_size=valid_split, stratify=train[stratify_label])
        
            print(f"\nDistribution of classes in train set in {grp}\n{train[stratify_label].value_counts()}")
            print(f"Distribution of classes in test set in {grp}\n\{test[stratify_label].value_counts()}")
            lang_split_idx[grp] = {'train_idx':train.index.values.tolist(), 
                                   'test_idx':test.index.values.tolist()
                                  }
            if valid_split is not None:
                print(f"Distribution of classes in valid set in {grp}\n{valid[stratify_label].value_counts()}\n")
                lang_split_idx[grp] = {'train_idx':train.index.values.tolist(), 
                                       'test_idx':test.index.values.tolist(), 
                                       'valid_idx':valid.index.values.tolist()
                                      }
                
        if self._savepath is not None:
            with open(self._savepath.joinpath("lang_split_idx.json"), "w")  as f:
                json.dump(lang_split_idx, f)
        return lang_split_idx

# https://huggingface.co/transformers/v3.5.1/custom_datasets.html    
class EncodedDataset(torch.utils.data.Dataset): # torch
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
    def __getdatasetsplits__(self, indices):
        split = self.__getitem__(indices)
        dataset = Dataset.from_dict(split)
        return dataset
    
    def splitdata(self, split_indices):
        allsplits = []
        for i in split_indices:
            allsplits.append(self.__getdatasetsplits__(i))
        return allsplits