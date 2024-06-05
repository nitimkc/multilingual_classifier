# 2-*- coding: utf-8 -*-
"""
1. Process data based on model checkpoint and configurations provided in final_configs.json 

"""

from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer
from torch.utils.data import Subset, DataLoader

class DataProcessor(object):
    
    def __init__(self, config, encoder=LabelEncoder(), return_type_ids=False):
        
        self._config = config
        self._encoder = encoder
        self._return_type_ids = return_type_ids
        
        self.model_checkpoint = self._config['MODEL_CHECKPOINT']
        if ((self._config['MAX_LEN'] is not None and self._config['MAX_LEN']>128) and ('bernice' in self.model_checkpoint)):
            self._config['MAX_LEN'] = 128
            print(f"Max length for {self.model_checkpoint} set to 128 by default")
        print(f"Final configurations for processing training + validation data\n{self._config}")

    def label_encoder(self, target):
        le = self._encoder
        return le.fit_transform(target)

    def tokenizer(self):    
        # statistical tokenizer # subwords, chunks of words 
        return AutoTokenizer.from_pretrained(self.model_checkpoint, 
                                             use_fast = False,    # use one of the fast tokenizers (backed by Rust), available for almost all models
                                             # max_length=self._config['MAX_LEN'] # pass max length only when encoding not when instantiating the tokenizer
                                             )
    
    def feature_encoder(self, features):
        tokenizer = self.tokenizer()

        feature_encodings = tokenizer.batch_encode_plus(
            features.astype(str).values.tolist(), 
            padding=True, 
            truncation=True, 
            max_length=self._config['MAX_LEN'],
            # is_split_into_words=True, # added for multilingual versions refer 4624.err
            # return_attention_mask=True,
            return_token_type_ids=self._return_type_ids, 
            return_tensors='pt',
            )
        print(f"Dimensions of encoded features: {feature_encodings['input_ids'].shape}")
        print(f"Encoding contains: {[i for i in feature_encodings.keys()]}")
        return feature_encodings

    def encoded_data(self, features, labels):
        encoded_features = self.feature_encoder(features)
        encoded_labels = self.label_encoder(labels)
        if encoded_features['input_ids'].shape[0] == encoded_labels.shape[0]:
            return encoded_features, encoded_labels
        else:
            print("encoded features and labels do not have same length")