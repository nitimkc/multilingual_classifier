## 4-*- coding: utf-8 -*-
"""
function to translate one tweet from souce language to target using a translator and save
"""

import json
from pathlib import Path
import time

import pandas as pd
import deepl

def translate_to_one(df, source, target, languages, translator, savepath):
    
    # test=False
    # if test:
    # df = df.iloc[:2,:]
    counter = 0
    try:
        for idx, tweet in df["tweet"].iteritems():
            if type(tweet) == str:
                # translated = "text"
                translated = translator.translate_text(tweet, source_lang=source, target_lang=languages[target], tag_handling=None) 
            else: 
                translated = "unable to translate"
            translated_dict = {"id":df["id"][idx], "translated_tweet":translated}
            with open(savepath.joinpath(f"{source}_translatedto_{target}.jsonl"), 'a', encoding="utf-8") as file:
                json.dump(translated_dict, file, default=str)
                file.write('\n')
            counter +=1
    except Exception as excep:
        print(f"Unable to translate as of tweet id: {df['id'][idx]}")
        raise SystemExit(excep)
    
    return counter