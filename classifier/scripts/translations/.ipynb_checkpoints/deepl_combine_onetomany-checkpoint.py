## -*- coding: utf-8 -*-
"""
1. load translated data for each language, combine with original and save
2. Repeat for each of the five langauges

"""
import os
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wrapper_deepl import translate_to_one

def get_parser(prog_name):
    """Constructs and returns the argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Combine translated tweets"
    )
    parser.add_argument(
        "--f",
        dest="original_file",
        required=True,
        help="file in csv that contains all annotated tweets",
    )
    parser.add_argument(
        "--fp",
        dest="path_of_files",
        required=True,
        help="path to files that contains translated tweets for each language as jsonl",
    )
    parser.add_argument(
        "--lang",
        dest="language_to_combine",
        required=True,
        help="Language for which the translations are to be combined",
    )
    return parser

def main():
    
    parser = get_parser(prog_name=None)
    args = parser.parse_args()

    # Use most translation features of the library
    original_df = pd.read_csv(args.original_file)
    DATA_PATH = Path(args.path_of_files)
    lang = args.language_to_combine

    translated_df_list = []
    # for each language in the original dataframe
    for key, lang_df in original_df.groupby('lang'):
        if key == lang:
            print(f'original lang - {key}')

            # edit and append original dataframe for that lang
            original_lang_df = lang_df[["id","lang","tweet","final_annotation"]].copy()
            original_lang_df["source_lang"] = lang
            original_lang_df["index"] = original_lang_df.index
            original_lang_df = original_lang_df[["index","id","lang","tweet","final_annotation","source_lang"]]
            print(original_lang_df.shape)
            translated_df_list.append(original_lang_df) 

            # load translated data for every other language in the original dataframe
            translated_files = [i for i in DATA_PATH.glob('*.jsonl') if i.stem.startswith(key)]
            for file in translated_files:
                print(f'translate to lang - {file.stem[-2:]}')
                translated_df = pd.read_json(str(file), lines=True, encoding='utf-8')
                # translated_df.rename(columns={'translated_tweet':"tweet"}, inplace=True)
                print(translated_df.shape)
                
                # create a new dataframe that contains original dataframe values except the tweet as translated tweets
                new_df = original_lang_df[["index","id","lang","final_annotation"]].copy()
                new_df.rename(columns={"lang":"source_lang"}, inplace=True)
                new_df["lang"] = file.stem[-2:]
                new_df["tweet"] = translated_df["translated_tweet"].tolist()
                # new_df = new_df.merge(translated_df, on='id', how='outer')
                # merge not working with ids even when it exists
                print(new_df.shape)
                new_df = new_df[["index","id","lang","tweet", "final_annotation","source_lang"]]

                # append to the list
                translated_df_list.append(new_df)

            # convert list of appended dataframe to one dataframe and save
            print(len(translated_df_list))
            alldf = pd.concat(translated_df_list)
            print(alldf.shape)
            alldf.to_csv(DATA_PATH.joinpath(f"{lang}_tweets_translated.csv"), index=False)
        else:
            pass


if __name__ == "__main__":
    main()    
