## *- coding: utf-8 -*-
"""
1. load translated data for each language, filter translation of all language to one and save
2. Run deepl_combine_onetomany.py first to have required data.

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
        "--fp",
        dest="path_of_files",
        required=True,
        help="path to files that contains translated tweets for each language as jsonl",
    )
    parser.add_argument(
        "--sp",
        dest="destination_path",
        required=True,
        help="path where file in csv is to be saved",
    )
    return parser

def main():
    
    parser = get_parser(prog_name=None)
    args = parser.parse_args()

    # Use most translation features of the library
    DATA_PATH = Path(args.path_of_files)
    SAVE_PATH = Path(args.destination_path)

    translated_df_list = [pd.read_csv(i) for i in DATA_PATH.glob('*.csv')]
    translated_df = pd.concat(translated_df_list)
    print(translated_df.shape)

    translated_df_lang = {}
    for key, lang_df in translated_df.groupby('lang'):
        print(key)
        print(lang_df.shape)
        print(lang_df.head())
        print(lang_df['index'].nunique()) 
        # lang_df.to_csv(SAVE_PATH.joinpath(f"tweets_translated_{key}.csv"), index=False)


if __name__ == "__main__":
    main()    
