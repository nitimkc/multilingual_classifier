## -*- coding: utf-8 -*-
"""
1. Connect to deepl API and obtain translations for each tweet

"""
import os
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
import deepl

from wrapper_deepl import translate_to_one

load_dotenv()

env_auth_key = "DEEPL_API_KEY"
env_server_url = "DEEPL_SERVER_URL"

languages = {"en":"EN-US", "fr":"FR", "es":"ES", "it":"IT", "de":"DE", "da":"DA"}

def get_parser(prog_name):
    """Constructs and returns the argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Translate tweets in csv using the DeepL API "
        "(https://www.deepl.com/docs-api)."
    )
    parser.add_argument(
        "--auth_key",
        default=None,
        help="authentication key as given in your DeepL account; the "
        f"{env_auth_key} environment variable is used as secondary fallback",
    )
    parser.add_argument(
        "--server_url",
        default=None,
        metavar="URL",
        help=f"alternative server URL for testing; the {env_server_url} "
        f"environment variable may be used as secondary fallback",
    )
    parser.add_argument(
        "--file",
        dest="file_to_translate",
        required=True,
        help="file in csv that contains tweets to translate in column tweet and defines source language in column lang",
    )
    parser.add_argument(
        "--lang",
        dest="language_of_text",
        required=True,
        help="The language of original tweet that requires translations.",
    )
    parser.add_argument(
        "--id",
        dest="id_to_translate_from",
        type=int,
        required=True,
        help="The id of the tweet from which to start translating.",
    )
    parser.add_argument(
        "--tr_lang",
        dest="language_to_translate",
        required=True,
        help="The id of the tweet from which to start translating.",
    )
    return parser

def main():
    
    parser = get_parser(prog_name=None)
    args = parser.parse_args()

    auth_key = args.auth_key or os.getenv(env_auth_key)
    server_url = args.server_url or os.getenv(env_server_url)
    
    if auth_key is None:
        raise Exception(
            f"Please provide authentication key via the {env_auth_key} "
            "environment variable or --auth_key argument"
        )

    # Create a Translator object, and call get_usage() to validate connection
    translator: deepl.Translator = deepl.Translator(
        auth_key, server_url=server_url
    )
    u: deepl.Usage = translator.get_usage()
    u.any_limit_exceeded

    # Use most translation features of the library
    DATA_FILE = Path(args.file_to_translate)
    df = pd.read_csv(DATA_FILE)
    
    # for each language in the original dataframe
    for lang, lang_df in df.groupby("lang"):

        # if language in loop is the language for which translation is required
        if lang==args.language_of_text:
            print(f"Original language: {lang}")

            # save the language to translate to from parser
            translatetolang = args.language_to_translate
            print(f"Language to translate to : {translatetolang}")

            # if tweet id start translating from that tweet id and onwards
            if args.id_to_translate_from>0:
                start = lang_df[lang_df["id"]==args.id_to_translate_from].index[0]
                start_idx = lang_df.index.get_loc(start)
                df_to_translate = lang_df.iloc[start_idx:]
                print(f"Translating text starting index :{start_idx}")
            
            else:
                # otherwise translate all tweets
                df_to_translate = lang_df
            # print(df_to_translate.head())
            print(f"Number of tweets to translate :{df_to_translate.shape[0]}")
            print(f"No. of characters to translate: {np.sum(df_to_translate['tweet'].str.len())}")
            
            OUT_PATH = DATA_FILE.parent.parent.joinpath('translated')
            n = translate_to_one(df_to_translate, lang, translatetolang, languages, translator, OUT_PATH)
            print(f"Success - {n} tweets in {lang} translated to {translatetolang} and saved.")

if __name__ == "__main__":
    main()
# activate venv run
# python deepl_connect.py --file /gaueko0/users/nmishra/multiling_fludetection/data/all/alldata.csv --lang 'de' --id 1267392250394431488 --tr_lang it   
            
