"""
Use Tweet Reader to Read Tweets
"""

import time
import argparse
from pathlib import Path

from reader import PickledCorpusReader

from datetime import datetime
import pytz

import pandas as pd
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
parser.add_argument("output_dir", type=str, help="Directory where output are to be stores. One folder per language.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)
LANG = args.language
OUT_PATH = Path(args.output_dir).joinpath('annotation_sample')
OUT_PATH.mkdir(parents=True, exist_ok=True)
print(OUT_PATH)

# read daily data files and save pos-tagged version to pickle folder
start = time.time()
pkl_corpus = PickledCorpusReader(DATA_PATH.__str__())
tweets = list(pkl_corpus.tweet())
tweets = pd.DataFrame(tweets)
print(f'{(time.time() - start)/60} mins to read all tweets') # 762.8961067199707 ~12min
