"""
Use Tweet Reader to Read Tweets
"""

import argparse
from pathlib import Path
import pickle
from reader import JSONCorpusReader

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("result_dir", type=str, help="Directory where results are to be stores. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)
LANG = args.language
OUTPATH = Path(args.result_dir) if LANG is None else Path(args.result_dir).joinpath(LANG)
OUTPATH.mkdir(parents=True, exist_ok=True)
print(OUTPATH)

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

corpus = JSONCorpusReader(DATA_PATH.__str__(), fileids=DOC_PATTERN, cat_pattern=CAT_PATTERN)
stats_summary = corpus.describe(categories=LANG)
tokens = stats_summary['tokens_freq']
for (k,v) in stats_summary.items():
    if k!='tokens_freq':
        print(k,v)

# store token frequency dictionary
with open(OUTPATH.joinpath(f'token_freq_{LANG}.pickle'), 'wb') as handle:
    pickle.dump(tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)




