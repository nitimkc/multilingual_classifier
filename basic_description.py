"""
Use Tweet Reader to read tweets, gather statistical descriptions and 
plot frequency table for N most common token
"""

import argparse
from pathlib import Path
import pickle
import string
from reader import JSONCorpusReader

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("result_dir", type=str, help="Directory where results are to be stores. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
parser.add_argument("--top_n", type=int, default=20, help="Number of most common tokens to extract.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)
LANG = args.language
N = args.top_n
OUTPATH = Path(args.result_dir) if LANG is None else Path(args.result_dir).joinpath(LANG)
OUTPATH.mkdir(parents=True, exist_ok=True)
print(OUTPATH)

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

corpus = JSONCorpusReader(DATA_PATH.__str__(), fileids=DOC_PATTERN, cat_pattern=CAT_PATTERN)
# print(corpus.abspaths())
# for cat in corpus.categories():
#     print(cat)
# print(corpus.resolve(fileids=None, categories=LANG))

# tweet_sizes = corpus.sizes(categories=LANG)
# print(next(tweet_sizes))

# tweets = corpus.docs()
# atweet = next(tweets)
# for k,v in atweet.items():
#     print(f"{k}----------{v}")
# print(len(tweets))``

# test = corpus.fields(fields='text')
# test = corpus.fields(fields=['lang', 'text'], categories=LANG)
# test = corpus.tokenized(categories=LANG)
# test = corpus.process_tweets(categories=LANG)

# print(next(test))
# print(next(test))

# basic summaries
stats_summary = corpus.describe(categories=LANG, stopwords=False)
for (k,v) in stats_summary.items():
    if k!='tokens_freq':
        print(k,v)

# store token frequency dictionary and plot n most common tokens
tokens = stats_summary['tokens_freq']
top_tokens = tokens.most_common(N)
print(top_tokens)

freq_series = pd.Series(dict(tokens)).sort_values(ascending=False)
nopunc_series = freq_series[~freq_series.index.isin(list(string.punctuation))]
# list(string.punctuation) + ['url', '@user']

with open(OUTPATH.joinpath(f'token_freq_series_{LANG}.pickle'), 'wb') as handle:
    pickle.dump(freq_series, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(OUTPATH.joinpath(f'token_nopunc_series_{LANG}.pickle'), 'wb') as handle:
    pickle.dump(nopunc_series, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x=nopunc_series.index[:N], y=nopunc_series[:N].values, ax=ax)
plt.xticks(rotation=30)
plt.title(f"Frequency Distribution of Non-Punctuation Tokens in {LANG} Language Corpus")
plt.xlabel('Count')
plt.xlabel('Tokens')
plt.savefig(OUTPATH.joinpath(f'token_freq_distplot_{LANG}.jpg'))


