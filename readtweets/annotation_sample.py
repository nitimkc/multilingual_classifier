"""
Use Tweet Reader to read tweets, gather statistical descriptions and 
plot frequency table for N most common token
"""
#!/usr/bin/env python3
import argparse
from pathlib import Path

import time
from datetime import datetime
import pytz

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from reader import PickledCorpusReader

# function to sample rows from pandas dataframe
def sample_tweets(df, min_date, max_date, N, by_cols=['year','month'], replace=True):
    sub_df = df[(df['date']>=min_date) & (df['date']<=max_date)]
    # print(sub_df.groupby(by_cols).size().shape)
    
    sample = sub_df.groupby(by_cols).apply(pd.Series.sample, replace=replace, n=N).reset_index(level=0, drop=True)
    sample.drop_duplicates(subset=['id'], inplace=True)
    return sample

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("result_dir", type=str, help="Directory where results are to be stores. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)
LANG = args.language
OUT_PATH = Path(args.output_dir).joinpath('annotation_sample')
OUT_PATH.mkdir(parents=True, exist_ok=True)
print(OUT_PATH)


# read daily data files and save pos-tagged version to pickle folder
start = time.time()
pkl_corpus = PickledCorpusReader(DATA_PATH.__str__())
# tweets = list(pkl_corpus.tweet())
# tweets = pd.DataFrame(tweets)
tweet = list(pkl_corpus.docs())
tweets = pd.DataFrame.from_records([twt for twtlist in tweet for twt in twtlist])
print(tweets.shape)
print(f'{(time.time() - start)/60} mins to read all tweets') # 762.8961067199707 ~12min

# remove duplicate tweet IDs and RTs
tweets = tweets.drop_duplicates(subset=['id'])
tweets = tweets[~tweets['text'].str.startswith("RT")]
tweets = tweets[~tweets['lang'].isin(['ca', 'eu'])]
print(tweets.shape)
print(tweets['lang'].value_counts())

# convert to datetime and create month and year columns
tweets['date'] = pd.to_datetime(tweets['date'])
tweets['year'] = tweets['date'].dt.year
tweets['month'] = tweets['date'].dt.month

# remove data already sampled
files = [i for i in OUT_PATH.glob('*.csv')]
sampled_df = [pd.read_csv(i) for i in files]
sampled_df = pd.concat(sampled_df)
df_tosample = tweets[~tweets['id'].isin(sampled_df['id'])]
print(sampled_df.shape)
print(df_tosample.shape)

# sample new tweets and save file
for lang, grp in df_tosample.groupby('lang'):
    print(lang)
    
    if lang in ['es', 'it', 'de']:
        sampled = sample_tweets(grp, '2018-09-15', '2023-03-15', 25)
    elif lang == 'fr':
        sampled = sample_tweets(grp, '2011-01-01', '2023-03-15', 10)
    elif lang == 'en':
        sampled = sample_tweets(grp, '2018-09-15', '2019-04-29', 25)
    sampled.sort_values(by='year', ascending=False, inplace=True)
    print(sampled.shape)
    print(sampled.head())
    sampled = sampled[['id','lang','year','text']]
    
    N = 200 if lang == 'en' else 600
    final_sample = sampled.sample(N)
    filename = 'nonoverlap'
    final_sample.to_csv(OUT_PATH.joinpath(f"filename_{lang}.csv"), encoding='utf-8-sig', index=False)