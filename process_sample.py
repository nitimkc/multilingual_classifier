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

from reader import JSONCorpusReader

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("result_dir", type=str, help="Directory where results are to be stores. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
# parser.add_argument("--top_n", type=int, default=20, help="Number of most common tokens to extract.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)
LANG = args.language
# N = args.top_n
OUTPATH = Path(args.result_dir) #if LANG is None else Path(args.result_dir).joinpath(LANG)
OUTPATH.mkdir(parents=True, exist_ok=True)
print(OUTPATH)

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

# read data and convert to pandas dataframe
started = time.time()
corpus = JSONCorpusReader(str(DATA_PATH), fileids=DOC_PATTERN, cat_pattern=CAT_PATTERN, encoding='latin-1')
tweet = corpus.docs(categories=LANG)
tweets = pd.DataFrame.from_records(list(tweet))
print(tweets.shape)
print(tweets.head())
print(time.time() - started)

# make a copy of the full pandas dataframe (save in chucks to avoid memory issues)
# started = time.time()
# # tweets.to_json(OUTPATH.joinpath("full_df_new.jsonl"), orient="records", lines=True, force_ascii=False)
# N = 5000
# chunks = [tweets[i:i+N] for i in range(0, tweets.shape[0], N)]
# f = open(OUTPATH.joinpath("full_df_new.jsonl"), mode="a", encoding='utf8')
# for chunk_df in chunks:
#     f.write(chunk_df.to_json(orient="records", lines=True, force_ascii=False))
# f.close()
# print(time.time() - started)

# remove duplicate tweet IDs and RTs
tweets = tweets.drop_duplicates(subset=['id'])
tweets = tweets[~tweets['text'].str.startswith("RT")]

# filter full data by language to ensure having it regarless of which query it came from
df_short = tweets[tweets['lang'].isin(corpus.categories()[:-1])] # -1 due to 'tt' dir

# filter by date utc offset 1 to 2 hrs depending on Daylight Savings
df_short['created_at'] = df_short['created_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z').astimezone(pytz.utc))
df_short.sort_values(by='created_at', inplace=True)
df_short['date'] = df_short['created_at'].apply(lambda x: x.strftime("%Y-%m-%d"))
df_short['month'] = df_short['created_at'].dt.month
df_short.head()

# save file by lang and date
print(df_short['lang'].value_counts())
for lang, grp in df_short.groupby('lang'):
    for day, lang_grp in grp.groupby('date'):
        print(lang, day, lang_grp.shape)
        fpath = OUTPATH.joinpath(lang)
        fpath.mkdir(parents=True, exist_ok=True)
        lang_grp.to_json(fpath.joinpath(f"{day}.jsonl"), orient='records', lines=True)

# plot ts all language
fig, ax = plt.subplots(figsize=(12, 4))
df_short.groupby(['date','lang']).size().unstack().plot(legend=True, alpha=0.7, ax=ax)
plt.title("# of tweets in query (after language filter)")
plt.xlabel('time')
plt.show()

# plot ts per language
for key, grp in df_short.groupby('lang'):
    lang_df = grp.groupby('date').size()
    # lang_df = lang_df[lang_df>1]
    lang_df.index = pd.to_datetime(lang_df.index)
    lang_df.plot(alpha=0.4, figsize=(10,4))
    plt.title(f"# of tweets in {key} language query (after language filter)")
    plt.xlim(min(lang_df.index), max(lang_df.index))
    plt.xlabel('time')
    plt.show()
    fpath = OUTPATH.parent.joinpath('summary').joinpath('ts_plot')
    fpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath.joinpath(f"ts_{key}.jpg"))

# sample by language and month for year 2018+
df_sample = df_short[df_short['created_at'].dt.year>=2018]
for lang, grp in df_sample.groupby('lang'):
        sample = (grp.groupby(['month']).apply(pd.Series.sample, replace=True, n=10)
        .droplevel([0]).reset_index()
        )
        print(sample.shape)
        sample.drop(['index', 'date','month'], axis=1, inplace=True)
        fpath = OUTPATH.parent.joinpath('summary').joinpath('sample_2018+')
        fpath.mkdir(parents=True, exist_ok=True)
        sample.to_csv(fpath.joinpath(f"sample_{lang}.csv"), index=False)
        sample.to_json(fpath.joinpath(f"sample_{lang}.jsonl"), orient='records', lines=True)
