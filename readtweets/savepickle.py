"""
Use Tweet Reader to Read Tweets
Save as pickled files
"""

import time
import argparse
from pathlib import Path

import pickle
from reader import JSONCorpusReader
from preprocess import Preprocessor

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
parser.add_argument("output_dir", type=str, help="Directory where output are to be stores. One folder per language.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)
LANG = args.language
OUT_PATH = Path(args.output_dir) if LANG is None else Path(args.output_dir).joinpath(LANG)
OUT_PATH = DATA_PATH.parent.joinpath('anonymized')
OUT_PATH.mkdir(parents=True, exist_ok=True)
print(OUT_PATH)

# read daily data files and save anonymized version to data annotation sample
start = time.time()
anon = True
corpus = JSONCorpusReader(DATA_PATH.__str__())
tweets_processor = Preprocessor(corpus, target=OUT_PATH, anonymize=anon)
test = tweets_processor.transform()
print(time.time() - start) # ~1hr


# corpus reader usage
# corpus = JSONCorpusReader(DATA_PATH.__str__())
# tweet = corpus.docs()
# atweet = next(tweet)
# for k,v in atweet.items():
#     print(f"{k}----------{v}") 

# preprocessor usage
# tweets_processor = Preprocessor(corpus, target=OUT_PATH, anonymize=True)
# doc_path = tweets_processor.abspath('it/2019-08-06.jsonl')
# print(doc_path)
# processed = tweets_processor.process('it/2019-08-16.jsonl', tag_pos=True)
# print(processed)
# anonymized = tweets_processor.anonymized_docs('it/2019-08-16.jsonl')
# print(next(anonymized))