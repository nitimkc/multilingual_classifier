"""
Use Tweet Reader to Read Tweets
"""

import argparse
from pathlib import Path

from reader import JSONCorpusReader

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Main directory where tweets resides. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir)

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

corpus = JSONCorpusReader(DATA_PATH.__str__(), fileids=DOC_PATTERN, cat_pattern=CAT_PATTERN)
# print(corpus.abspaths())
# for cat in corpus.categories():
#     print(cat)
# print(corpus.resolve(fileids=None, categories='es'))

# tweet_sizes = corpus.sizes(categories='es')
# print(next(tweet_sizes))

# tweets = corpus.docs()
# atweet = next(tweets)
# for k,v in atweet.items():
#     print(f"{k}----------{v}")
# print(len(tweets))

# test = corpus.fields(fields='text')
# test = corpus.fields(fields=['lang', 'text'], categories='es'
# test = corpus.tokenized(categories='es')
# test = corpus.process_tweets(categories='es')

# print(next(test))
# print(next(test))

test = corpus.describe(categories='eu')
print(test)



