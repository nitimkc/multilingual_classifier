"""
# Runtime ~4hrs
"""
#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import time

from reader import JSONCorpusReader

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
parser.add_argument("result_dir", type=str, help="Directory where results are to be stores. One folder per language.")
# parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
# parser.add_argument("--top_n", type=int, default=20, help="Number of most common tokens to extract.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir) #DATA_PATH = Path('Z:/proj/nmishra/p20230704_NM_multilingual_classifier/data')

OUTPATH = Path(args.result_dir) #if LANG is None else Path(args.result_dir).joinpath(LANG)
OUTPATH.mkdir(parents=True, exist_ok=True)
print(OUTPATH)

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

started = time.time()

corpus = JSONCorpusReader(str(DATA_PATH), fileids=DOC_PATTERN, cat_pattern=CAT_PATTERN, encoding='latin-1')
corpus.get_geo(geopath=OUTPATH.joinpath('geo.jsonl'))

print(time.time() - started)

# started = time.time()

# corpus = JSONCorpusReader(str(DATA_PATH), fileids=DOC_PATTERN, cat_pattern=CAT_PATTERN, encoding='latin-1')
# geos = corpus.get_geo()
# # print(next(geos))
# for geo in geos:
#     with open(OUTPATH.joinpath('geo.jsonl'), 'a', encoding='latin-1') as f:
#         json.dump(geo, f)

# # print(time.time() - started)

