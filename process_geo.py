"""
# Runtime ~4hrs
# need to re-run with language field because description requires knowledge of language
conda install -c conda-forge geopy
"""
#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import time

from reader import JSONCorpusReader
from geo_locator import TweetGeoProcessor

parser = argparse.ArgumentParser(description="Twitter Meta Analysis")
parser.add_argument("data_dir", type=str, help="Directory where tweets resides. One folder per language.")
# parser.add_argument("result_dir", type=str, help="Directory where results are to be stores. One folder per language.")
parser.add_argument("--language", type=str, default=None, help="Language of the tweet to work with.")
# parser.add_argument("--top_n", type=int, default=20, help="Number of most common tokens to extract.")
args = parser.parse_args()

DATA_PATH = Path(args.data_dir) 
LANG = args.language
# OUTPATH = Path(args.result_dir) #if LANG is None else Path(args.result_dir).joinpath(LANG)
# OUTPATH.mkdir(parents=True, exist_ok=True)
# print(OUTPATH)

started = time.time()

DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

corpus = JSONCorpusReader(str(DATA_PATH), fileids=DOC_PATTERN, encoding='latin-1')
geos = corpus.get_geo(categories=LANG) # contains 'id','place_id','geo','location','description'
print(next(geos))

geoprocessor = TweetGeoProcessor()
# locations = []
# for geo in geos:
    # print(geoprocessor.get_item(geo, 'location'))
    # locations.append(geoprocessor.get_location(geo)) 

print(time.time() - started)

