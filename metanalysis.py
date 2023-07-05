from reader import JSONCorpusReader

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'

test = JSONCorpusReader("/PROJECTES/ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data", fileids=DOC_PATTERN)