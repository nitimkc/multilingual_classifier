#!/bin/bash

# (
#python basic_description.py \\fs02.isglobal.lan\HPC_ADAPTATION\proj\nmishra\p20230515_NM_tweets_learning\tweets\data \\fs02.isglobal.lan\HPC_ADAPTATION\proj\nmishra\p20230704_NM_multilingual_classifier\summary\without_stopwords --language=eu --top_n=25
# python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=eu --top_n=25
# python process_sample.py Z:/proj/nmishra/p20230704_NM_multilingual_classifier/summary Z:/proj/nmishra/p20230704_NM_multilingual_classifier/summary
python process_sample.py Z:/proj/nmishra/p20230515_NM_tweets_learning/tweets/data Z:/proj/nmishra/p20230704_NM_multilingual_classifier/data --language=fr
# ) > test.log