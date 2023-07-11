#!/bin/bash

(
#python basic_description.py \\fs02.isglobal.lan\HPC_ADAPTATION\proj\nmishra\p20230515_NM_tweets_learning\tweets\data \\fs02.isglobal.lan\HPC_ADAPTATION\proj\nmishra\p20230704_NM_multilingual_classifier\summary\without_stopwords --language=eu --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=eu --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=ca --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=de --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=it --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=fr --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=es --top_n=25
python basic_description.py //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230515_NM_tweets_learning/tweets/data //fs02.isglobal.lan/HPC_ADAPTATION/proj/nmishra/p20230704_NM_multilingual_classifier/summary/without_stopwords --language=en --top_n=25
) > basic_description_withoutstopwords.log