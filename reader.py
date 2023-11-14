"""
Tweet Reader Class Object
"""

import json
import os
import time
import codecs
import pickle

import logging
log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

from six import string_types
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

import nltk
# nltk.download('averaged_perceptron_tagger')
# from nltk.tag import pos_tag

from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')

log = logging.getLogger("log")
log.setLevel('WARNING')

CAT_PATTERN = r'([a-z_\s]+)/.*'
PKL_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.pickle'
DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.jsonl'

class JSONCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for raw line-delimited JSON documents to enable preprocessing.
    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8', word_tokenizer=TweetTokenizer(), **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        self._word_tokenizer = word_tokenizer

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            print(f"processing categories: {categories}")
            return self.fileids(categories)
        return fileids

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete tweet from line delimited JSON document,
        closing after done, reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as file:
                for line in file:
                    try:
                        yield json.loads(line)
                    except json.decoder.JSONDecodeError as readerror:
                        if readerror.msg == "Extra data":
                            yield json.loads(line.replace("\n", " "))
                            print(f"Json decode error in file {path}:\n{line}")

    def tokenize(self, fileids=None, categories=None):
        """
        Returns tokenized tweets.
        """
        for doc in self.docs(fileids, categories):
            yield self._word_tokenizer.tokenize(doc.get('text', None))

    def fields(self, fields, fileids=None, categories=None):
        """
        extract particular fields from the json object. Can be string or an 
        iterable of fields. If just one fields in passed in, then the values 
        are returned, otherwise dictionaries of the requested fields returned
        """
        if isinstance(fields, string_types):
            fields = [fields,]

        if "text" in fields:
            raise KeyError("To extract 'text' field use process_tweets or processed_tweets")

        for doc in self.docs(fileids, categories):
            if (len(fields) == 1) & (fields[0] in doc):
                yield doc[fields[0]]
            else:
                yield {key : doc.get(key, None) for key in fields}
                
    # def get_geo(self, fileids=None, categories=None, geopath=None):
    #     """
    #     check if the tweet as geo object and return it if it exists
    #     """
    #     for geo_fields in self.fields(['id', 'place_id', 'geo', 'user_info', 'lang'], fileids, categories):
    #         geo = {k:v for k,v in geo_fields.items() if k in ['id', 'place_id', 'geo']}
    #         geo['location'] = geo_fields['user_info'].get('location', None)
    #         geo['description'] = geo_fields['user_info'].get('description', None)
    #         if geopath is not None:
    #             with open(geopath, 'a', encoding='latin-1') as file:
    #                 json.dump(geo, file)
    #                 file.write('\n')
    #         else:
    #             yield geo

    def describe(self, fileids=None, categories=None, stopwords=False):
        """
        Performs a single pass of the corpus and returns a nested dictionary with a 
        variety of metrics concerning the state of the corpus, including frequency
        distribution of tokens.
        apply stopwords filter, default to not apply
        """
        started = time.time()
        if stopwords:
            lang_dict = {'en':'english', 'fr':'french', 'de':'german', 'es':'spanish', 'it':'italian', 'ca':'catalan', 'eu':'basque'}
            stop_words = {k:'v' for k in nltk.corpus.stopwords.words(fileids=lang_dict[categories])}

        # Structures to perform counting
        counts  = nltk.FreqDist()
        tokens  = nltk.FreqDist()

        # Perform single pass over tweet and count
        for processed in self.tokenize(fileids, categories):
            tokens_list = [token for token in processed if token.lower() not in stop_words] if stopwords else processed
            for token in tokens_list:
                counts['tokens'] += 1
                tokens[token.lower()] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics  = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'tokens':  counts['tokens'],
            'vocab':  len(tokens),
            'lexdiv': float(counts['tokens']) / float(len(tokens)),
            'tokens_freq': tokens,
            'secs':   time.time() - started,
        }

class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader pickled files.
    """

    def __init__(self, root, fileids=PKL_PATTERN,  tagged=False, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

        self.tagged = tagged

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            print(f"processing categories: {categories}")
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. This uses a generator to acheive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, encoding, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def tweet(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for tweet in doc:
                yield tweet

    def processed_tweet(self, fileids=None, categories=None):
        """
        Returns  generator of the text of processed tweet after 
        concatenating its individual tokens with space.
        """
        for tweet in self.tweet(fileids, categories):
            if len(tweet['text'][0])==1:
                tweet['text'] = " ".join(tweet['text'])
            else:
                tweet['text'] = " ".join([token for (token, tag) in tweet['text']]) 
            yield tweet

    def token(self, fileids=None, categories=None):
        """
        Returns a generator of (token, tag) tuples or only tokens
        depending on whats in the data.
        """
        for tweet in self.tweet(fileids, categories):
            for token in tweet['text']:
                if len(token)==1:
                    yield token
                else:
                    yield token[0]

