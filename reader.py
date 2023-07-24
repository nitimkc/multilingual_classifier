"""
Tweet Reader Class Object
"""

import json
import os
import time
import codecs
import logging
import pickle
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

class JSONCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for raw line-delimited JSON documents to enable preprocessing.
    """

    def __init__(self, root, fileids, encoding='utf8', word_tokenizer=TweetTokenizer(), **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """

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
                            line.replace("\n", " ")
                            print(f"Json decode error in file {path}:\n{line}")

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

    def fields(self, fields, fileids=None, categories=None):
        """
        extract particular fields from the json object. Can be string or an 
        iterable of fields. If just one fields in passed in, then the values 
        are returned, otherwise dictionaries of the requested fields returned
        """
        if isinstance(fields, string_types):
            fields = [fields,]

        if "text" in fields:
            raise KeyError("To extract 'text' field, please use other methods: process_tweets, processed_tweets")

        for doc in self.docs(fileids, categories):
            if (len(fields) == 1) & (fields[0] in doc):
                yield doc[fields[0]]
            else:
                yield {key : doc.get(key, None) for key in fields}

    def get_geo(self, fileids=None, categories=None):
        """
        check if the tweet as geo object and return it if it exists        
        """
        geo = self.fields('geo', fileids, categories)
        if geo is not None:
            yield geo

    def process_tweets(self, fileids=None, categories=None):
        """
        Returns processed tokens from tokenized tweets.
        """
        for tweet in self.docs(fileids, categories):
            tokenized_tweet = self._word_tokenizer.tokenize(tweet.get('text', None))
            # pos_tag(tokens, lang=self.fields('lang')) supports english and russian only

            processed = []
            for token in tokenized_tweet:
                if '@' in token:
                    token = '@user'
                elif ('http' or 'https') in token:
                    token = 'URL'
                else:
                    pass
                processed.append(token.strip())
            yield processed

    def processed_tweets(self, fileids=None, categories=None):
        """
        Returns processed tokens as a string.
        """
        for processed_tokens in self.process_tweets(fileids, categories):
            yield " ".join(processed_tokens)

    def describe(self, fileids=None, categories=None, stopwords=False):
        """
        Performs a single pass of the corpus and returns a nested dictionary with a 
        variety of metrics concerning the state of the corpus, including frequency
        distribution of tokens.
        """
        started = time.time()
        if stopwords:
            lang_dict = {'en':'english', 'fr':'french', 'de':'german', 'es':'spanish', 'it':'italian', 'ca':'catalan', 'eu':'basque'}
            stop_words = {k:'v' for k in nltk.corpus.stopwords.words(fileids=lang_dict[categories])}

        # Structures to perform counting
        counts  = nltk.FreqDist()
        tokens  = nltk.FreqDist()

        # Perform single pass over tweet and count
        for processed in self.process_tweets(fileids, categories):
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

