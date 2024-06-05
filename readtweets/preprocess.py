import os
from pathlib import Path
import time
import pickle
import re

import logging
log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

log = logging.getLogger("log")
log.setLevel('WARNING')

def anonymize_tweet(tweet):
    tweet = re.sub(r"http\S+", 'URL', tweet)
    tweet = re.sub(r"@\S+", '@user', tweet)
    return tweet

class Preprocessor(object):
    """
    The preprocessor wraps a corpus object (ex `JSONCorpusReader`)
    and manages the stateful tokenization and part of speech tagging into a
    directory that is stored in a format that can be read by the
    `JSONPickledCorpusReader`. This format is more compact and necessarily
    removes a variety of fields from the document. This format however is more
    easily accessed for common parsing activity.
    """

    def __init__(self, corpus, target=None, anonymize=False, **kwargs):
        """
        The corpus is the `JSONCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.corpus = corpus
        self.target = target
        self.anonymize = anonymize

    def fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        """
        Returns the absolute path to the target fileid from the corpus fileid.
        """
        docpath = Path(self.corpus.abspath(fileid))
        parent = docpath.relative_to(self.corpus.root).parent
        return self.target.joinpath(parent, docpath.stem + '.pickle')

    def anonymized_docs(self, fileid):
        """
        Returns anonymized tweets.
        """
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = target.parent

        # Make sure the directory exists or raise error if its a file
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir():
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )

        # Create a data structure for the pickle        
        docs = list(self.corpus.docs(fileid)) 
        docs = [anonymize_tweet(doc.get('text', None)) for doc in docs]

        # get additional tweet info
        doc_tweet_info = list(self.corpus.fields(fileids=fileid, fields=['id','date','lang']))
        if len(doc_tweet_info) != len(docs):
            raise ValueError(
                f"Number of tweet text and tweet info are not matching in the document {target}"
            )

        # add tweet info to anonymized tweet
        anon_docs = []
        for (info, tweet) in zip(doc_tweet_info, docs):
            info.update({'text':tweet})
            anon_docs.append(info)

        # Open and serialize the pickle to disk
        with open(target, 'wb') as doc_file:
            pickle.dump(anon_docs, doc_file, pickle.HIGHEST_PROTOCOL)
        del docs        # Clean up the document
        return target   # Return the target fileid


    def tagged(self, fileid):
        """
        For a single file does the preprocessing work.
        This method is called multiple times from the transform runner.
        """
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = target.parent

        # Make sure the directory exists or raise error if its a file
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir():
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )

        # Create a data structure for the pickle
        docs = list(self.corpus.docs(fileid)) 
        docs = [anonymize_tweet(doc.get('text', None)) for doc in docs]
        docs = [self.corpus._word_tokenizer.tokenize(doc.get('text', None)) for doc in docs]
        docs = [ [pos_tag(tweet) for tweet in doc] for doc in docs] # supports english and russian only

        doc_tweet_info = list(self.corpus.fields(fileids=fileid, fields='id'))
        if len(doc_tweet_info) != len(docs):
            raise ValueError(
                f"Number of tweet text and tweet info are not matching in the document {target}"
            )
        
        # add tweet info to anonymized tweet
        tagged_docs = []
        for (info, tweet) in zip(doc_tweet_info, docs):
            info.update({'text':tweet})
            tagged_docs.append(info)
        print(len(tagged_docs))
        print(tagged_docs[-1])

        # Open and serialize the pickle to disk
        with open(target, 'wb') as doc_file:
            pickle.dump(tagged_docs, doc_file, pickle.HIGHEST_PROTOCOL)
        del docs                 # Clean up the document
        del tagged_docs  
        return target            # Return the target fileid

    def transform(self, fileids=None, categories=None):
        """
        Transform the wrapped corpus, writing out the segmented, tokenized,
        and part of speech tagged corpus as a pickle to the target directory.
        This method will also directly copy files that are in the corpus.root
        directory that are not matched by the corpus.fileids().
        """
        self.target.mkdir(parents=True, exist_ok=True)

        # Resolve the fileids to start processing and return the list of
        # target file ids to pass to downstream transformers.
        if self.anonymize:
            return [
                self.anonymized_docs(fileid)
                for fileid in self.fileids(fileids, categories)
            ]
        else:
            return [
                self.tagged(fileid)
                for fileid in self.fileids(fileids, categories)
            ]
            
