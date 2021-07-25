# coding: utf-8
# Copyright statement: Some codes refer to the following open source project:
#
# Natural Language Toolkit: Twitter Tokenizer
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Christopher Potts <cgpotts@stanford.edu>
#         Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
#         Pierpaolo Pantone <> (modifications)
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

# Input: pos_comment = ["sentence1", "sentence2", "sentence3"...]
#        neg_commenr = ["sentence11", "sentence22", "sentence33"...]
# Output: pos_comment = [['word1', 'word2'...]
#                        ['w1', 'w2', ...]...
#                       ]


# regular expression operations
import re
# string operation
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from Text_Preprocess.TextTokenizer import *


# helper class for doing text preprocessing
class Text_Preprocess():

    def __init__(self):

        # instantiate tokenizer class
        self.tokenizer = TextTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        # get the english stopwords
        self.stopwords_en = stopwords.words('english')
        # get the english punctuation
        self.punctuation_en = string.punctuation
        # Instantiate stemmer object
        self.stemmer = PorterStemmer()

    # 移除不想要的字符
    def __remove_unwanted_characters__(self, comment):

        # remove forward style text "RT"
        # comment = re.sub(r'^RT[\s]+', '', comment)
        # remove hyperlinks
        # comment = re.sub(r'https?:\/\/.*[\r\n]*', '', comment)
        # remove hashtags
        # comment = re.sub(r'#', '', comment)
        # remove email address
        # comment = re.sub('\S+@\S+', '', comment)

        # remove numbers
        comment = re.sub(r'\d+', '', comment)
        # remove '\n' ,换行符
        comment = re.sub(r'\n', '', comment)
        # remove '\t', tab符
        comment = re.sub(r'\t', '', comment)
        # remove '\r', 回车符
        comment = re.sub(r'\r', '', comment)
        # remove '/'
        comment = re.sub(r'/', '', comment)
        # remove ':'
        comment = re.sub(r':', '', comment)
        # remove ';'
        comment = re.sub(r';', '', comment)
        # remove '\"'
        comment = re.sub(r'\"', '', comment)

        ## return removed text
        return comment

    def __tokenize_comment__(self, comment):
        # tokenize comment
        return self.tokenizer.tokenize(comment)

    def __remove_stopwords__(self, comment_tokens):
        # remove stopwords
        comment_clean = []

        for word in comment_tokens:
            if word not in self.stopwords_en:
                comment_clean.append(word)
        return comment_clean

    # 移除非字母字符
    def __remove_not_alpha__(self, comment_tokens):

        comment_isalpha = []
        for word in comment_tokens:
            # remove not alpha
            if word.isalpha():
                comment_isalpha.append(word)

        return comment_isalpha

    def __text_stemming__(self, comment_tokens):
        # store the stemmed word and remove the none alphabet
        comments_stem = []

        for word in comment_tokens:
            # stemming word
            stem_word = self.stemmer.stem(word)
            comments_stem.append(stem_word)

        return comments_stem

    def preprocess(self, comments):

        comments_processed = []

        for _, comment in tqdm(enumerate(comments)):

            # apply removing unwated characters and remove style of forward, URL
            comment = self.__remove_unwanted_characters__(comment)
            # apply nltk tokenizer
            comment_tokens = self.__tokenize_comment__(comment)
            # apply stop words removal
            comment_clean = self.__remove_stopwords__(comment_tokens)
            # apply not alphabet removal
            comment_clean_1 = self.__remove_not_alpha__(comment_clean)
            # apply stemmer
            comment_stems = self.__text_stemming__(comment_clean_1)

            # remove 0 word sentence
            if len(comment_stems) >= 1:
                comments_processed.extend([comment_stems])

        return comments_processed
