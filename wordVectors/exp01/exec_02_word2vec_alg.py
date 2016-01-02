# https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

import logging
import numpy as np
import json
import codecs
from gensim.models import word2vec

###########################################################################
# Load preprocessed data
###########################################################################

# built-in logging module for Word2Vec for nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with codecs.open('output/sentences_list.json', 'r', encoding='utf-8') as f:
	sentences = json.load(f)

###########################################################################
# Creating vectors from word2vec (Using skip-gram)
###########################################################################

print "Creating the word vectors...\n"
# Initialize and train the WordVector model
num_features = 300    # Word vector dimensionality, size of the numpy-vector, size of the neural-network-layers
min_word_count = 40   # Minimum word count, to filter very rarely used words
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words, to filter stopwords

"""
 Train word2vec on the sentences.
 The first pass collects words and their frequencies to build an internal dictionary tree structure.
 The second pass trains the neural model.
"""
model = word2vec.Word2Vec(sentences, workers=2, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# Serialize the model
model.save("output/IMDB_300features_40minwords_10context.cnpy")
