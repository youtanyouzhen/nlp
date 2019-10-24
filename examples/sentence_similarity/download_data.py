#Import Packages
import sys
# Set the environment path
# sys.path.append("../../")  
import os
sys.path.append(os.getcwd())  
print(os.getcwd())
from collections import Counter
import math
import numpy as np
from tempfile import TemporaryDirectory

import scrapbook as sb
import scipy
from scipy.spatial import distance
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import nltk
#  nltk.download("punkt", quiet=False)
#Import utility functions

from utils_nlp.dataset.preprocess import to_lowercase, to_spacy_tokens
from utils_nlp.dataset import stsbenchmark
from utils_nlp.dataset.preprocess import (
    to_lowercase,
    to_spacy_tokens,
    rm_spacy_stopwords,
)
from utils_nlp.models.pretrained_embeddings import word2vec
from utils_nlp.models.pretrained_embeddings import glove
from utils_nlp.models.pretrained_embeddings import fasttext

print("System version: {}".format(sys.version))
print("Gensim version: {}".format(gensim.__version__))