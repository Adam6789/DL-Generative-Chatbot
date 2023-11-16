from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.stem import *

def loadEmbedding(init=True):
    if init:
        nltk.download('brown')
        nltk.download('punkt')
        model = gensim.models.Word2Vec(brown.sents())
        model.save('brown.embedding')

w2v = gensim.models.Word2Vec.load('brown.embedding')