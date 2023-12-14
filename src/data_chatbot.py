import pandas as pd
from torchtext.datasets import SQuAD1
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string
import ast
import matplotlib.pyplot as plt
from nltk.corpus import brown
import torch
import nltk
import random
from sklearn.utils import shuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_df():
    train_dataset, dev_dataset = SQuAD1()
    data_dict = {
        "Question": [],
        "Answer": []
    }
    for example in train_dataset + dev_dataset:
        data_dict["Question"].append(example[1]),
        data_dict["Answer"].append(example[2][0])
    df = pd.DataFrame(data_dict)
    df.to_csv("data/squad1_data.csv")      
    #df = pd.read_csv("data/squad1_data.csv",usecols=[1,2])      
    return df
    

def prepare_text(sentence):
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = "".join([c.lower() for c in str(sentence) if c not in string.punctuation])
    tokens = tokenizer.tokenize(sentence) 
    tokens = [ps.stem(a) for a in tokens]
    return tokens
    

def train_valid_split(SRC, TRG, share=0.8):

    '''
    Input: SRC, our list of questions from the dataset
            TRG, our list of responses from the dataset

    Output: Training and valid datasets for SRC & TRG

    '''
    border = int(len(SRC)*share)
    SRC_train_dataset = SRC[:border]
    SRC_valid_dataset = SRC[border:]
    TRG_train_dataset = TRG[:border]
    TRG_valid_dataset = TRG[border:]
    return SRC_train_dataset, SRC_valid_dataset, TRG_train_dataset, TRG_valid_dataset


def questions_answers():
    df_train = load_df()
    questions = [prepare_text(sentence) for sentence in df_train['Question'].values.tolist()]
    answers = [prepare_text(sentence) for sentence in df_train['Answer'].values.tolist()]
    questions_train, questions_valid, answers_train, answers_valid = train_valid_split(questions, answers)
    return questions_train, questions_valid, answers_train, answers_valid

def show_lengths(questions_train, questions_valid, answers_train, answers_valid):
    fig, (one,two) = plt.subplots(1,2)
    fig.tight_layout(pad=1.0)
    one.hist([len(question) for question in questions_train + questions_valid])
    two.hist([len(question) for question in answers_train + answers_valid])
    one.set_title("Length of questions")
    two.set_title("Length of answers")
    plt.show()
    
def toTensor(vocab, sentences):
    tensors = []
    for sentence in sentences:
        vector = []
        for token in sentence:
            vector.append(vocab.word2index[token])
        tensors.append(torch.LongTensor(vector))
    return tensors

def tokenize_questions(sentences, vocab):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentence = []
        for word in sentence:
            try:
                digit = vocab.word2index[word]
            except:
                print(f"Word {word} is not part of the vocabulary!")
            tokenized_sentence.append(digit)
        # the following line is the only difference in comparison to tokenize_answers()
        tokenized_sentence = [vocab.word2index["<SOS>"]] + tokenized_sentence + [vocab.word2index["<EOS>"]] 
        tokenized_sentences.append(torch.LongTensor(tokenized_sentence).to(device))
    return tokenized_sentences

def tokenize_answers(sentences, vocab):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentence = []
        for word in sentence:
            try:
                digit = vocab.word2index[word]
            except:
                print(f"Word {word} is not part of the vocabulary!")
            tokenized_sentence.append(digit)
        tokenized_sentence =  tokenized_sentence + [vocab.word2index["<EOS>"]] 
        tokenized_sentences.append(torch.LongTensor(tokenized_sentence).to(device))
    return tokenized_sentences

def pretrained_w2v(init):
    if init:
        nltk.download('brown')
        nltk.download('punkt')

        #Output, save, and load brown embeddings

        model = gensim.models.Word2Vec(brown.sents())
        model.save('brown.embedding')

    w2v = gensim.models.Word2Vec.load('brown.embedding')
    return w2v


        

    
