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
    #df.to_csv("data/squad1_data.csv")      
    #df = pd.read_csv("data/squad1_data.csv",usecols=[1,2])      
    return df
    

def prepare_text(sentence):
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = "".join([c.lower() for c in str(sentence) if c not in string.punctuation])
    tokens = tokenizer.tokenize(sentence) 
    tokens = [ps.stem(a) for a in tokens]
    return tokens

def questions_answers(amount):
    df_train = load_df()
    questions = [prepare_text(sentence) for sentence in df_train['Question'].values.tolist()][:amount]
    answers = [prepare_text(sentence) for sentence in df_train['Answer'].values.tolist()][:amount]
    return questions, answers

def show_lengths(questions, answers):
    fig, (one,two) = plt.subplots(1,2)
    fig.tight_layout(pad=1.0)
    one.hist([len(question) for question in questions])
    two.hist([len(answer) for answer in answers])
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

def vectorize_questions(sentences, vocab):
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

def vectorize_answers(sentences, vocab):
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


        

    
