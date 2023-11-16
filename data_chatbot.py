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
import gensim
import random
from sklearn.utils import shuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_df(init=True, source_name="squad1"):
    if source_name == "squad1":

        # save data
        if init:
            train_dataset, dev_dataset = SQuAD1()
            def save_df(dataset, file_name):
                data_dict = {
                    "Question": [],
                    "Answer": []
                }
                for example in dataset:
                    data_dict["Question"].append(example[1]),
                    data_dict["Answer"].append(example[2][0])
                df_train = pd.DataFrame(data_dict)
                df_train.to_csv(file_name)

            save_df(train_dataset, "train_data_squad1.csv")
            save_df(dev_dataset, "test_data_squad1.csv")       
        # load data
        df_train = pd.read_csv("train_data_squad1.csv",usecols=[1,2])
        df_test = pd.read_csv("test_data_squad1.csv",usecols=[1,2])
        df_train.append(df_test)
        
    elif source_name == "poc":
        # save data
        if init:
            df_names = pd.read_csv('names.csv')
            poc_questions = []
            poc_answers = []
            for i, row in list(df_names.iterrows())[:25]:
                poc_questions.append("What is your name?")
                name = row["First Name"]+" "+row["Last Name"]
                poc_answers.append(f"['My name is {name}']")
            for i in range(25):
                poc_questions.append("What is your name?")
                one = random.randint(0,99)
                two = random.randint(0,99)
                name_one = df_names["First Name"].iloc[one]+" "+df_names["Last Name"].iloc[two]
                name_two = df_names["First Name"].iloc[two]+" "+df_names["Last Name"].iloc[two]
                poc_answers.append(f"['Our name is {name_one} and {name_two}']")
            for i, row in list(df_names.iterrows()):     
                name = row["First Name"]+" "+row["Last Name"]
                if row["Gender"] == "Female":
                    poc_questions.append("What is her name?")
                    poc_answers.append(f"['Her name is {name}']")
            for i, row in list(df_names.iterrows()):
                name = row["First Name"]+" "+row["Last Name"]
                if row["Gender"] == "Male":
                    poc_questions.append("What is his name?")
                    poc_answers.append(f"['His name is {name}']")
            df_train = pd.DataFrame(list(zip(poc_questions, poc_answers)),
                           columns =['question', 'answer'])
            df_train = shuffle(df_train)
            df_train.to_csv('poc_data.csv')
        # load data
        df_train = pd.read_csv('poc_data.csv')

        
    return df_train
    

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


def questions_answers(source_name="squad1"):
    df_train = load_df(source_name=source_name)
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


# def tokenize_question(questions, vocab):
#     '''
#     no vocab increase!!!

#     Inputs: questions, list
#             vocab, dictionary with words
#     Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens

#     '''
#     new_questions = []

#     for i, question in enumerate(questions):
#         new_question = []
#         new_question = [vocab.words["<SOS>"]] + new_question + [vocab.words["<EOS>"]] #+ [vocab.words["<PAD>"]]*padding_size
#         new_questions.append(torch.LongTensor(new_question))
#     return new_questions


        

    
