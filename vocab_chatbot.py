class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.count = 0
        self.index2word = {}
                                       
    def indexWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.count       # word2index
            self.index2word[str(self.count)] = word  # index2word
            self.count += 1
            return True
        else:
            return False
        




    
