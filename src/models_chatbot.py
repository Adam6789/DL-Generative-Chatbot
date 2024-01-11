import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout=0):
        
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, i, h):
        
        '''
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state (actually a tuple of hidden state and cell state)
        '''
        embedding = self.embedding(i)
        x,y = h
        o, h= self.lstm(embedding, h)
        o = self.dropout(o)
        
        return o, h
    

class Decoder(nn.Module):
      
    def __init__(self, hidden_size, output_size, dropout):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        

        self.lstm = nn.LSTM(hidden_size, hidden_size)
  
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(p=dropout)
       
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, i, h):
        
        '''
        Inputs: i, the target vector
        Outputs: o, the decoder output
                h, the hidden state (actually a tuple of hidden state and cell state)
        '''

        embedding = self.embedding(i)

        o, h = self.lstm(embedding, h)

        o = self.linear(o)
        
        o = self.dropout(o)

        o = self.softmax(o)
       
        return o, h
        
        

class Seq2Seq(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, dropout_E=0, dropout_D=0, teacher_forcing_ratio=1):
        
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, dropout_E)
        self.decoder = Decoder(hidden_size, output_size, dropout_D)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size

    def forward(self, src, trg): 
        '''
        Inputs: src, the source vector
                trg, the target vector
        Outputs: o, the prediction
                
        '''
        
        src.to(device)
        trg.to(device)
        start = torch.LongTensor([[0]]).to(device)

        # encoder
        hidden = (torch.zeros(1,self.hidden_size).to(device), torch.zeros(1,self.hidden_size).to(device))
        for word in src:
            o, hidden = self.encoder(word.view(-1), hidden)
            x,y = hidden
   
        # decoder
        o = start
        prediction = []
        for word in trg:
            o, hidden = self.decoder(o.view(-1), hidden)
            x,y = hidden
            
            prediction.append(o)
            
            if self.training:
                o = word if random.random() < self.teacher_forcing_ratio else torch.argmax(o,dim=1)
            else:
                o = torch.argmax(o,dim=1) 
                o.detach()

        prediction = torch.stack(prediction)
        prediction = prediction

        return prediction.squeeze()

    

