import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, batch_size, print_each, lr, model, version, questions, answers, vQ, vA):  
    validation_batches = 5

    
    
    if Path(f"models/model_{version}.pt").is_file():
        model.load_state_dict(torch.load(f"models/model_{version}.pt", map_location=torch.device('cpu')))
        print(f"Loading from checkpoint: 'models/model_{version}.pt'")
    else:
        print(f"Nothing to load at checkpoint: 'models/model_{version}.pt'")
        
    model.to(device) 
    print(f"Computing on {device}.\n")
    
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss(reduction='sum')
    
    epoch = 0
    for epoch in range(1,epochs+1):     
        train_loss = 0
        valid_loss = 0
        Q_batches, A_batches = heteroDataLoader(questions, answers, batch_size)
        for i, (batch_q, batch_a) in enumerate(zip(Q_batches[:-validation_batches], A_batches[:-validation_batches]),1):   
            model.train()
            batch_loss = 0
    
            for m, (q, a) in enumerate(zip(batch_q, batch_a)):  
                output = model(q,a)
                loss = loss_fn(output.squeeze(),a.squeeze())
                batch_loss += loss
                train_loss += loss / a.size(0)

  
            batch_loss.backward()
            optim.step()
            optim.zero_grad()
    
        for n, (batch_q, batch_a) in enumerate(zip(Q_batches[-validation_batches:], A_batches[-validation_batches:])):     
            model.eval()
            for q, a in zip(batch_q, batch_a):      
                output = model(q,a)
                loss = loss_fn(output.squeeze(),a.squeeze())
                valid_loss += loss / a.size(0)


        if epoch % print_each == 0:
            batches = len(questions) // batch_size
            valid_loss = round(valid_loss.item() / (validation_batches * batch_size * print_each),3)
            train_loss = round(train_loss.item() / ((batches - validation_batches) * batch_size * print_each),3)    
            print(f"epoch: {epoch}/{epochs}", "\ttrain_loss:",train_loss, "\tvalid_loss", valid_loss)

           

            randint = random.randint(1, len(questions))
            question = questions[randint-1]
            answer = answers[randint-1]
            prediction = model(question, answer)
            text = ""

            for x in question:
                text += vQ.index2word[str(x.item())] + " "
            print("question:",text)
            text = ""
            for x in answer:
                text += vA.index2word[str(x.item())] + " "
            print("answer:", text)
            text = ""
            for x in prediction:
                text += vA.index2word[str(torch.argmax(x,dim=0).item())] + " "
            print("prediction:", text, "\n")


            torch.save(model.state_dict(),f"models/model_{version}.pt")
            train_loss = 0
            valid_loss = 0
        
                
def heteroDataLoader(single_questions, single_answers, batch_size, shuffle = True):

    """
    Inputs:
    -------
    dataset: list
        A list of single samples.
    Outputs:
    --------
    batches: list
        A list of lists with each having multiple samples.
    """
    len_batches = len(single_questions) // batch_size
    indices = list(range(0,len(single_questions)))
    
    temp = list(zip(single_questions, single_answers))
    random.shuffle(temp)  
    single_questions, single_answers = zip(*temp)
    single_questions, single_answers = list(single_questions), list(single_answers)           
    
                   
    
    random.shuffle(indices)
    Q_batches = []
    A_batches = []
        
    for i in range(len_batches):
        Q_batches.append(single_questions[i*batch_size:(i+1)*batch_size])
        A_batches.append(single_answers[i*batch_size:(i+1)*batch_size])
    return Q_batches, A_batches
                
                
                      
    
    
                