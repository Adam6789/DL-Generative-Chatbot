import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def pretrain(model, vQ, vA, w2v):
    
    hidden_size = len(list(model.encoder.parameters())[0][1])
    
    weights_matrix = list(model.encoder.parameters())[0].detach().numpy()
    words_found = 0
    # known_words = []
    # unknown_words = []
    for i, word in enumerate(vQ.words):
        try: 
            weights_matrix[i] = w2v.wv[word]
            words_found += 1
    #         known_words.append(word)
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(hidden_size, ))
    #         unknown_words.append((word,i))
    print(f"For {words_found} of {len(vQ.words)} words an entry has been found in the brown corpus.")
    weights_matrix = torch.from_numpy(weights_matrix)
    model.encoder.embedding.load_state_dict({'weight':weights_matrix})
    
    # DECODER
    
    weights_matrix = list(model.decoder.parameters())[0].detach().numpy()
    words_found = 0
    # known_words = []
    # unknown_words = []
    for i, word in enumerate(vA.words):
        try: 
            weights_matrix[i] = w2v.wv[word]
            words_found += 1
    #         known_words.append(word)
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(hidden_size, ))
    #         unknown_words.append((word,i))
    print(f"For {words_found} of {len(vA.words)} words an entry has been found in the brown corpus.")
    weights_matrix = torch.from_numpy(weights_matrix)
    model.decoder.embedding.load_state_dict({'weight':weights_matrix})
    return model


def train(epochs, batch_size, print_each, lr, model, version, questions, answers, vQ, vA):  

    
    
    if Path(f"model_{version}.pt").is_file():
        model.load_state_dict(torch.load(f"model_{version}.pt", map_location=torch.device('cpu')))
        print(f"Loading from checkpoint: 'model_{version}.pt'")
    else:
        print(f"Nothing to load at checkpoint: 'model_{version}.pt'")
        
    model.to(device) 
    print(f"Computing on {device}.\n")
    
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss(reduction='sum')
    
    epoch = 0
    for epoch in range(1,epochs+1):     
        train_loss = 0
        valid_loss = 0
        Q_batches, A_batches = heteroDataLoader(questions, answers, batch_size)
        for i, (batch_q, batch_a) in enumerate(zip(Q_batches[:-5], A_batches[:-5]),1):   
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

       

    
        

        
        for n, (batch_q, batch_a) in enumerate(zip(Q_batches[-5:], A_batches[-5:])):     
            # evaluation loop
            model.eval()
            #batch_loss = 0
            for q, a in zip(batch_q, batch_a):      
                output = model(q,a)
                try:
                    loss = loss_fn(output.squeeze(),a.squeeze())
                except:

                    print("could not be computed for:", q, a, output)
                    print(q.shape, a. shape, output.shape)
                #batch_loss += loss
                valid_loss += loss / a.size(0)

            #valid_loss += batch_loss / batch_size

        if epoch % print_each == 0:
            batches = len(questions) // batch_size
            valid_loss = round(valid_loss.item() / (5 * batch_size * print_each),3)
            train_loss = round(train_loss.item() / ((batches - 5) * batch_size * print_each),3)    
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


            torch.save(model.state_dict(),f"model_{version}.pt")
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
                
                
                      
    
    
                