import torch
def apply_chatbot(model, question, vocab_source, vocab_target, max_length):
    """
    renders predictions untill the end of sequence or 15...
    """
    try:
        max_length = len(answer)
    except:
        max_length
    src = question
    dummy = [0]*max_length
    trg = torch.LongTensor(dummy)

    predicted_indices=model(src, trg)

    predicted_words = [vocab_target.index2word[str(torch.argmax(a,dim=0).item())] for a in predicted_indices]
    truncated_sequence =[]
    for word in predicted_words:
        truncated_sequence.append(word)
        if word == "<EOS>":
            break
    predicted_answer =  " ".join(truncated_sequence)
    question = " ".join([vocab_source.index2word[str(a.item())] for a in question[0]])
    print("question:", question)
#     if answer != None:
#         answer = " ".join([vocab_target.index2word[str(a.item())] for a in answer])
#         print("answer:",answer)
    print("predicted_answer:",predicted_answer)
    print("")
