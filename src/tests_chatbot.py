import pytest
import torch
from models_chatbot import Seq2Seq


@pytest.fixture()
def question():
    question = torch.LongTensor([1,2,3])
    return question

@pytest.fixture()
def answer():
    answer = torch.LongTensor([2,4,6])
    return answer

def test_seq2seq_forward(question, answer):
    # test training mode
    model = Seq2Seq(5, 3, 8, dropout_E=0, dropout_D=0, teacher_forcing_ratio=1)
    prediction = model(question, answer)
    assert len(prediction) == len(answer), f"Length of prediction does not match length of answer: {len(prediction)} vs. {len(answer)}"
    assert len(prediction.shape) == len(answer.shape)+1, f"Number of dimensions of prediction should exceed the number of dimensions of answer by one, but prediction has dim {prediction.shape} and answer has dim {answer.shape}"
    # test evaluation mode
    model.eval()
    prediction = model(question, answer)
    assert len(prediction) == len(answer), f"Length of prediction does not match length of answer: {len(prediction)} vs. {len(answer)}"
    assert len(prediction.shape) == len(answer.shape)+1, f"Number of dimensions of prediction should exceed the number of dimensions of answer by one, but prediction has dim {prediction.shape} and answer has dim {answer.shape}"

