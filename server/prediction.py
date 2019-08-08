#-*- coding: utf-8 -*-
import sys
import torch
# from classification import networkPara
import torch.nn as nn
import torch.nn.functional as F
from data_helper import lineToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename = 'params.txt'
parameters = open(filename, encoding='utf-8').read().strip().split('\n')
n_letters = int(parameters[0])
n_hidden = int(parameters[1])
n_categories = int(parameters[2])
n_letters = int(parameters[3])
all_categories = parameters[4].split(' ')
all_letters = parameters[5]

# Just return an output given a line
def evaluate(rnn, line_tensor):
    rnn.hidden = rnn.init_hidden()
    ###############
    ####.cuda()####
    ###############
    output = rnn(line_tensor)
    return output


# Running on the User input
def predict(input_line, rnn, n_predictions=1):
    # print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(rnn, lineToTensor(input_line, n_letters, all_letters))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        value = topv[0][0].item()
        category_index = topi[0][0].item()
        return all_categories[category_index]
        # for i in range(n_predictions):
        #     value = topv[0][i].item()
        #     category_index = topi[0][i].item()
        #     print('(%.2f) %s' % (value, all_categories[category_index]))
        #     predictions.append([value, all_categories[category_index]])
            
class RNN(nn.Module):
    """LSTM class"""
    def __init__(self, input_size, hidden_size, output_size):
        '''
        :param input_size: number of input coming in
        :param hidden_size: number of he hidden units
        :param output_size: size of the output
        '''
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        #LSTM
        self.lstm = nn.LSTM(input_size, hidden_size).to(device)
        self.hidden2Cat = nn.Linear(hidden_size, output_size).to(device)
        self.hidden = self.init_hidden()

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        output = self.hidden2Cat(lstm_out[-1]) #many to one
        output = F.log_softmax(output, dim=1)

        return output

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size).to(device),
                torch.zeros(1, 1, self.hidden_size).to(device))


# rnn = RNN(n_letters, n_hidden, n_categories) #LSTM model
# rnn.load_state_dict(torch.load('net_params.pkl'))
#rnn = torch.load('model.pkl')
#predict(' '.join(sys.argv[1:]))
#predict(sys.argv[1])