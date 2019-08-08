#-*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import os
import glob
import time
import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_helper import readFiles, lineToTensor, unicodeToAscii


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line, n_letters, all_letters)
    return category, line, category_tensor, line_tensor



#creating the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    
# Training the network
def trainNetwork():
    criterion = nn.NLLLoss()
    learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories) #LSTM model
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    def train(category_tensor, line_tensor):
        rnn.zero_grad()
        rnn.hidden = rnn.init_hidden()

        output = rnn(line_tensor)[-1]

        loss = criterion(output.unsqueeze(0), category_tensor)
        loss.backward()

        optimizer.step()

        return output.unsqueeze(0), loss.item()

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    n_iters = 250000
    print_every = 5000
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor.to(device), line_tensor.to(device))

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    torch.save(rnn, 'model.pkl')
    torch.save(rnn.state_dict(), 'net_params.pkl')



if __name__ == '__main__':
    filenames = glob.glob('data/types/*.txt')
    n_hidden = 128
    category_lines, all_categories, n_categories, n_letters, all_letters= readFiles(filenames)
    with open("params.txt",'w',encoding='utf-8') as file:
        file.write('\n'.join([str(n_letters), str(n_hidden), str(n_categories), str(n_letters), str(' '.join(all_categories)), all_letters]))

    trainNetwork()