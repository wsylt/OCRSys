import Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filename = 'params.txt'
parameters = open(filename, encoding='utf-8').read().strip().split('\n')
n_letters = int(parameters[0])
n_hidden = int(parameters[1])
n_categories = int(parameters[2])
n_letters = int(parameters[3])
all_categories = parameters[4].split(' ')
all_letters = parameters[5]

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


cls_rnn = RNN(n_letters, n_hidden, n_categories)
cls_rnn.load_state_dict(torch.load('net_params.pkl', map_location = 'cpu'))
cls_rnn.eval()

import os.path as op
import numpy as np
import json
import h5py
import codecs
import time
import sys
import rnnseg.pretrain as cws
import warnings
from rnnseg.rnn_cws import loadModel
from rnnseg.utils import viterbi
from sklearn import model_selection
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from gensim.models import Word2Vec

cws_info_file = op.dirname(op.abspath(__file__))+"/rnnseg/cws.info"
keras_model_file = op.dirname(op.abspath(__file__))+"/rnnseg/cws_keras_model"
keras_model_weights_file = op.dirname(op.abspath(__file__))+"/rnnseg/keras_model_weights"
cwsInfo = cws.loadCwsInfo(cws_info_file)
segmodel = loadModel(keras_model_file, keras_model_weights_file)

#------------------------copy everything above to the main part--------------------------------

#cls_rnn = torch.load('model.pkl')
#checking params

sample1 = 'Gasket-NC T=25mmPN20 gasket-NC'
sample2 = '1 Gasket 2 GASKET,PN20(Class 150), Flat ring, RF, 1.5 mm(1/16)thick, ASME B16.21 7551FD01 pc 20 32.06 641.2 7551FG01'
sample3 = '1 8402078 Fibre gasket F14, Aramid fiber with nitrile binder, RF, B16,5, NPS3/4,CLASS150, t=0.0625in27*57*1.5875 0.75 片 14 2.83 39.62'


out1 = Segmentation.segment([sample1], "./dic.json", cls_rnn, segmodel, cwsInfo)
print(out1)
out2 = Segmentation.segment([sample2], "./dic.json", cls_rnn, segmodel, cwsInfo)
print(out2)
out3 = Segmentation.segment([sample3], "./dic.json", cls_rnn, segmodel, cwsInfo)
print(out3)
