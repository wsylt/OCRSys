from PIL import Image
from flask_cors import CORS
from flask import Flask, request
from io import BytesIO
import base64
import requests
import hashlib
import urllib.parse 
import json
import time
import segmentation

### load module
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

###

def md5(str):
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

app = Flask(__name__)
CORS(app, supports_credentials=True)
 
@app.route('/api/ocr', methods=['GET','POST'])
def ocr():
    img = request.json.get('img')
    if (img == ''):
        return 'null'
    img = img.partition(',')[-1]

    url = "http://deepi.sogou.com:80/api/sogouService"
    pid = "7f37f594e2b22c13ee73c656ca6105dd"
    service = "basicOpenOcr"
    salt = str(time.time())
    SecretKey = "03ee22d82c65dc45435da716bce42994"
    # base64 string file picture,too long in the case we will omit string
    imageShort = img[0:1024]
    sign = md5(pid+service+salt+imageShort+SecretKey);
    payload = "lang=zh-CHS&pid=" + pid + "&service=" + service + "&sign=" + sign + "&salt=" + salt + "&image=" + urllib.parse.quote(img)
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'accept': "application/json"
        }
    response = requests.request("POST", url, data=payload, headers=headers)
    #print(response.text)
    return (response.text)

@app.route('/api/segment', methods=['GET','POST'])
def seg():
    return segmentation.segment(request.json.get('data'), './dic.json', cls_rnn, segmodel, cwsInfo)

    # dic = json.loads(response.text)
    # txt = ''
    # for i in dic['result']:
    # 	txt += i['content']
    # #return txt
    # #print (json.dumps(dic, indent=2))
    # #return json.dumps(dic, indent=2)

if __name__ == '__main__':
    app.run(port=5000)
