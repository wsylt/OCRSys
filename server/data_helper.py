#-*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import os
import torch
import string
import unicodedata

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s,allLetters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allLetters
    )


def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    for line in lines:
        all_letters += letterCheck(all_letters, line)
        
    return [unicodeToAscii(line, all_letters) for line in lines], all_letters


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, n_letters, all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# all_letters = string.ascii_letters + " .,;'/()*-#"

def letterCheck(all_letters, s):
    letters = all_letters
    temp = ''
    for c in unicodedata.normalize('NFD', s):
        if not c in letters:
            letters = letters + c
            temp = temp+c
    return temp


def readFiles(filenames):
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    all_letters = string.ascii_letters+string.digits+'@#$%&*+-=\/|'
    for filename in filenames:
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines, all_letters = readLines(filename, all_letters)
        category_lines[category] = lines
    
    n_categories = len(all_categories)
    n_letters = len(all_letters)
    
    return category_lines, all_categories, n_categories, n_letters, all_letters

