import numpy as np
import pandas as pd
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

model = load_model('models/initial_model')

text = (open("plots.txt").read())
text=text.lower()
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length-seq_length, 1):
    sequence = text[i:(i + seq_length)]
    label =text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

string_mapped = X[30]
full_string = [n_to_char[value] for value in string_mapped]
print(full_string)
# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))


    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]
txt=""
for char in full_string:
    txt = txt+char
print(txt)
