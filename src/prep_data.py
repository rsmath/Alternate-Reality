"""
This module will prepare the text data for the RNN to use.
"""

from tensorflow import one_hot
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import io


path = "../text_data/merged.txt"

with io.open(path, encoding='utf-8') as corpus:
    text = corpus.read().lower()

print('Corpus length:', len(text))


