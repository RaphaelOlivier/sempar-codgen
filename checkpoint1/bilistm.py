from __future__ import print_function
import time
import random
import math
import argparse

from collections import Counter, defaultdict

import dynet as dy
import numpy as np

start = time.time()

# description
w2i_source = defaultdict(lambda: len(w2i_source))

# code
w2i_target = defaultdict(lambda: len(w2i_target))

# read data set
# input file format is "word1 word2 ..."
def read_dataset(source_file, target_file):
    with open(source_file, "r", encoding='utf-8', errors='ignore') as s_file, open(target_file, "r", encoding='utf-8', errors='ignore') as t_file:
        for source_line, target_line in zip(s_file, t_file):
            sent_src = [w2i_source[x] for x in source_line.strip().split(" ") + ['</s>']]
            sent_trg = [w2i_target[x] for x in ['<s>'] + target_line.strip().split(" ") + ['</s>']] 
            yield (sent_src, sent_trg)

# -------------------------
# read training data
# -------------------------
use_django = False

if use_django:
    modulo = 1000
    print("Using Django dataset...")
    train_source_file = "django_dataset/django.train.desc"
    train_target_file = "django_dataset/django.train.code"
    dev_source_file = "django_dataset/django.dev.desc"
    dev_target_file = "django_dataset/django.dev.code"
    test_source_file = "django_dataset/django.test.desc"
    test_target_file = "django_dataset/django.test.code"
else:
    modulo = 100
    print("Using HS dataset...")
    train_source_file = "hs_dataset/hs.train.desc"
    train_target_file = "hs_dataset/hs.train.code"
    dev_source_file = "hs_dataset/hs.dev.desc"
    dev_target_file = "hs_dataset/hs.dev.code"
    test_source_file = "hs_dataset/hs.test.desc"
    test_target_file = "hs_dataset/hs.test.code"

read_dataset(train_source_file, train_target_file)
train = list(read_dataset(train_source_file, train_target_file))
#print (train[0])
unk_source = w2i_source["<unk>"]
eos_source = w2i_source['</s>']
w2i_source = defaultdict(lambda: unk_source, w2i_source)
unk_target = w2i_target["<unk>"]
eos_target = w2i_target['</s>']
sos_target = w2i_target['<s>']
w2i_target = defaultdict(lambda: unk_target, w2i_target)
i2w_target = {v: k for k, v in w2i_target.items()}

source_vocab = len(w2i_source)
print("Source vocabulary size: {}".format(source_vocab))
target_vocab = len(w2i_target)
print("Target vocabulary size: {}".format(target_vocab))

dev = list(read_dataset(dev_source_file, dev_target_file))
test = list(read_dataset(test_source_file, test_target_file))

def get_max_sent_size(train):
    max_size = 0
    for each_instance in train:
        sent_length = len(each_instance[1])
        if(sent_length > max_size):
            max_size = sent_length
        #print(str(each_instance[1]) + " " + str(sent_length) + " " + str(max_size))
    return max_size

def get_avg_sent_size(train):
    total = 0.0
    for each_instance in train:
        sent_length = len(each_instance[1])
        total += sent_length
        #print(str(each_instance[1]) + " " + str(sent_length) + " " + str(max_size))
    avg = total/len(train)
    return avg

max_sent_size = get_max_sent_size(train)
avg_sent_size = get_avg_sent_size(train)

print("train max length: " + str(max_sent_size))
print("train avg length: " + str(avg_sent_size))

max_sent_size_dev = get_max_sent_size(dev)
avg_sent_size  = get_avg_sent_size(dev)
print("dev max length: " + str(max_sent_size_dev))
print("dev avg length: " + str(avg_sent_size))

max_sent_size_test = get_max_sent_size(test)
avg_sent_size  = get_avg_sent_size(test)
print("test max length: " + str(max_sent_size_test))
print("test avg length: " + str(avg_sent_size))

max_sent_size = 100
# start Dynet and define trainer
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

num_layer = 1
embedding_size = 128
hidden_size = 128

# Create word embedding matrix
source_lookup = model.add_lookup_parameters((source_vocab, embedding_size))
target_lookup = model.add_lookup_parameters((target_vocab, embedding_size))

w_softmax = model.add_parameters((target_vocab, hidden_size))
b_softmax = model.add_parameters((target_vocab))

# LSTM
forward_encoder = dy.LSTMBuilder(num_layer, embedding_size, hidden_size, model)
backward_encoder = dy.LSTMBuilder(num_layer, embedding_size, hidden_size, model)

def embed(sentence):
    return [source_lookup[x] for x in sentence]

def lstm(initial_state, input_vectors):
    state = initial_state
    output_vectors = []
    #print(input_vectors[0].value())
    #exit(0)
    for vector in input_vectors:
        state = state.add_input(vector)
        output_vector = state.output()
        output_vectors.append(output_vector)
    return output_vectors

def encode(forward_encoder, backward_encoder, sentence):
    reversed_sentence = list(reversed(sentence))

    forward_vectors = lstm(forward_encoder.initial_state(), sentence)
    b_vectors = lstm(backward_encoder.initial_state(), reversed_sentence)
    backward_vectors = list(reversed(b_vectors))
    output_vectors = [dy.concatenate(list(x)) for x in zip(forward_vectors, backward_vectors)]
    return output_vectors

def decode(decoder, vectors, output):
    w = dy.parameter(w_softmax)
    b = dy.parameter(b_softmax)

embedded_sent = embed(train[0][1])
#print(embedded_sent)
encode(forward_encoder, backward_encoder, embedded_sent)