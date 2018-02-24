from __future__ import print_function
import time
import random
import math
import sys
import argparse

start = time.time()

from collections import Counter, defaultdict

import dynet as dy
import numpy as np

# natural language query
w2i_source = defaultdict(lambda: len(w2i_source))

# code parse tree
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
use_hs = True
if use_django:
    train_source_file = "django_dataset/query_train_.txt"
    train_target_file = "django_dataset/tree_train_.txt"

    #train_source_file = "django_dataset/query_dev_.txt"
    #train_target_file = "django_dataset/tree_dev_.txt"
    dev_source_file = "django_dataset/query_dev_.txt"
    dev_target_file = "django_dataset/tree_dev_.txt"
    test_source_file = "django_dataset/query_test_.txt"
    test_target_file = "django_dataset/tree_test_.txt"

if use_hs:
    train_source_file = "hs_dataset/query_hs_train_.txt"
    train_target_file = "hs_dataset/tree_hs_train_.txt"

    dev_source_file = "hs_dataset/query_hs_dev_.txt"
    dev_target_file = "hs_dataset/tree_hs_dev_.txt"
    test_source_file = "hs_dataset/query_hs_test_.txt"
    test_target_file = "hs_dataset/tree_hs_test_.txt"

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

num_of_words_source = len(w2i_source)
print("Source vocabulary size: {}".format(num_of_words_source))
num_of_words_target = len(w2i_target)
print("Target vocabulary size: {}".format(num_of_words_target))

dev = list(read_dataset(dev_source_file, dev_target_file))
test = list(read_dataset(test_source_file, test_target_file))
# start Dynet and define trainer

model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

NUM_LAYERS = 1
EMBED_SIZE = 64
HIDDEN_SIZE = 256
BATCH_SIZE = 16

# Lookup parameters for word embeddings -  creates a embedding matrix
LOOKUP_SOURCE = model.add_lookup_parameters((num_of_words_source, EMBED_SIZE))
LOOKUP_TARGET = model.add_lookup_parameters((num_of_words_target, EMBED_SIZE))

# "Word"-level LSTM
SOURCE_LSTM_BUILDER = dy.VanillaLSTMBuilder(NUM_LAYERS, EMBED_SIZE, HIDDEN_SIZE, model)
TARGET_LSTM_BUILDER = dy.VanillaLSTMBuilder(NUM_LAYERS, EMBED_SIZE, HIDDEN_SIZE, model)

W_softmax = model.add_parameters((num_of_words_target, HIDDEN_SIZE))
bias_softmax = model.add_parameters((num_of_words_target))

MAX_SENT_SIZE = 50

# Creates batches where all source sentences are the same length 
def create_batches(sorted_dataset, max_batch_size):
    source = [x[0] for x in sorted_dataset]
    src_lengths = [len(x) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batches.append((prev_start, batch_size)) # creates a batch when a different length sentence is obtained
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    return batches

def calc_loss(sent):
    dy.renew_cg()

    # Transduce all batch elements with an LSTM
    src = sent[0]
    trg = sent[1]


    #initialize the LSTM
    init_state_src = SOURCE_LSTM_BUILDER.initial_state()

    #get the output of the first LSTM
    src_output = init_state_src.add_inputs([LOOKUP_SOURCE[x] for x in src])[-1].output()
    # bascially - forward prop  - encoder output 
    #now step through the output sentence
    all_losses = []

    #current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, dy.tanh(src_output)])
    current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, src_output])
    # actiavtion over the output 
    prev_word = trg[0]
    W_sm = dy.parameter(W_softmax)
    b_sm = dy.parameter(bias_softmax)

    for next_word in trg[1:]:
        #feed the current state into the 
        current_state = current_state.add_input(LOOKUP_TARGET[prev_word])
        output_embedding = current_state.output()

        s = dy.affine_transform([b_sm, W_sm, output_embedding])
        all_losses.append(dy.pickneglogsoftmax(s, next_word))

        prev_word = next_word
    return dy.esum(all_losses)

def generate(sent):
    dy.renew_cg()

    # Transduce all batch elements with an LSTM
    #sent_reps = [LSTM_SRC_BUILDER.transduce([LOOKUP_SRC[x] for x in src])[-1] for src in sent]

    dy.renew_cg()

    # Transduce all batch elements with an LSTM
    src = sent


    #initialize the LSTM
    init_state_src = SOURCE_LSTM_BUILDER.initial_state()

    #get the output of the first LSTM # gets the embedding of every word  - 
    src_output = init_state_src.add_inputs([LOOKUP_SOURCE[x] for x in src])[-1].output()

    #generate until a eos tag or max is reached
    current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, dy.tanh(src_output)])

    prev_word = sos_target
    trg_sent = []
    W_sm = dy.parameter(W_softmax)
    b_sm = dy.parameter(bias_softmax)

    for i in range(MAX_SENT_SIZE):
        #feed the previous word into the lstm, calculate the most likely word, add it to the sentence
        current_state = current_state.add_input(LOOKUP_TARGET[prev_word])
        output_embedding = current_state.output()
        s = dy.affine_transform([b_sm, W_sm, output_embedding])
        #print("dy val" + str(dy.log_softmax(s).value()))
        probs = dy.log_softmax(s).value()
        next_word = np.argmin(probs)

        if next_word == eos_target:
            break
        prev_word = next_word
        trg_sent.append(i2w_target[next_word])
    return trg_sent

def shuffle_data(train, max_batch_size):
    source = [x[0] for x in train]
    src_lengths = [len(x) for x in source]
    batches = {}
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batch =  [x for x in range(prev_start, prev_start+batch_size)]
            batches[prev] = batch # creates a batch when a different length sentence is obtained
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    for batch in batches:
        random.shuffle(batches[batch])
    keys = list(batches.keys())
    random.shuffle(keys)
    shuffled_train_list = [] # indexes of the samples
    for key in keys:
        list_samples = batches[key]
        for sample in list_samples:
            shuffled_train_list.append(sample)

    shuffle_data = []
    for i in shuffled_train_list:
        shuffle_data.append(train[i])
    return shuffle_data


ITERATION = 20
print("iteration: " + str(ITERATION))
for ITER in range(ITERATION):
  # Perform training # sorting based on the length of training data
  train.sort(key=lambda t: len(t[0]), reverse=True)
  dev.sort(key=lambda t: len(t[0]), reverse=True)
  train_order = create_batches(train, BATCH_SIZE) 
  dev_order = create_batches(dev, BATCH_SIZE)
  train = shuffle_data(train, BATCH_SIZE)
  #random.shuffle(train)
  train_words, train_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(train):
    my_loss = calc_loss(sent)
    train_loss += my_loss.value()
    train_words += len(sent)
    my_loss.backward()
    trainer.update()
    if (sent_id+1) % 100 == 0:
      print("--finished %r sentences" % (sent_id+1))
  print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

  # Evaluate on dev set
  dev_words, dev_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(dev):
    my_loss = calc_loss(sent)
    dev_loss += my_loss.value()
    dev_words += len(sent)
    trainer.update()
  print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), time.time()-start))

#generate parse tree
writer = open("output_tree.txt", 'w')
sentences = []
for sent_id, sent in enumerate(test):
    translated_sent = generate(sent[0])
    sentences.append(translated_sent)
for sent in sentences:
    writer.write(sent+"\n")
writer.close()