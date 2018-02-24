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
    modulo = 10
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
print('id for MERGE', w2i_target['#MERGE#'])

num_of_words_source = len(w2i_source)
print("Source vocabulary size: {}".format(num_of_words_source))
num_of_words_target = len(w2i_target)
print("Target vocabulary size: {}".format(num_of_words_target))

dev = list(read_dataset(dev_source_file, dev_target_file))
#test = list(read_dataset(test_source_file, test_target_file))
test = list(read_dataset(train_source_file, train_target_file))

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

MAX_SENT_SIZE = get_max_sent_size(train)
AVG_SENT_SIZE = get_avg_sent_size(train)

print("train max length: " + str(MAX_SENT_SIZE))
print("train avg length: " + str(AVG_SENT_SIZE))

MAX_SENT_SIZE_DEV = get_max_sent_size(dev)
AVG_SENT_SIZE = get_avg_sent_size(dev)
print("dev max length: " + str(MAX_SENT_SIZE_DEV))
print("dev avg length: " + str(AVG_SENT_SIZE))

MAX_SENT_SIZE_TEST = get_max_sent_size(test)
AVG_SENT_SIZE = get_avg_sent_size(test)
print("test max length: " + str(MAX_SENT_SIZE_TEST))
print("test avg length: " + str(AVG_SENT_SIZE))

MAX_SENT_SIZE = 100
# start Dynet and define trainer
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

NUM_LAYERS = 1
EMBED_SIZE = 128
HIDDEN_SIZE = 128
ATTENTION_SIZE = 128
BATCH_SIZE = 16

# Lookup parameters for word embeddings -  creates a embedding matrix
LOOKUP_SOURCE = model.add_lookup_parameters((num_of_words_source, EMBED_SIZE))
LOOKUP_TARGET = model.add_lookup_parameters((num_of_words_target, EMBED_SIZE))

#trying dropout
#expression = LOOKUP_SOURCE[5]
#dy.dropout(expression, 0.3)

# "Word"-level LSTM
SOURCE_LSTM_BUILDER = dy.VanillaLSTMBuilder(NUM_LAYERS, EMBED_SIZE, HIDDEN_SIZE, model)
TARGET_LSTM_BUILDER = dy.VanillaLSTMBuilder(NUM_LAYERS, EMBED_SIZE, HIDDEN_SIZE, model)

W_softmax = model.add_parameters((num_of_words_target, HIDDEN_SIZE))
bias_softmax = model.add_parameters((num_of_words_target))

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

    current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, dy.tanh(src_output)])
    #current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, src_output])
    # actiavtion over the output 
    prev_word = trg[0]
    W_sm = dy.parameter(W_softmax)
    b_sm = dy.parameter(bias_softmax)

    for t, next_word in enumerate(trg[1:]):
        #feed the current state into the 
        current_state = current_state.add_input(LOOKUP_TARGET[prev_word])
        #hidden_state = src_output.add_inputs(current_state)
        #print('hidden_state', hidden_state.output())
        output_embedding = current_state.output()

        s = dy.affine_transform([b_sm, W_sm, output_embedding])
        item_loss = dy.pickneglogsoftmax(s, next_word)
        all_losses.append(item_loss)
        # if t < 4: 
        #     print('type of s', type(s))
        #     prob = dy.softmax(s)
        #     print('sum of prob', dy.sum_elems(prob).value())
        #     print('Prob of correct word', prob.value()[next_word])
        #     print('Prob of merge', prob.value()[3])
        #     print('word at %d, loss=%.4f ' % (t, item_loss.value()))

        prev_word = next_word
    # print('sum(loss) = ', dy.esum(all_losses).value(), 'len(trg)', len(trg[1:]), 'avg loss', dy.esum(all_losses).value() / len(trg[1:]))

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
    #current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, dy.tanh(src_output)])

    current_state = TARGET_LSTM_BUILDER.initial_state().set_s([src_output, src_output])
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
        next_word = np.argmax(probs)

        if next_word == eos_target:
            break
        prev_word = next_word
        trg_sent.append(i2w_target[next_word])
    return trg_sent

def shuffle_data(train, max_batch_size):
    source = [x[0] for x in train]
    src_lengths = [len(x) for x in source]
    batches = {}
    for i, length in enumerate(src_lengths):
        if length not in batches:
            batches[length] = []
        batches[length].append(i)
    for batch in batches:
        random.shuffle(batches[batch])
    keys = list(batches.keys())
    random.shuffle(keys)
    shuffled_train_list = [] # indexes of the 
    for key in keys:
        list_samples = batches[key]
        for sample in list_samples:
            shuffled_train_list.append(sample)

    shuffle_data = []
    for i in shuffled_train_list:
        shuffle_data.append(train[i])
    return shuffle_data

ITERATION = 3
print("iteration: " + str(ITERATION))
lowest_dev_loss = float("inf")
decreasing_counter = 0
for ITER in range(ITERATION):
  # Perform training # sorting based on the length of training data
  #train.sort(key=lambda t: len(t[0]), reverse=True)
  #dev.sort(key=lambda t: len(t[0]), reverse=True)
  #train_order = create_batches(train, BATCH_SIZE) 
  #dev_order = create_batches(dev, BATCH_SIZE)

  #print (len(train))
  #shuffled_train = shuffle_data(train, BATCH_SIZE)
  #print (len(train))
  random.shuffle(train)

  if ITER == 10:
    print('something')
    print('norm of gradient of parameters, norm of the parameters')

  train_words, train_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(train):#shuffled_train):
    my_loss = calc_loss(sent)
    # if sent_id > 10:
    #     exit(0)
    # print('loss ', my_loss.value())
    # print('number of words', len(sent))

    train_loss += my_loss.value()
    train_words += len(sent[1][1:]) # sent (src, trg), trg[1:] <s>
    # if sent_id > 50:
    #     exit(0)
    # print('loss = ', train_loss / train_words)

    my_loss.backward()
    trainer.update()
    if (sent_id+1) % modulo == 0:
      print("--finished %r sentences" % (sent_id+1))
      print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
  print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

  # Evaluate on dev set
  dev_words, dev_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(dev):
    my_loss = calc_loss(sent)
    dev_loss += my_loss.value()
    dev_words += len(sent)
    trainer.update()
  dev_loss_per_word = dev_loss/dev_words
  print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, dev_loss_per_word, math.exp(dev_loss/dev_words), time.time()-start))
  if lowest_dev_loss > dev_loss_per_word:
  	lowest_dev_loss = dev_loss_per_word
  else:
    print("old learning rate: " + str(trainer.learning_rate))
    trainer.learning_rate = trainer.learning_rate/2.0
    print("new learning rate: " + str(trainer.learning_rate))
    decreasing_counter += 1

  if decreasing_counter == 5:
    exit(0)



#generate parse tree
writer = open("output_train.txt", 'w')
sentences = []
for sent_id, sent in enumerate(test):
    translated_sent = generate(sent[0])
    sentences.append(translated_sent)
for sent in sentences:
    writer.write(str(sent)+"\n")
writer.close()