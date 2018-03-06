import dynet as dy
import time
import random
import math
import sys

from model import ASTNet

from argparse import ArgumentParser
from collections import Counter, defaultdict


parser = ArgumentParser(description='Baseline Code Generator')
parser.add_argument('--data', type=str, default='hs',
                    help='Dataset to be used')
parser.add_argument('--iter', type=int, default=20,
                    help='Training iteration')

args, _ = parser.parse_known_args()

mode = args.data
ITERATION = args.iter
start = time.time()

# description
sourceIndexer = defaultdict(lambda: len(sourceIndexer))

# code
targetIndexer = defaultdict(lambda: len(targetIndexer))

# read data set
# input file format is "word1 word2 ..."


def read_dataset(source_file, target_file):
    with open(source_file, "r", encoding='utf-8', errors='ignore') as s_file, open(target_file, "r", encoding='utf-8', errors='ignore') as t_file:
        for source_line, target_line in zip(s_file, t_file):
            sent_src = [sourceIndexer[x]
                        for x in source_line.strip().split(" ")] + [eos_source]
            sent_trg = [targetIndexer[x]
                        for x in target_line.strip().split(" ")] + [eos_target]
            yield (sent_src, sent_trg)


# choose dataset
if mode == "django":
    modulo = 1000
    print("Using Django dataset...")
    train_source_file = "../data/django_dataset/django.train.desc"
    train_target_file = "../data/django_dataset/django.train.code"
    dev_source_file = "../data/django_dataset/django.dev.desc"
    dev_target_file = "../data/django_dataset/django.dev.code"
    test_source_file = "../data/django_dataset/django.test.desc"
    test_target_file = "../data/django_dataset/django.test.code"
else:
    modulo = 100
    print("Using HS dataset...")
    train_source_file = "../data/hs_dataset/hs.train.desc"
    train_target_file = "../data/hs_dataset/hs.train.code"
    dev_source_file = "../data/hs_dataset/hs.dev.desc"
    dev_target_file = "../data/hs_dataset/hs.dev.code"
    test_source_file = "../data/hs_dataset/hs.test.desc"
    test_target_file = "../data/hs_dataset/hs.test.code"

# Special words
unk_source = sourceIndexer["<unk>"]
eos_source = sourceIndexer['</s>']
unk_target = targetIndexer["<unk>"]
eos_target = targetIndexer['</s>']

# Reading data
read_dataset(train_source_file, train_target_file)
train_data = list(read_dataset(train_source_file, train_target_file))


sourceIndexer = defaultdict(lambda: unk_source, sourceIndexer)

targetIndexer = defaultdict(lambda: unk_target, targetIndexer)
targetDictionnary = {v: k for k, v in targetIndexer.items()}

vocab_length_source = len(sourceIndexer)
print("Source vocabulary size: {}".format(vocab_length_source))
vocab_length_target = len(targetIndexer)
print("Target vocabulary size: {}".format(vocab_length_target))

dev_data = list(read_dataset(dev_source_file, dev_target_file))
test_data = list(read_dataset(test_source_file, test_target_file))

num_layer = 1
embedding_size = 128
hidden_size = 64
att_size = 32

# start Dynet and define trainer
model = ASTNet(vocab_length_source, vocab_length_target, targetIndexer, targetDictionnary,
               num_layer, embedding_size, hidden_size, att_size)


def train(train_data, log_writer):
    random.shuffle(train_data)
    train_words, train_loss = 0, 0.0
    start = time.time()

    for sent_id, sent in enumerate(train_data):
        input_sent, output_sent = sent[0], sent[1]
        loss = model.forward(input_sent, output_sent, mode="train")
        train_loss += loss.value()
        train_words += len(sent[1][1:])
        model.backward_and_update(loss)

        if (sent_id+1) % modulo == 0:
            print("--finished %r sentences" % (sent_id+1))
            print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
                  (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

            log_writer.write("--finished %r sentences\n" % (sent_id+1))
            log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                             (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                     (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    # Evaluate on development set

    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev_data):
        input_sent, output_sent = sent[0], sent[1]
        loss = model.forward(input_sent, output_sent, mode="validate")
        dev_loss += loss.value()
        dev_words += len(sent[1][1:])

    dev_loss_per_word = dev_loss/dev_words
    dev_perplexity = math.exp(dev_loss/dev_words)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, dev_loss_per_word, dev_perplexity, time.time()-start))

    return dev_loss_per_word, dev_perplexity


# Training
log_writer = open("../exp/log/"+str(ITERATION)+"_iter_"+mode+".log", 'w')

print("iteration: " + str(ITERATION))
lowest_dev_loss = float("inf")
successive_decreasing_counter = 0
lowest_dev_perplexity = float("inf")
for ITER in range(ITERATION):
    # Perform training
    dev_loss_per_word, dev_perplexity = train(train_data, log_writer)
    if lowest_dev_perplexity > dev_perplexity:
        lowest_dev_perplexity = dev_perplexity
        print("----------------------------------")
        print("Saving lowest dev perplexity: " +
              str(lowest_dev_perplexity) + " at iteration: " + str(ITER) + "...")
        model.save("../exp/models/"+mode+"_"+str(ITER) +
                   "lowest_iter_AdamTrainer.model")
        print("----------------------------------")
    if lowest_dev_loss > dev_loss_per_word:
        lowest_dev_loss = dev_loss_per_word
        successive_decreasing_counter = 0
    else:
        print("old learning rate: " + str(model.get_learning_rate()))
        model.reduce_learning_rate(5)
        print("new learning rate: " + str(model.get_learning_rate()))
        successive_decreasing_counter += 1

    if successive_decreasing_counter == 3:
        print("Early stopping...")
        break

log_writer.close()

model.save("../exp/models/"+mode+"_"+str(ITERATION)+"_iter_AdamTrainer.model")

# generate result
writer = open("../exp/results/test_"+mode+"_" +
              str(ITERATION)+"_iter.result", 'w')
sentences = []
print("Generating result...")
for sent_id, sent in enumerate(test_data):
    input_sent, output_sent = sent[0], sent[1]
    generated_sent = model.forward(input_sent, mode="predict")
    sentences.append(generated_sent)

print("Writing result...")
for sent in sentences:
    writer.write(sent+"\n")
writer.close()
