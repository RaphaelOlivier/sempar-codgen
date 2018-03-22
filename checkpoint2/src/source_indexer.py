import dynet as dy
import time
import random
import math
import sys

#from model1 import ASTNet

from argparse import ArgumentParser
from collections import Counter, defaultdict


parser = ArgumentParser(description='Checkpoint2 Code Generator')
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

# read data set
# input file format is "word1 word2 ..."

def read_dataset(source_file):
    with open(source_file, "r", encoding='utf-8', errors='ignore') as s_file:
        for source_line in s_file:
            sent_src = [sourceIndexer[x] for x in source_line.strip().split(" ")] + [eos_source]
            yield (sent_src)


# choose dataset
if mode == "django":
    modulo = 1000
    print("Using Django dataset to generate the indexer...")
    train_source_file = "../../data/django_dataset/django.train.desc"
    dev_source_file = "../../data/django_dataset/django.dev.desc"
    test_source_file = "../../data/django_dataset/django.test.desc"
else:
    modulo = 100
    print("Using HS dataset to generate the indexer...")
    train_source_file = "../../data/hs_dataset/hs.train.desc"
    dev_source_file = "../../data/hs_dataset/hs.dev.desc"
    test_source_file = "../../data/hs_dataset/hs.test.desc"

# Special words
unk_source = sourceIndexer["<unk>"]
eos_source = sourceIndexer['</s>']

vocab_length_source = len(sourceIndexer)
print("Source vocabulary size: {}".format(vocab_length_source))
train_source = list(read_dataset(train_source_file))

dev_data = list(read_dataset(dev_source_file))
test_data = list(read_dataset(test_source_file))

sourceIndexer = defaultdict(lambda: unk_source, sourceIndexer)
print (len(sourceIndexer))

