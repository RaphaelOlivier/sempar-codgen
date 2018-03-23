import dynet as dy
import time
import random
import math
import sys

#from model1 import ASTNet

from argparse import ArgumentParser
from collections import Counter, defaultdict

# description

class VocabIndexer:
    def __init__(self):
        self.building = True
        self.vocab = {"<unk>":0,"<eos>":1}
        self.length = 2

    def build(self):
        self.building=True

    def isBuilt(self):
        self.building=False

    def __getitem__(self,w):
        if w in self.vocab.keys():
            return self.vocab[w]
        else:
            if self.building:
                self.vocab[w]=self.length
                self.length+=1
                return self.length-1
            else:
                return self.vocab["<unk>"]

# read data set
# input file format is "word1 word2 ..."
class SourceDataset:
    def __init__(self,mode):
        if mode == "django":
            print("Using Django dataset to generate the indexer...")
            train_source_file = "../../data/django_dataset/django.train.desc"
            dev_source_file = "../../data/django_dataset/django.dev.desc"
            test_source_file = "../../data/django_dataset/django.test.desc"
        else:
            print("Using HS dataset to generate the indexer...")
            train_source_file = "../../data/hs_dataset/hs.train.desc"
            dev_source_file = "../../data/hs_dataset/hs.dev.desc"
            test_source_file = "../../data/hs_dataset/hs.test.desc"

        self.indexer = VocabIndexer()
        self.indexer.build()

        self.train_str, self.train_index = self.read_dataset(train_source_file)
        self.indexer.isBuilt()

        print("Source vocabulary size: {}".format(self.vocab_length))

        self.dev_str, self.dev_index = self.read_dataset(dev_source_file)
        self.test_str, self.test_index = self.read_dataset(test_source_file)

    def read_dataset(self,source_file):
        data_index = []
        data_str = []
        with open(source_file, "r", encoding='utf-8', errors='ignore') as s_file:
            for source_line in s_file:
                sent = source_line.strip().split(" ") + ["<eos>"]
                sent_int = [self.indexer[x] for x in sent] 
                data_str.append(sent)
                data_index.append(sent_int)
        return data_str, data_index

    @property
    def vocab_length(self):
        return self.indexer.length
