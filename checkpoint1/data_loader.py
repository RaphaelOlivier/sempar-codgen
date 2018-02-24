from __future__ import print_function
import time
import random
import math
import sys
import argparse
from collections import Counter, defaultdict
import numpy as np


class DataLoader():

	def __init__(self, dataset, filePath):
		if dataset is 'django':
			self.train_source_file = filePath + "django_dataset/query_train_.txt"
			self.train_target_file = filePath + "django_dataset/tree_train_.txt"
			self.dev_source_file = filePath + "django_dataset/query_dev_.txt"
			self.dev_target_file = filePath + "django_dataset/tree_dev_.txt"
			self.test_source_file = filePath + "django_dataset/query_test_.txt"
			self.test_target_file = filePath + "django_dataset/tree_test_.txt"
		elif dataset is 'hs':
			self.train_source_file = filePath + "hs_dataset/query_hs_train_.txt"
			self.train_target_file = filePath + "hs_dataset/tree_hs_train_.txt"
			self.dev_source_file = filePath + "hs_dataset/query_hs_dev_.txt"
			self.dev_target_file = filePath + "hs_dataset/tree_hs_dev_.txt"
			self.test_source_file = filePath + "hs_dataset/query_hs_test_.txt"
			self.test_target_file = filePath + "hs_dataset/tree_hs_test_.txt"
	def read_dataset(self, data):
		if data is "train":
			source = self.train_source_file
			target = self.train_target_file
		elif data is "dev":
			source = self.dev_source_file
			target = self.dev_target_file
		elif data is "test":
			source = self.test_source_file
			target = self.test_target_file
		# natural language query
		w2i_source = defaultdict(lambda: len(w2i_source))
		# code parse tree
		w2i_target = defaultdict(lambda: len(w2i_target))
		with open(source, "r", encoding='utf-8', errors='ignore') as s_file, open(target, "r", encoding='utf-8', errors='ignore') as t_file:
			for source_line, target_line in zip(s_file, t_file):
				sent_src = [w2i_source[x] for x in source_line.strip().split(" ") + ['</s>']]
				sent_trg = [w2i_target[x] for x in ['<s>'] + target_line.strip().split(" ") + ['</s>']] 
				print (sent_src)
				print (sent_trg)
				yield  (sent_src, sent_trg) 

filePath = "/Users/pavvaru/Documents/Spring 2018/NNLP/Project/sempar-codgen/checkpoint1"
load_data = DataLoader("django", '')
train = list(load_data.read_dataset("train"))
print (train[0])

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
