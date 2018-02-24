# test accuarcy
from __future__ import print_function
import time
import random
import math
import sys
import argparse
from nltk.translate.bleu_score import sentence_bleu

start = time.time()

from collections import Counter, defaultdict
import numpy as np

goldenTest = "hs_dataset/tree_hs_test_.txt"
predicted = "output_tree.txt"

total = 0
correct = 0
with open(goldenTest, "r", encoding='utf-8', errors='ignore') as s_file, open(predicted, "r", encoding='utf-8', errors='ignore') as t_file:
        for golden_line, predicted_line in zip(s_file, t_file):
        	
        	if golden_line == predicted_line:
        		correct = correct + 1
        	total = total + 1
        print ("exact accuracy : " + str(correct/total))


# calculate bleu score
score = 0
cum_score = 0
with open(goldenTest, "r", encoding='utf-8', errors='ignore') as s_file, open(predicted, "r", encoding='utf-8', errors='ignore') as t_file:
        for golden_line, predicted_line in zip(s_file, t_file):
        	reference = golden_line.split(' ')
        	candidate = predicted_line.split(' ')
        	score = score + (sentence_bleu(reference, candidate)) # bleu score per sentence
       		cum_score = cum_score + (sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))) # cumulative score blue


        print ("average bleu over all sentences : "  + str(score/total))
        print ("cumulative bleu score over all sentences : ", str(cum_score/total)) 



