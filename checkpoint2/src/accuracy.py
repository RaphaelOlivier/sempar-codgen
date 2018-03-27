# test accuracy
from __future__ import print_function
import time
import random
import math
import sys
import argparse
from nltk.translate.bleu_score import sentence_bleu

from collections import Counter, defaultdict
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser(description='Checkpoint2 Accuracy script')
parser.add_argument('--data', type=str, default='django',
                    help='Dataset to be used')
parser.add_argument('--predicted_path', type=str, default="../../data/result.code",
                    help='Path storing predicted code')
args, _ = parser.parse_known_args()


#mode = 'heartstone'
mode = args.data
predicted = args.predicted_path
goldenTest, predicted = None, None

if mode == 'hs':
    goldenTest = "../../data/hs_dataset/hs.test.code"
    #predicted = "../../data/result.code"

if mode == 'django':
    goldenTest = "../../data/django_dataset/django.test.code"
    #predicted = "../../data/result.code"


total = 0
correct = 0
i=1
with open(goldenTest, "r", encoding='utf-8', errors='ignore') as s_file, open(predicted, "r", encoding='utf-8', errors='ignore') as t_file:
    for golden_line, predicted_line in zip(s_file, t_file):
        # print(golden_line+predicted_line)
        if golden_line[:-1] == predicted_line[:-1]:
            #print("correct !")
            correct = correct + 1
            print(i)
        total = total + 1
        i+=1
    print("exact accuracy : " + str(correct/total))


# calculate bleu score
score = 0
cum_score = 0
with open(goldenTest, "r", encoding='utf-8', errors='ignore') as s_file, open(predicted, "r", encoding='utf-8', errors='ignore') as t_file:
    for golden_line, predicted_line in zip(s_file, t_file):
        reference = golden_line[:-1].split(' ')
        candidate = predicted_line[:-1].split(' ')
        cum_score = cum_score + (sentence_bleu(reference, candidate))  # cumulative score blue
    total_score = cum_score*100.0/total
    print('BLEU score (percent): %.2f' % (total_score))
