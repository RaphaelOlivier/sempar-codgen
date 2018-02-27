# test accuracy
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

mode = 'heartstone'
# mode='django'

goldenTest, predicted = None, None

if mode == 'heartstone':
    goldenTest = "hs_dataset/hs.test.code"
    predicted = "results/test_hs_50_iter.result"

if mode == 'django':
    goldenTest = "django_dataset/django.test.code"
    predicted = "results/test_django_10_iter.result"


total = 0
correct = 0
with open(goldenTest, "r", encoding='utf-8', errors='ignore') as s_file, open(predicted, "r", encoding='utf-8', errors='ignore') as t_file:
    for golden_line, predicted_line in zip(s_file, t_file):

        if golden_line == predicted_line:
            correct = correct + 1
        total = total + 1
    print("exact accuracy : " + str(correct/total))


# calculate bleu score
score = 0
cum_score = 0
with open(goldenTest, "r", encoding='utf-8', errors='ignore') as s_file, open(predicted, "r", encoding='utf-8', errors='ignore') as t_file:
    for golden_line, predicted_line in zip(s_file, t_file):
        reference = golden_line.split(' ')
        candidate = predicted_line.split(' ')
        cum_score = cum_score + (sentence_bleu(reference, candidate))  # cumulative score blue
    total_score = cum_score*100.0/total
    print('BLEU score (percent): %.2f' % (total_score))
