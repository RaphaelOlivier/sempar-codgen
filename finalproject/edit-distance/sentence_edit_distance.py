#usage: python2 editdistance.py 1 raw_test.txt dictionary_test.txt output.txt

from __future__ import division
from string import ascii_lowercase as al
import glob
import re
import collections
import os
import sys
import math

def distance(raw_sentence, dict_sentence):
	#print raw_word
	#print dict_word
	raw_sentence = raw_sentence.split(' ')
	dict_sentence = dict_sentence.split(' ')
	m = len(raw_sentence)+1
	n = len(dict_sentence)+1
	matrix = [[0]*(m) for _ in range(n)]
	#print matrix
	for i in range(n):
		matrix[i][0] = i # n rows

	for i in range(m):
		matrix[0][i] = i # m columns
	#print matrix
	#print "................."
	for i in range(1,n):
		for j in range(1,m):
			if raw_sentence[j-1] == dict_sentence[i-1]:
				penalty = 0
			else:
				penalty = 1
			matrix[i][j] = min(matrix[i-1][j]+1 , matrix[i][j-1]+1 , matrix[i-1][j-1] + penalty) 
	#print matrix[n-1][m-1]
	return matrix[n-1][m-1]

def Levenshtein_edit_distance(rawtext,dictionary,output):

	vocabulary = []
	raw = []
	raw_ptr = open(rawtext)
	dict_ptr = open(dictionary)
	for raw_sentence in raw_ptr:
		print (raw_sentence)
		raw.append(raw_sentence)
	for dict_sentence in dict_ptr:
		print (dict_sentence)
		vocabulary.append(dict_sentence)

	output_ptr = open(output, 'w')
	i = 0
	for raw_sentence in raw:
		min_distance = 99999;
		for dict_sentence in vocabulary:
			if(raw_sentence == dict_sentence ):
				
				#print "hello"
				correct_sentence = dict_sentence
				min_distance = 0
				continue;
			else:
				answer = distance(raw_sentence, dict_sentence)
				if(answer < min_distance):
					min_distance = answer
					correct_sentence = dict_sentence	
		i = i+1
		if( i == 500):
			output_ptr.write(correct_sentence + " " + str(min_distance+1))
		else:
			output_ptr.write(correct_sentence + " " + str(min_distance+1)+"\n")		
		
raw = sys.argv[1]
dictionary = sys.argv[2]
output = sys.argv[3]

Levenshtein_edit_distance(raw,dictionary,output)
matrix = [[0]*(5) for _ in range(5)]
for i in range(5):
	matrix[i][0] = i

