#usage: python2 editdistance.py 1 raw_test.txt dictionary_test.txt output.txt

from __future__ import division
from string import ascii_lowercase as al
import glob
import re
import collections
import os
import sys
import math
import numpy as np

def sentence_distance(first_sentence, second_sentence):
	first_sentence = first_sentence.split(' ')
	second_sentence = second_sentence.split(' ')
	m = len(first_sentence)+1
	n = len(second_sentence)+1
	matrix = np.zeros((n, m), dtype=int) #[[0]*(m) for _ in range(n)]
	#print matrix

	for i in range(n):
		matrix[i][0] = i # n rows

	for i in range(m):
		matrix[0][i] = i # m columns
	print("-----------")
	for i in range(1,n):
		for j in range(1,m):
			if first_sentence[j-1] == second_sentence[i-1]:
				penalty = 0
			else:
				penalty = 1

			#get min
			matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1] + penalty)

	#print matrix[n-1][m-1]
	print matrix
	return matrix, matrix[n-1][m-1]

def align(first_sentence, second_sentence, matrix):
	first_sentence = first_sentence.split(' ')
	second_sentence = second_sentence.split(' ')
	m = len(first_sentence)
	n = len(second_sentence)
	first_index_dict = {}
	second_index_dict = {}
	i = n
	j = m
	reverse1 = []
	reverse2 = []
	print("first sentence: " + str(first_sentence) + " " + str(len(first_sentence)))
	print("second sentence: " + str(second_sentence) + " " + str(len(second_sentence)))
	#print("i: " + str(i))
	#print("j: " + str(j))
	unedited_words = {}
	while i != 0 and j != 0:		
		first_index = j-1
		second_index = i-1
		#print("first index: " + str(first_index))
		#print("second index: " + str(second_index))
		word1 = first_sentence[first_index]
		word2 = second_sentence[second_index]
		same_words = (first_sentence[first_index] == second_sentence[second_index])
		if same_words:
			unedited_words[word1] = first_index
			temp_value = matrix[i-1][j-1]
		else:
			temp_value = matrix[i-1][j-1] + 1

		if matrix[i][j] == temp_value:
			i = i-1
			j = j-1
			reverse1.append(word1)
			reverse2.append(word2)
			first_index_dict[first_index] = [word1, word2, same_words]
			second_index_dict[second_index] = [word1, word2, same_words]
		elif matrix[i][j] == matrix[i-1][j] + 1:
			#print("2nd case -- i: " + str(i) + " j: " + str(j))
			reverse1.append("-")
			reverse2.append(word2)
			i = i-1
			first_index_dict[first_index] = ["NULL", word2, same_words]
			second_index_dict[second_index] = ["NULL", word2, same_words]
		else:
			#print("3rd case -- i: " + str(i) + " j: " + str(j) + " sec idx: " + str(second_index))
			reverse1.append(word1)
			reverse2.append("-")
			j = j-1
			first_index_dict[first_index] = [word1, "NULL", same_words]
			second_index_dict[second_index] = [word1, "NULL", same_words]
			print("--------------------")
		

	print reverse1[::-1]
	print reverse2[::-1]
	print unedited_words
	print first_index_dict
	print second_index_dict
	return unedited_words, first_index_dict, second_index_dict

first = "a x b c e f"
second = "a b d e v w x y"
matrix, dist = sentence_distance(first, second)
print("distance: " + str(dist))
align(first, second, matrix)

#	x = first_sentence
#	x_m = second_sentence
def simi(first_sentence, second_sentence):
	max_score = max(len(first_sentence.split()), len(second_sentence.split()))
	simi = 1.0 - (sentence_distance(first_sentence, second_sentence)[1]/max_score)
	return simi

simi_score = simi(first, second)
print ("simi score: " + str(simi_score))