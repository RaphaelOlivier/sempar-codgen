#usage: python2 editdistance.py 1 raw_test.txt dictionary_test.txt output.txt

from __future__ import division
from string import ascii_lowercase as al
import glob
import re
import collections
import os
import sys
import math


class Dictionary_Trie:

	#inititalization
	def __init__(self):
		self.string = None
		self.next_char = {} # a dictionary to store the the trie
		self.maxCost = 6

	def add_to_trie(self, string):
		TrieNode = self
		for character in string:
			if character not in TrieNode.next_char: # make a new node if its not present
				TrieNode.next_char[character] = Dictionary_Trie()
			TrieNode = TrieNode.next_char[character]
		TrieNode.string = string # after all the characters are found or stored in the tree, store the string

def make_trie(dictionary):
	dict_ptr = open(dictionary)
	vocabulary = []
	for dict_words in dict_ptr:
		vocabulary.append(dict_words[:-1])
	for term in vocabulary:
		trie.add_to_trie(term)

def search_word(word):

	answer = [] # stores the best match word
	
	first_row = []
	m = len(word) + 1
	for i in range(m):
		first_row.append(i) #since in DP, the current row uses the previoUS row for calculations
	prev_character = None
	#print first_row
	for character in trie.next_char:
		searchTrie(word, trie.next_char[character], character, prev_character, first_row, answer)
	return answer

def searchTrie(word ,node, character,prev_character,  prev_row, answer):
	columns = len(word) + 1
	presentRow = [prev_row[0] + 1]
	for column in range(1, columns):
		if(word[column-1] == character):
			penalty = 0
		else:
			penalty = 1
		ins_cost = presentRow[column - 1] +1 
		del_cost = prev_row[column] + 1
		subs_cost = penalty + prev_row[column - 1]
		final_cost = min(ins_cost,del_cost,subs_cost )
		# check for transposition using the optimal string aligment method
		if( column >1 ):
			#print word[column-1]
			if(word[column-1] == prev_character and word[column-2] == character ):
				swap_cost = prev_row[column-2] + 1
				final_cost = min(swap_cost,final_cost)

		presentRow.append(final_cost)

	if presentRow[-1] <= node.maxCost:
		if node.string != None:
			answer.append([node.string,presentRow[-1]])

	#there is atleast one element in 	
	if min(presentRow)<=node.maxCost:
		prev_character = character
		maximum_cost = presentRow[-1]
		for letter in node.next_char:
			searchTrie(word,node.next_char[letter] , letter,prev_character, presentRow, answer)

def Trie_string_edit_distance(raw, dictionary, output):
	make_trie(dictionary)
	raw_ptr = open(raw)
	raw = []
	output_ptr = open(output,'w')
	for raw_words in raw_ptr:
		raw.append(raw_words[:-1])
	i = 0
	for word in raw:
		result = search_word(word)
		#print word
		result= sorted(result,key=lambda x: x[1])
		#print result[0]
		i = i + 1
		if(not result):
			print word
		else:
			if( i == 500):
				output_ptr.write(result[0][0] + " " + str(result[0][1]))
			else:
				output_ptr.write(result[0][0] + " " + str(result[0][1]) + "\n")
		#print "............."

def distance(raw_word, dict_word):
	#print raw_word
	#print dict_word
	m = len(raw_word)+1
	n = len(dict_word)+1
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
			if raw_word[j-1] == dict_word[i-1]:
				penalty = 0
			else:
				penalty = 1
			matrix[i][j] = min(matrix[i-1][j]+1 , matrix[i][j-1]+1 , matrix[i-1][j-1] + penalty) 
	#print matrix[n-1][m-1]
	return matrix[n-1][m-1]

def with_transpose_distance(raw_word, dict_word):
	#print raw_word
	#print dict_word
	m = len(raw_word)
	n = len(dict_word)
	matrix = [[0]*(m+1) for _ in range(n+1)]
	#print matrix
	for i in range(-1,n):
		matrix[i][-1] = i # n rows

	for i in range(-1,m):
		matrix[-1][i] = i # m columns
	#print matrix
	#print "................."
	for i in range(n):
		for j in range(m):
			if raw_word[j] == dict_word[i]:
				penalty = 0
			else:
				penalty = 1
			matrix[i][j] = min(matrix[i-1][j]+1 , matrix[i][j-1]+1 , matrix[i-1][j-1] + penalty) 
			if( i >1 and j > 1 ):
				if(raw_word[j] == dict_word[i-1] and raw_word[j-1] == dict_word[i] ):
					matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2]+1)
	#print matrix[n-1][m-1]
	return matrix[n-1][m-1]

def Levenshtein_edit_distance(rawtext,dictionary,output):

	vocabulary = []
	raw = []
	raw_ptr = open(rawtext)
	dict_ptr = open(dictionary)
	for raw_words in raw_ptr:
		raw.append(raw_words[:-1])
	for dict_words in dict_ptr:
		vocabulary.append(dict_words[:-1])

	output_ptr = open(output, 'w')
	i = 0
	for raw_word in raw:
		min_distance = 99999;
		for dict_word in vocabulary:
			if(raw_word == dict_word ):
				
				#print "hello"
				correct_word = dict_word
				min_distance = 0
				continue;
			else:
				answer = distance(raw_word, dict_word)
				if(answer < min_distance):
					min_distance = answer
					#print dict_word
					correct_word = dict_word
		#print raw_word +" "+ correct_word	
		i = i+1
		#print raw_word +" "+ correct_word
		if( i == 500):
			output_ptr.write(correct_word + " " + str(min_distance+1))
		else:
			output_ptr.write(correct_word + " " + str(min_distance+1)+"\n")		
		

def Optimal_String_Alignment(rawtext,dictionary,output):

	vocabulary = []
	raw = []
	raw_ptr = open(rawtext)
	dict_ptr = open(dictionary)
	for raw_words in raw_ptr:
		raw.append(raw_words[:-1])
	for dict_words in dict_ptr:
		vocabulary.append(dict_words[:-1])

	output_ptr = open(output, 'w')
	i= 0
	for raw_word in raw:
		min_distance = 99999;
		for dict_word in vocabulary:
			answer = with_transpose_distance(raw_word, dict_word)
			if(answer < min_distance):
				min_distance = answer
				#print dict_word
				correct_word = dict_word
				#print correct_word
		i = i+1
		#print raw_word +" "+ correct_word
		if( i == 500):
			output_ptr.write(correct_word + " " + str(min_distance+1))
		else:
			output_ptr.write(correct_word + " " + str(min_distance+1)+ "\n")

def adjacent_transpose_distance(raw_word, dict_word):
	#print raw_word
	#print dict_word
	m = len(raw_word)
	n = len(dict_word)
	max_distance = len(raw_word) + len(dict_word)
	#creating a dictionary of character which will store the previous occurance of the aplhabet
	dict_dictWord = {x:0 for i,x in enumerate(al,1)}
	#print dict_dictWord
	matrix = [[0]*(m+2) for _ in range(n+2)]
	#print matrix
	for i in range(-1,n):
		matrix[i][-1] = max_distance# n rows
		matrix[i][0] = i 

	for i in range(-1,m):
		matrix[-1][i] = max_distance
		matrix[0][i] = i# m columns
	#print matrix
	#print "................."
	for i in range(1,n):
		prev_loc_col = 0 # for each row which represents a new character in the vocabulary word
		for j in range(1,m):
			l = prev_loc_col
			#get position of the previous occurence of this character
			if raw_word[j] == dict_word[i]:
				penalty = 0
				prev_loc_col = j # will be used in the next iteration as the previous loc
			else:
				penalty = 1
			k = dict_dictWord[raw_word[j]] # the occurence of the raw word charcter in the dict word previously 
			#print k,l
		    #last match column
			# l = db ( db just holds a chacracter)
			subs_cost = matrix[i-1][j-1] + penalty
			trans_cost = matrix[k-1][l-1]+ (i-k-1)+1+(j-l-1) # cost of tranferring between i,j and k,j
			ins_cost = matrix[i][j-1] + 1
			del_cost = matrix[i-1][j]+1
			matrix[i][j] = min( subs_cost , ins_cost, del_cost, trans_cost ) 
		dict_dictWord[dict_word[i]] = i; #updating the last occurence of the character in dictionary
	#print dict_dictWord 
	#print matrix
	return matrix[n-1][m-1]


def Distance_Adjacent_transposition(rawtext,dictionary,output):

	vocabulary = []
	raw = []
	raw_ptr = open(rawtext)
	dict_ptr = open(dictionary)
	for raw_words in raw_ptr:
		raw.append(raw_words[:-1])
	for dict_words in dict_ptr:
		vocabulary.append(dict_words[:-1])

	output_ptr = open(output, 'w')
	i = 0
	for raw_word in raw:
		min_distance = 99999;
		for dict_word in vocabulary:
			answer = adjacent_transpose_distance(raw_word, dict_word)
			if(answer < min_distance):
				min_distance = answer
				#print raw_word
				correct_word = dict_word
		i = i + 1		#print correct_word
		#print raw_word +" "+ correct_word
		if( i == 500):
			output_ptr.write(correct_word + " " + str(min_distance))
		else:
			output_ptr.write(correct_word + " " + str(min_distance) + "\n")


mode = int(sys.argv[1])
raw = sys.argv[2]
dictionary = sys.argv[3]
output = sys.argv[4]

if( mode is 1 ):
	Levenshtein_edit_distance(raw,dictionary,output)
	matrix = [[0]*(5) for _ in range(5)]
	for i in range(5):
		matrix[i][0] = i
	#print matrix
elif (mode is 2):
	Optimal_String_Alignment(raw, dictionary, output)
elif (mode is 3):
	#need to do bonus task
	trie = Dictionary_Trie()
	#make_trie(dictionary)
	#result = search_word("clic")
	#print result
	#search(trie,"desfilar")
	Trie_string_edit_distance(raw, dictionary, output)
elif (mode is 4):
	print "heloo"
	Distance_Adjacent_transposition(raw, dictionary, output) # bonus of task 1
else:
	print "unknown mode"


