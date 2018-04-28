import sys
import os
import numpy as np
from string import punctuation


file1 = sys.argv[1]

def process_class_names(instance):
	words = instance.split(' ')
	words = words[:10]
	clss = ''
	start = words.index('class')
	end = words.index('(')
	clss = words[start+1]
	for i in range(start+2, end):
		clss = clss +' ' + words[i]
	original_clss = clss
	clss = strip_punctuation(clss)
	clss = " ".join(clss.split())
	words = clss.split(' ')
	clss = ''
	for word in words:
		if word[0].isupper():
			clss = clss + ' '+ word
		else:
			clss = clss + word
	clss = clss.strip()
	instance = instance.replace(original_clss,clss,1)
	print (instance)

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def post_process_HS():

	fileptr = open(file1, 'r')
	for instance in fileptr:
		process_class_names(instance)

if __name__=="__main__":
    post_process_HS()