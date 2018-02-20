from dataset import DataEntry, DataSet, Vocab, Action

from nn.utils.io_utils import deserialize_from_file

#--------------------------------------
# ADDED
#--------------------------------------
#import dynet as dy
#import random
#import math
#import sys
def write_to_file(output_file, dataset, max_num):
	query_writer = open("query_"+output_file, 'w')
	code_writer = open("tree_"+output_file, 'w')
	print("Writing to file...")
	for index in range(max_num):
		query_writer.write(" ".join(dataset.get_examples(index).query)+"\n")
		code_parse_tree = str(dataset.get_examples(index).parse_tree)
		if "\n" in code_parse_tree:
			print("index: " + str(index))
		#code_writer.write(repr(code_parse_tree)+"\n")
		code_writer.write(code_parse_tree.replace("\n", "\\n")+"\n")

	query_writer.close()
	code_writer.close()

def main():
	'''
		Read file from Django data set
	'''
	train_data, dev_data, test_data = deserialize_from_file("data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin")

	#uncomment below for Hearthstone data set
	#train_data, dev_data, test_data = deserialize_from_file("hs.freq3.pre_suf.unary_closure.bin")

	print("----- TRAIN -----")
	train_length = len(train_data.examples)
	print(train_length) #16000 instances for django, 533 for hs
	write_to_file("train_.txt", train_data, train_length)

	print("----- DEV -----")
	dev_length = len(dev_data.examples)
	print(dev_length) #1000 instances for django, 66 for hs
	write_to_file("dev_.txt", dev_data, dev_length)

	print("----- TEST -----")
	test_length = len(test_data.examples)
	print(test_length) #1801 instances, 66 for hs
	write_to_file("test_.txt", test_data, test_length)
    
    #print(train_data.get_examples(0).query)
    #print(train_data.get_examples(0).parse_tree)

if __name__ == '__main__':
	main()