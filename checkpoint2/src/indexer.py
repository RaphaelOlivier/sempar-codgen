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
		query = " ".join(dataset.get_examples(index).query)
		if "\n" in query:
			print("query index: " + str(index))
		query_writer.write(query.replace("\n", "\\n")+"\n")

		code_parse_tree = str(dataset.get_examples(index).parse_tree)
		if "\n" in code_parse_tree:
			print("code parse tree index: " + str(index))
		#code_writer.write(repr(code_parse_tree)+"\n")
		code_writer.write(code_parse_tree.replace("\n", "\\n")+"\n")

	query_writer.close()
	code_writer.close()

def export_id(my_map, output_file):
	writer = open(output_file, 'w')
	for token, index in my_map.iteritems():
		writer.write(str(token) + "\t" + str(index) + "\n")
	writer.close()

def set_to_id(terminal_node_set, output_file):
	writer = open(output_file, 'w')
	index = 0
	for element in terminal_node_set:
		writer.write(str(element) + "\t" + str(index) + "\n")
		index += 1
	writer.close()

def run_hs():
	train_data, dev_data, test_data = deserialize_from_file("data/hs.freq3.pre_suf.unary_closure.bin")


	print("----- TRAIN HS -----")
	
	train_annot = train_data.annot_vocab.token_id_map
	export_id(train_annot, "indexer/annot_vocab_hs.tsv")
	train_terminal = train_data.terminal_vocab.token_id_map
	export_id(train_terminal, "indexer/terminal_vocab_hs.tsv")
	train_grammar = train_data.grammar.rule_to_id
	export_id(train_grammar, "indexer/rules_hs.tsv")
	train_nodes = train_data.grammar.node_type_to_id
	export_id(train_nodes, "indexer/nodes_hs.tsv")
	train_terminal = dev_data.grammar.terminal_nodes
	set_to_id(train_terminal, "indexer/frontier_nodes_hs.tsv")
	print("done HS")

def run_django():
	train_data, dev_data, test_data = deserialize_from_file("data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin")


	print("----- TRAIN DJANGO -----")
	
	train_annot = train_data.annot_vocab.token_id_map
	export_id(train_annot, "indexer/annot_vocab_django.tsv")
	train_terminal = train_data.terminal_vocab.token_id_map
	export_id(train_terminal, "indexer/terminal_vocab_django.tsv")
	train_grammar = train_data.grammar.rule_to_id
	export_id(train_grammar, "indexer/rules_django.tsv")
	train_nodes = train_data.grammar.node_type_to_id
	export_id(train_nodes, "indexer/nodes_django.tsv")
	train_terminal = dev_data.grammar.terminal_nodes
	set_to_id(train_terminal, "indexer/frontier_nodes_django.tsv")
	print("done Django")

def main():
	print("----- START -----")
	run_django()
	run_hs()


if __name__ == '__main__':
	main()