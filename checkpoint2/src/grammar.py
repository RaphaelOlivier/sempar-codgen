import numpy as np
class Node():
	def __init__(self, node_type, node_label, node_idx, parent):
		self.node_type = node_type
		self.node_label = node_label
		self.node_type_idx = node_idx
		self.parent = parent

	def get_node_type(self):
		return self.node_type

	def get_node_label(self):
		return self.node_label

	def get_node_type_index(self):
		return self.node_type_idx

	def get_parent(self):
		return self.parent

	def print_node(self):
		#print(self.node_type)
		print(self.node_type + " " + str(self.node_label) + " " + self.node_type_idx + " " + self.parent)

def index_reader(file_name, remove_paranthesis, index_to_val):
	indexer = {}
	with open(file_name, 'r') as input_file:
		for each_line in input_file.readlines():
			value, index = each_line.rstrip().split("\t")
			if remove_paranthesis:
				value = value.replace("(", "").replace(")", "")
			if index_to_val:
				indexer[index] = value
			else:
				indexer[value] = index
	return indexer

mode = "hs"
rules = index_reader("../indexer/rules_"+mode+".tsv", True, True)
node_types = index_reader("../indexer/nodes_"+mode+".tsv", False, False)
frontier_nodes = index_reader("../indexer/frontier_nodes_"+mode+".tsv", True, False)
annot_vocab = index_reader("../indexer/annot_vocab_"+mode+".tsv", False, False)
terminal_vocab = index_reader("../indexer/terminal_vocab_"+mode+".tsv", False, False)

def get_node_index(token):
	return node_types[token]

def get_vocab_index(token):
	return terminal_vocab[token]

def action_type(node_type):
	if node_type in frontier_nodes:
		return "gen"
	else:
		return "apply"

def rules_from_node():
	rule_index_list = rules.values()
	return np.array(list(rule_index_list))

def get_children(rule_index):
	#print("---- " + rule_index + " ----")
	parent, raw_children = rules[rule_index].split(" -> ")
	parent = parent.strip()
	raw_children = raw_children.strip()
	children_list = raw_children.split(",")
	#print(parent)
	#print(raw_children)
	#print(children_list)
	children = []
	for child in children_list:
		split_child = child.split("{")
		node_type = split_child[0].strip()
		node_label = None
		if len(split_child) == 2:
			node_label = split_child[1].replace("}", "").strip()
		node_idx = get_node_index(node_type)
		child_node = Node(node_type, node_label, node_idx, parent)
		children.append(child_node)
		child_node.print_node()

	return children
get_children("44")
get_children("1")