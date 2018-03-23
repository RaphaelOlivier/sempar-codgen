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

class Indexer():

	def __init__(self,mode):

		self.mode = mode
		self.rules = Indexer.index_reader("../../data/indexer/rules_"+self.mode+".tsv", True, True)
		self.node_types = Indexer.index_reader("../../data/indexer/nodes_"+self.mode+".tsv", False, False)
		self.frontier_nodes = Indexer.index_reader("../../data/indexer/frontier_nodes_"+self.mode+".tsv", True, False)
		self.annot_vocab = Indexer.index_reader("../../data/indexer/annot_vocab_"+self.mode+".tsv", False, False)
		self.terminal_vocab = Indexer.index_reader("../../data/indexer/terminal_vocab_"+self.mode+".tsv", False, False)

		self.rules_index = {v:k for k,v in self.rules.items()}
		self.node_types_index = {v:k for k,v in self.node_types.items()}
		self.frontier_nodes_index = {v:k for k,v in self.frontier_nodes.items()}
		self.annot_vocab_index = {v:k for k,v in self.annot_vocab.items()}
		self.terminal_vocab_index = {v:k for k,v in self.terminal_vocab.items()}

	@staticmethod
	def index_reader(file_name, remove_paranthesis, index_to_val):
		indexer = {}
		with open(file_name, 'r') as input_file:
			for each_line in input_file.readlines():
				# print(each_line)
				value, index = each_line.rstrip().split("\t")
				if remove_paranthesis:
					value = value.replace("(", "").replace(")", "")
				if index_to_val:
					indexer[index] = value
				else:
					indexer[value] = index
		return indexer

	def get_node_index(self, token):
		return self.node_types[token]

	def get_vocab_index(self, token):
		return self.terminal_vocab[token]

	def get_vocab(self, token_index):
		return self.terminal_vocab_index[token_index]

	def action_type(self, node_type):
		if node_type in self.frontier_nodes:
			return "gen"
		else:
			return "apply"

	@property
	def vocab_length(self):
		return len(self.terminal_vocab)

	def rules_from_node(self):
		rule_index_list = self.rules.values()
		return np.array(list(rule_index_list))

	def get_children(self, rule_index):
		#print("---- " + rule_index + " ----")
		parent, raw_children = self.rules[rule_index].split(" -> ")
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
			node_idx = self.get_node_index(node_type)
			child_node = Node(node_type, node_label, node_idx, parent)
			children.append(child_node)
			child_node.print_node()

		return children
if __name__ == "__main__":
	indexer = Indexer("hs")
	indexer.get_children("44")
	indexer.get_children("1")
