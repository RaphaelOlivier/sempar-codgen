import numpy as np

class Indexer():

	def __init__(self,mode):

		self.mode = mode
		self.rules = Indexer.index_reader("../../data/indexer/rules_"+self.mode+".tsv", False, True)
		self.node_types = Indexer.index_reader("../../data/indexer/nodes_"+self.mode+".tsv", False, False)
		self.frontier_nodes = Indexer.index_reader("../../data/indexer/frontier_nodes_"+self.mode+".tsv", True, False)
		self.annot_vocab = Indexer.index_reader("../../data/indexer/annot_vocab_"+self.mode+".tsv", False, False)
		self.terminal_vocab = Indexer.index_reader("../../data/indexer/terminal_vocab_"+self.mode+".tsv", False, False)
		self.terminal_vocab_lower = {k.lower():v for k,v in self.terminal_vocab.items()}
		self.integer_indexes = self.extract_integer_vocab()
		self.rules_by_node = self.rules_index_by_node_type_index()
		self.rules_to_children = self.extract_children_from_rules()
		#print(self.rules_to_children[9],self.rules_to_children[57])
		#self.rules_index = {v:k for k,v in self.rules.items()}
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
				index = int(index)
				value = value[1:-1]
				if remove_paranthesis:
					value = value.replace("(", "").replace(")", "")
				if index_to_val:
					indexer[index] = value
				else:
					indexer[value] = index
		return indexer

	def rules_index_by_node_type_index(self):
		rules_by_node_type = [[] for n in self.node_types]
		for i in range(self.rule_length):
			parent, raw_children = self.rules[i].split(" -> ")
			#parent = self.appostrophize(parent)
			# print(parent)
			i_node = self.get_node_index(parent)
			rules_by_node_type[i_node].append(i)
		return rules_by_node_type

	def extract_children_from_rules(self):
		rules_to_children = [[] for r in self.rules]
		for i in range(self.rule_length):
			#print(self.rules[i])
			parent, raw_children = self.rules[i].split(" -> ")

			raw_children = raw_children.strip()
			children_list = raw_children.split(",")

			for child in children_list:
				split_child = child.strip().strip("(").strip(")").split("{")
				node_type = split_child[0].strip()
				#node_type = self.appostrophize(node_type)
				node_label = None
				if len(split_child) >= 2:
					node_label="{".join(split_child[1:]).strip()[:-1]
				node_idx = self.get_node_index(node_type)
				child_node = (node_idx,node_label)
				rules_to_children[i].append(child_node)
		return rules_to_children

	def get_node_index(self, token):
		return self.node_types[token]

	def get_node_type(self, index):
		return self.node_types_index[index]

	def get_integer_indexes(self):
		return self.integer_indexes

	def extract_integer_vocab(self):
		ints = list()
		for w,ind in self.terminal_vocab.items():
			try:
				n = int(w)
				ints.append(ind)
			except:
				pass
		print(ints)
		return np.array(ints)

	def appostrophize(self,word):
		if(word[-1]!="'"):
			word = word + "'"
		if(word[0]!="'"):
			word = "'" + word
		return word

	def get_vocab_index(self, token, lower=False):
		# print(self.terminal_vocab)
		if not lower:
			if token in self.terminal_vocab.keys():
				return self.terminal_vocab[token]
			else:
				return self.terminal_vocab["<unk>"]
		else:
			token=token.lower()
			if token in self.terminal_vocab_lower.keys():
				return self.terminal_vocab_lower[token]
			else:
				return self.terminal_vocab["<unk>"]

	def get_vocab(self, token_index):
		return self.terminal_vocab_index[token_index]

	def get_rule(self, index):
		return self.rules[index]

	def action_type(self, node_type):
		# print(self.node_types_index)
		# print(self.node_types_index[node_type])
		if self.node_types_index[node_type] in self.frontier_nodes.keys():
			return "gen"
		else:
			return "apply"

	@property
	def vocab_length(self):
		return len(self.terminal_vocab)

	@property
	def node_length(self):
		return len(self.node_types)

	@property
	def rule_length(self):
		return len(self.rules)

	def rules_from_node(self, node_type):
		rule_index_list = self.rules_by_node[node_type]
		return np.array(rule_index_list)

	def get_children(self, rule_index):
		#print("---- " + rule_index + " ----")
		children = self.rules_to_children[rule_index]
		return [(t,l,self.action_type(t)) for (t,l) in children]

if __name__ == "__main__":
	indexer = Indexer("hs")
	indexer.get_children("44")
	indexer.get_children("1")
