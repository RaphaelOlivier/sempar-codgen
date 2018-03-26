from node import Node
import ast

def parse_ast(code_node):
	ast_tree = Node(type(code_node))

	print("============")
	print(type(code_node).__name__)
	
	#check if the node is a leaf
	if type(code_node) == str or type(code_node) == bool:
		return None
	if type(code_node) == list:
		for x in code_node:
			print(x)
			parse_ast(x)
			
		return None
	print("---- ======FOR====== ----")
	for child_name, child_type in ast.iter_fields(code_node):
		print("here " + str(child_name) + " " + str(child_type))
		if str(child_name) != "ctx":
			parse_ast(child_type)

	return None

def main():
	ifTree = ast.parse("""if True:
	pass
else:
	pass""")
	sortTree = ast.parse("""sorted(my_list, reverse=True)""")
	top_node = sortTree.body[0]
	parse_ast(top_node)

main()