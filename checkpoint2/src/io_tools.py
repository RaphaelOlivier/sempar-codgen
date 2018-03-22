import json
from tree import Tree

def json_to_tree_dataset(path,grammar):
    tree_list=[]
    with open(path,'r') as f:
        json_list = json.load(f)
        assert(type(json_list)==list)
        for dic in json_list:
            tree_list.append(Tree.make_from_dict(grammar,dic))
    return tree_list

def tree_to_json_dataset(path,grammar,tree_list):
    json_list=[]
    with open(path,'w') as f:
        for tree in tree_list:
            json_list.append(tree.to_dict())
        json.dump(json_list,f)
