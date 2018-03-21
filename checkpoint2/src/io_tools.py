import json
from tree import Tree

def json_to_tree_dataset(path):
    tree_list=[]
    with open(path,'r') as f:
        json_list = json.loads(f)
        assert(type(json_list)==list)
        for dic in json_list:
            tree_list.append(Tree.make_from_dict(dic))
    return tree_list

def tree_to_json_dataset(tree_list,path):
    json_list=[]
    with open(path,'w') as f:
        for tree in tree_list:
            json_list.append(tree.to_dict())
        json.dumps(json_list,f)
