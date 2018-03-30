from dataset import DataEntry, DataSet, Vocab, Action
import json
from io_utils import deserialize_from_file
from astnode import ASTNode
import lang.py.parse as parse
import astor
import ast

from argparse import ArgumentParser
parser = ArgumentParser(description='Checkpoint2 Code Generator')
parser.add_argument('--data', type=str, default='django',
                    help='Dataset to be used')
parser.add_argument('--task', type=str, default="ast2json",
                    help='Import or export')

parser.add_argument('--load', type=str, default='../checkpoint2/exp/results/hs.result.json',
                    help='Dataset to be used')
parser.add_argument('--write', type=str, default='../checkpoint2/exp/results/hs.result.code',
                    help='Training iteration')
parser.add_argument('--ref', type=str, default='../checkpoint2/exp/results/hs.test.code',
                    help='Training iteration')

args, _ = parser.parse_known_args()

def reverse_typename(t):
    if t == 'root':
        return t
    elif t == 'epsilon':
        return t

    elif t == 'int':
        return int
    elif t == 'float':
        return float

    elif t == 'bool':
        return bool
    elif t == 'str':
        return str
    elif t[-1]=='*':
        return reverse_typename(t[:-1])
    else:
        return vars(ast)[t]

def write_to_json_file(path,data):
    l = []
    g = data.grammar
    v = data.terminal_vocab
    for i in range(len(data.examples)):
    #for i in range(428,429):
        t = data.examples[i].parse_tree
        q = data.examples[i].query
        #print("query: " + str(t))
        d = t.to_dict(q,g,v)
        l.append(d)
        #print(d)

    with open(path,'w') as f:
        json.dump(l,f, encoding='latin1')

def write_to_code_file(mode, data, path_to_load, path_to_export, path_raw_code):
    g = data.grammar
    nt = {v:reverse_typename(k) for k,v in g.node_type_to_id.items()}

    #print(nt,g.node_type_to_id)
    v = data.terminal_vocab

    raw=[]
    with open(path_raw_code,'r') as f:
        for line in f:
            raw.append(line[:-1])

    with open(path_to_load,'r') as f:
        l = json.load(f, encoding='utf8')
    l_code = []
    for i in range(len(l)):
        print(raw[i])
        t = ASTNode.from_dict(l[i], nt,v)
        ast_tree = parse.decode_tree_to_python_ast(t)
        code = astor.to_source(ast_tree)[:-1]
        real_code = parse.de_canonicalize_code(code, raw[i])
        if(mode=="hs"):
            real_code = " ".join(parse.tokenize_code_adv(real_code, True)).replace("\n \n","\n").replace("\n","#NEWLINE#")
        if(mode=="django"):
            real_code = " ".join(parse.tokenize_code_adv(real_code, False))
        #print(real_code,raw[i])
        l_code.append(real_code)


    with open(path_to_export,'w') as f:
        for c in l_code:
            f.write(c+"\n")

def main(flag,task,path_load = None, path_write= None, path_raw_code=None):

    if flag == "django":
        train_data, dev_data, test_data = deserialize_from_file("../../django.cleaned.dataset.freq5.par_info.refact.space_only.bin")
    elif flag == "hs":
        train_data, dev_data, test_data = deserialize_from_file("../../hs.freq3.pre_suf.unary_closure.bin")
    #uncomment below for Hearthstone data set
    #train_data, dev_data, test_data = deserialize_from_file("hs.freq3.pre_suf.unary_closure.bin")
    if(task=="ast2json"):
        
        print("----- TRAIN -----")
        write_to_json_file("../data/"+flag+"_dataset/"+flag+".train.json", train_data)
        
        print("----- DEV -----")
        write_to_json_file("../data/"+flag+"_dataset/"+flag+".dev.json", dev_data)
        
        print("----- TEST -----")
        write_to_json_file("../data/"+flag+"_dataset/"+flag+".test.json", test_data)
        
    if(task=="json2ast"):
        print("----- TEST -----")
        write_to_code_file(flag, train_data, path_load, path_write,path_raw_code)

    #print(train_data.get_examples(0).query)
    #print(train_data.get_examples(0).parse_tree)

if __name__ == '__main__':
	main(args.data,args.task,args.load,args.write, args.ref)
