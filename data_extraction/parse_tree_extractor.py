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
"""
{'Index': <class '_ast.Index'>, 'Slice': <class '_ast.Slice'>, 'Sub': <class '_ast.Sub'>, 'For': <class '_ast.For'>,
 'UnaryOp': <class '_ast.UnaryOp'>, 'int': int, 'Lt': <class '_ast.Lt'>, 'Call': <class '_ast.Call'>,
  'comprehension': <class '_ast.comprehension'>, 'IsNot': <class '_ast.IsNot'>, 'operator': <class '_ast.operator'>, 'root': 'root',
   'Subscript': <class '_ast.Subscript'>, 'slice': <class '_ast.slice'>, 'Return': <class '_ast.Return'>, 'GtE': <class '_ast.GtE'>,
    'Tuple': <class '_ast.Tuple'>, 'expr*': <class '_ast.expr'>*, 'unaryop': <class '_ast.unaryop'>, 'Or': <class '_ast.Or'>,
     20: <class '_ast.Break'>, 21: <class '_ast.arguments'>, 22: <class '_ast.Not'>, 23: <class '_ast.ImportFrom'>,
      24: 'keyword*', 25: <class '_ast.AugAssign'>, 26: <class '_ast.LtE'>, 27: <class '_ast.BinOp'>,
       28: 'epsilon', 29: <class '_ast.BoolOp'>, 30: <class '_ast.List'>, 31: <class '_ast.ClassDef'>,
        32: <class '_ast.stmt'>, 33: 'cmpop*', 34: <class '_ast.boolop'>, 35: <class '_ast.alias'>,
         36: <class '_ast.Assign'>, 37: <class '_ast.FunctionDef'>, 38: <class '_ast.Lambda'>, 39: <class '_ast.And'>,
          40: <class '_ast.Compare'>, 41: <class '_ast.Gt'>, 42: <class '_ast.cmpop'>, 43: <class '_ast.keyword'>,
           44: <class '_ast.In'>, 45: <class '_ast.NotEq'>, 46: <class '_ast.expr'>, 47: <class '_ast.Is'>,
            48: 'comprehension*', 49: <class '_ast.Add'>, 50: 'alias*', 51: 'stmt*',
             52: 'str', 53: <class '_ast.Attribute'>, 54: <class '_ast.Eq'>, 55: <class '_ast.ListComp'>,
              56: <class '_ast.If'>},
OrderedDict([('Index', 0), ('Slice', 1), ('Sub', 2), ('For', 3),
 ('UnaryOp', 4), ('int', 5), ('Lt', 6), ('Call', 7),
  ('comprehension', 8), ('IsNot', 9), ('operator', 10), ('root', 11),
   ('Subscript', 12), ('slice', 13), ('Return', 14), ('GtE', 15),
    ('Tuple', 16), ('expr*', 17), ('unaryop', 18), ('Or', 19),
     ('Break', 20), ('arguments', 21), ('Not', 22), ('ImportFrom', 23),
      ('keyword*', 24), ('AugAssign', 25), ('LtE', 26), ('BinOp', 27),
       ('epsilon', 28), ('BoolOp', 29), ('List', 30), ('ClassDef', 31),
        ('stmt', 32), ('cmpop*', 33), ('boolop', 34), ('alias', 35),
         ('Assign', 36), ('FunctionDef', 37), ('Lambda', 38), ('And', 39),
          ('Compare', 40), ('Gt', 41), ('cmpop', 42), ('keyword', 43),
           ('In', 44), ('NotEq', 45), ('expr', 46), ('Is', 47),
            ('comprehension*', 48), ('Add', 49), ('alias*', 50), ('stmt*', 51),
             ('str', 52), ('Attribute', 53), ('Eq', 54), ('ListComp', 55), ('If', 56)])"""


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
        t = data.examples[i].parse_tree
        q = data.examples[i].query
        d = t.to_dict(q,g,v)
        l.append(d)

    with open(path,'w') as f:
        json.dump(l,f, encoding='latin1')

def write_to_code_file(data, path_to_load, path_to_export, path_raw_code):
    g = data.grammar
    nt = {v:reverse_typename(k) for k,v in g.node_type_to_id.items()}

    print(nt,g.node_type_to_id)
    v = data.terminal_vocab

    raw=[]
    with open(path_raw_code,'r') as f:
        for line in f:
            raw.append(line[:-1])

    with open(path_to_load,'r') as f:
        l = json.load(f, encoding='utf8')
    l_code = []
    for i in range(len(l)):
        t = ASTNode.from_dict(l[i], nt,v)
        ast_tree = parse.decode_tree_to_python_ast(t)
        code = astor.to_source(ast_tree)[:-1]
        real_code = parse.de_canonicalize_code(code, raw[i])

        print(real_code,raw[i])
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
        write_to_code_file(train_data, path_load, path_write,path_raw_code)

    #print(train_data.get_examples(0).query)
    #print(train_data.get_examples(0).parse_tree)

if __name__ == '__main__':
	main(args.data,args.task,args.load,args.write, args.ref)
