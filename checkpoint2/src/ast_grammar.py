import ast

class Grammar:
	def __init__(self):

		self.ast_grammars = {
			"FunctionDef":
			{
				"name":{
					"type": str,
					"is_optional": False,
					"is_list": False
				},
				"args":{
					"type": ast.arguments,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				},
				"decorator_list":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"ClassDef":
			{
				"name":{
					"type": str,
					"is_optional": False,
					"is_list": False
				},
				"bases":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				},
				"decorator_list":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"Return":
			{
				"value":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"Delete":
			{
				"target":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"Assign":
			{
				"targets":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"AugAssign":
			{
				"target":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"op":{
					"type": ast.operator,
					"is_optional": False,
					"is_list": False
				},
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"Print":
			{
				"dest":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"values":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"nl":{
					"type": bool,
					"is_optional": False,
					"is_list": False
				}
			},
			"For":
			{
				"target":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"iter":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				},
				"orelse":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				}
			},
			"While":
			{
				"test":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				},
				"orelse":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				}
			},
			"If":
			{
				"test":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.stmt, 
					"is_optional": False,
					"is_list": True
				},
				"orelse":{
					"type": ast.stmt, 
					"is_optional": False,
					"is_list": True
				}
			},
			"With":
			{
				"context_expr":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"optional_vars":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				}
			},
			"Raise":
			{
				"type":{
					"typ": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"inst":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"tback":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"TryExcept":
			{
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				},
				"handlers":{
					"type": ast.excepthandler,
					"is_optional": False,
					"is_list": True
				},
				"orelse":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				}
			},
			"TryFinally":
			{
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				},
				"finalbody":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				}
			},
			"Assert":
			{
				"test":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"msg":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"Import":
			{
				"names":{
					"type": ast.alias,
					"is_optional": False,
					"is_list": True
				}
			},
			"ImportFrom":
			{
				"module":{
					"type": str,
					"is_optional": True,
					"is_list": False
				},
				"":{
					"type": ast.alias,
					"is_optional": False,
					"is_list": True
				},
				"level":{
					"type": int,
					"is_optional": True,
					"is_list": False
				}
			},
			"Exec":
			{
				"body":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"globals":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"locals":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"Global":
			{
				"names":{
					"type": str,
					"is_optional": False,
					"is_list": True
				}
			},
			"Expr":
			{
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"BoolOp":
			{
				"op":{
					"type": ast.boolop,
					"is_optional": False,
					"is_list": False
				},
				"values":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"BinOp":
			{
				"left":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"op":{
					"type": ast.operator,
					"is_optional": False,
					"is_list": False
				},
				"right":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"UnaryOp":
			{
				"op":{
					"type": ast.unaryop,
					"is_optional": False,
					"is_list": False
				},
				"operand":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"Lambda":
			{
				"args":{
					"type": ast.arguments,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"IfExp":
			{
				"test":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"orelse":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"Dict":
			{
				"keys":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"values":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"Set":
			{
				"elts":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"ListComp":
			{
				"elt":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"generators":{
					"type": ast.comprehension,
					"is_optional": False,
					"is_list": True
				}
			},
			"SetComp":
			{
				"elt":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"generators":{
					"type": ast.comprehension,
					"is_optional": False,
					"is_list": True
				}
			},
			"DictComp":
			{
				"key":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"generators":{
					"type": ast.comprehension,
					"is_optional": False,
					"is_list": True
				}
			},
			"GeneratorExpr":
			{
				"elt":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"generators":{
					"type": ast.comprehension,
					"is_optional": False,
					"is_list": True
				}
			},
			"Yield":
			{
				"value":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"Compare":
			{
				"func":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"ops":{
					"type": ast.cmpop,
					"is_optional": False,
					"is_list": True
				},
				"comparators":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"Call":
			{
				"func":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"args":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"keywords":{
					"type": ast.keyword,
					"is_optional": False,
					"is_list": True
				},
				"starargs":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"kwargs":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"Repr":
			{
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"Num":
			{
				"n":{
					"type": object,
					"is_optional": False,
					"is_list": False
				}
			},
			"Str":
			{
				"s":{
					"type": str,
					"is_optional": False,
					"is_list": False
				}
			},
			"Attribute":
			{
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"attr":{
					"type": str,
					"is_optional": False,
					"is_list": False
				},
				"ctx":{
					"type": ast.expr_context,
					"is_optional": False,
					"is_list": False
				}
			},
			"Subscript":
			{
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"slice":{
					"type": ast.slice,
					"is_optional": False,
					"is_list": False
				},
				"ctx":{
					"type": ast.expr_context,
					"is_optional": False,
					"is_list": False
				}
			},
			"Name":
			{
				"id":{
					"type": str,
					"is_optional": False,
					"is_list": False
				},
				"ctx":{
					"type": ast.expr_context,
					"is_optional": False,
					"is_list": False
				}
			},
			"List":
			{
				"elts":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"ctx":{
					"type": ast.expr_context,
					"is_optional": False,
					"is_list": False
				}
			},
			"Tuple":
			{
				"elts":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"ctx":{
					"type": ast.expr_context,
					"is_optional": False,
					"is_list": False
				}
			},
			"attributes":
			{
				"lineno":{
					"type": int,
					"is_optional": False,
					"is_list": True
				},
				"col_offset":{
					"type": int,
					"is_optional": False,
					"is_list": False
				}
			},
			"Slice":
			{
				"lower":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"upper":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"step":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				}
			},
			"ExtSlice":
			{
				"dims":{
					"type": ast.slice,
					"is_optional": False,
					"is_list": True
				}
			},
			"Index":
			{
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"comprehension":
			{
				"target":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"iter":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"ifs":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"ExceptHandler":
			{
				"type":{
					"type": ast.expr,
					"is_optional": True,
					"is_list": False
				},
				"name":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				},
				"body":{
					"type": ast.stmt,
					"is_optional": False,
					"is_list": True
				}
			},
			"arguments":
			{
				"args":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				},
				"vararg":{
					"type": str,
					"is_optional": True,
					"is_list": False
				},
				"kwarg":{
					"type": str,
					"is_optional": True,
					"is_list": False
				},
				"defaults":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": True
				}
			},
			"keyword":
			{
				"arg":{
					"type": str,
					"is_optional": False,
					"is_list": False
				},
				"value":{
					"type": ast.expr,
					"is_optional": False,
					"is_list": False
				}
			},
			"alias":
			{
				"name":{
					"type": str,
					"is_optional": False,
					"is_list": False
				},
				"asname":{
					"type": str,
					"is_optional": True,
					"is_list": False
				}
			}
		}
		self.terminals = {
			"expr_context":{
				ast.Load,
				ast.Store,
				ast.Del,
				ast.AugLoad,
				ast.AugStore,
				ast.Param
			},
			"slice":{
				ast.Ellipsis
			},
			"boolop":{
				ast.And,
				ast.Or
			},
			"operator":{
				ast.Add,
				ast.Sub,
				ast.Mult,
				ast.Div,
				ast.Mod,
				ast.Pow,
				ast.LShift,
				ast.RShift,
				ast.BitOr,
				ast.BitXor,
				ast.BitAnd,
				ast.FloorDiv
			},
			"unaryop":{
				ast.Invert,
				ast.Not,
				ast.UAdd,
				ast.USub
			},
			"cmpop":{
				ast.Eq,
				ast.NotEq,
				ast.Lt,
				ast.LtE,
				ast.Gt,
				ast.GtE,
				ast.Is,
				ast.IsNot,
				ast.In,
				ast.NotIn
			}
		}

	def get_children(self, rule_parent):
		return self.ast_grammars[rule_parent]

	def action_type(self, node_type):
		#return APPLYRULE or GENTOKEN
		if node_type in self.ast_grammars:
			return "ApplyRule"
		else:
			return "GenToken"

#preprocessing Hearthstone
def preprocess_hs(raw_code):
	new_code = raw_code.replace(" #MERGE# ","")
	new_code = new_code.replace("#NEWLINE#", "\n")
	new_code = new_code.replace("#INDENT#", "\t")
	return new_code

#test preprocessing hearthstone data
with open("/Users/shayati/Documents/sem2/NN for NLP/checkpoint1/hs_dataset/hs.test.code", 'r') as input_file:
	for each_line in input_file.readlines():
		result = preprocess_hs(each_line)
		print(result)
		break

def preprocess_django(data):
	return None