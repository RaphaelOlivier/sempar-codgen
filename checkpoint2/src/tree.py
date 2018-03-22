class SubTree:
    def __init__(self, node_type,label, parent):
        self.node_type = node_type
        self.label=label
        self.action_type = None
        self.rule = None
        self.tokens= None
        self.tokens_type = None
        self.tokens_index = None
        self.time_step = None
        self.parent = None
        self.children = None
        self.child_to_explore = 0

    def next(self, old_time=None):
        # recursive function to return the correct next node
        assert(self.is_built())
        if(old_time == None):
            old_time = self.time_step
        if(self.child_to_explore < len(children)):
            child = self.children[self.child_to_explore]
            self.child_to_explore += 1
            child.time_step = old_time+1
            return child
        elif self.parent == None:
            return None
        else:
            return self.parent.next(old_time)

    @staticmethod
    def root():
        # create a subtree with only the root node
        st = SubTree(node_type=grammar.get_node_index("root"), label=None, parent=None)
        st.time_step = 0
        return st

    def is_built(self):
        # check that the node has an action
        return self.children != None and (self.rule != None or self.token != None)

    def get_token(i):
        assert(i<len(self.tokens))
        return self.tokens[i],self.tokens_type[i],self.token_index[i]

    def set_rule(self, rule, child_nodes):
        # set a rule and children
        self.rule = rule
        self.children = []
        for node_type, label in child_nodes:
            self.children = SubTree(parent=self, node_type=node_type, label=label)

    def set_token(self, token, tktype, tkindex):
        self.tokens.append(token)
        self.tokens_type.append(tktype)
        self.tokens_index.append(tkindex)

    def set_action_type(self,action_type):
        self.action_type=action_type
        if(self.action_type=="gen"):
            self.tokens=list()
            self.tokens_type=list()
            self.tokens_index=list()


    @staticmethod
    def from_dict(d,parent=None):
        node_type = d["node_type"]
        label = d["label"]
        st = SubTree(parent=parent, node_type=node_type, label=label)
        action_type = d["action_type"]
        st.set_action_type(action_type)
        if(action_type=="apply_rule"):
            st.rule = d["rule"]

            children=d["children"]
            if(children==[]):
                st.children=[]
            else:
                st.children=[]
                for child_d in children:
                    st.children.append(SubTree.from_dict(child_d,parent=st))
        else:
            assert(action_type == "gen")
            st.tokens=d["tokens"]
            st.tokens_type=d["tokens_type"]
            st.tokens_index = d["tokens_index"]


        return st

    def to_dict(self):
        d = dict()
        d["node_type"]=self.node_type
        d["label"]=self.label
        d["action_type"]=self.action_type
        if(self.action_type=="apply_rule"):
            d["rule"]=self.rule
        else:
            assert(self.action_type=="gen")
            d["tokens_index"]=self.tokens_index
            d["tokens_type"]=self.tokens_type
            d["tokens"]=self.tokens
        d["children"]=[]
        for child in self.children:
            d["children"].append(child.toDict())

        return d

class Tree:
    def __init__(self, grammar):
        # abstract class for trees
        self.grammar = grammar
        self.current_node = None
        self.need_to_move = True

    def move(self):
        # shift the node to the next one, and specify the action type associated to its frontier node
        if(self.need_to_move):
            self.current_node = self.current_node.next()
            self.current_node.set_action_type(grammar.action_type(self.current_node.node_type))
        return bool(self.current_node)

    def get_node_type(self):
        # value needed by the model
        return self.current_node.node_type

    def get_action_type(self):
        return self.current_node.action_type

    def get_parent_time(self):
        # value needed by the model
        return self.current_node.parent.time_step

    def has_ended(self):
        # to know if decoding is over (current node is None)
        return self.current_node == None

    @staticmethod
    def make_from_dict(grammar, d):
        t = Tree(grammar)
        t.current_node = SubTree.from_dict(d)
        return t

    def to_dict(self, grammar):
        d = self.current_node.toDict()
        return d


class BuildingTree(Tree):
    # Trees used in prediction
    def __init__(self, grammar):
        # create a tree with only a root node
        super.__init__(self, grammar)
        self.current_node = SubTree.root()
        self.current_node.action_type = "apply"

    def pick_and_set_rule(self, rules_probs):
        # from the rule probabilities, find the best one conditionned to the frontier node type and update the tree
        assert(self.current_node.action_type == "apply")
        rule_choices = self.grammar.rules_from_node()
        selected_probs = rule_probs[rule_choices]
        pred_rule = rule_choices[np.argmax(selected_probs)]
        child_nodes = self.grammar.get_children(r)
        self.current_node.set_rule(pred_rule, child_nodes)
        self.need_to_move=True

    def set_token(self, token, tktype, tkindex):
        # set a token, and its child if it was not an eos token
        assert(self.current_node.action_type == "gen")
        self.current_node.set_token(token, tktype, tkindex)
        end = (tkindex==grammar.get_vocab_index("<eos>"))
        if(end):
            self.need_to_move = True
        else:
            self.need_to_move = False



class OracleTree(Tree):
    # Golden rees used in training
    def __init__(self, grammar, indexedString):
        # create from an ast
        super.__init__(self, grammar)
        Tree.make_from_dict(grammar, indexedString)
        self.current_token_index=0

    def get_oracle_rule(self):
        # returns the correct rule for loss computation in the model
        assert(self.current_action_type == "apply")
        self.need_to_move=True
        return self.current_node.rule


    def get_oracle_token(self):
        # returns the correct token for loss computation in the model
        assert(self.current_node.action_type == "gen")
        token,tktype,tkindex = self.current_node.get_token_info(self.current_token_index)
        if(tkindex==grammar.get_vocab_index("<eos>")):
            self.need_to_move=True
            self.current_token_index=0
        else:
            self.need_to_move=False
            self.current_token_index+=1
        return tktype,tkindex
