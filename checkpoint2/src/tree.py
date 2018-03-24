import numpy as np

class SubTree:
    def __init__(self, node_type,label, parent):
        self.node_type = node_type
        self.label=label
        self.action_type = None
        self.rule = None
        self.tokens= None
        self.tokens_type = None
        self.tokens_vocab_index = None
        self.tokens_query_index = None
        self.time_step = None
        self.parent = parent
        self.children = None
        self.child_to_explore = 0

    def next(self, old_time=None):
        # recursive function to return the correct next node
        # assert(self.is_built())
        # print(len(self.children),self.child_to_explore)
        if(old_time is None):
            old_time = self.time_step
        if(self.child_to_explore < len(self.children)):
            child = self.children[self.child_to_explore]
            self.child_to_explore += 1
            child.time_step = old_time+1
            #print(child.node_type,child.parent,child.time_step, child.rule,child.tokens)
            # assert(child.is_well_built())
            return child
        else:
            self.child_to_explore = 0
            self.time_step = None
            if self.parent == None:
                return None
            else:
                return self.parent.next(old_time)

    @staticmethod
    def root(grammar):
        # create a subtree with only the root node
        st = SubTree(node_type=grammar.get_node_index("'root'"), label=None, parent=None)
        st.time_step = 0
        st.children=[]
        return st

    def is_well_built(self):
        # check that the node has an action
        return self.children != None and ((self.action_type == "apply" and self.rule != None) or (self.action_type == "gen" and len(self.tokens)>0))


    def get_token_info(self,i, max_copy_index):
        # print(i,len(self.tokens))
        assert(i<len(self.tokens))
        token = self.tokens[i]
        tktype = self.tokens_type[i]
        tkvocindex = self.tokens_vocab_index[i]
        tkcopindex = self.tokens_query_index[i]
        if(tkcopindex is not None and tkcopindex>=max_copy_index):
            tkcopindex=None
        return tkvocindex,tkcopindex,tktype=="vocab"

    def set_rule(self, rule, child_nodes):
        # set a rule and children
        self.rule = rule
        for node_type, label, action_type in child_nodes:
            st = SubTree(parent=self, node_type=node_type, label=label)
            st.set_action_type(action_type)
            st.children = []
            self.children.append(st)

    def set_token(self, token, tktype, tkvocabindex,tkqueryindex):
        self.tokens.append(token)
        self.tokens_type.append(tktype)
        self.tokens_vocab_index.append(tkvocabindex)
        self.tokens_query_index.append(tkqueryindex)

    def set_action_type(self,action_type):
        self.action_type=action_type
        if(self.action_type=="gen"):
            self.tokens=list()
            self.tokens_type=list()
            self.tokens_vocab_index=list()
            self.tokens_query_index=list()


    @staticmethod
    def from_dict(d,parent=None):
        node_type = d["node_type"]
        label = d["label"]
        st = SubTree(parent=parent, node_type=node_type, label=label)
        st.children=[]
        action_type = d["action_type"]
        st.set_action_type(action_type)
        if(action_type=="apply"):
            assert(d["rule"] is not None)
            st.rule = d["rule"]
            children=d["children"]

            for child_d in children:
                child = SubTree.from_dict(child_d,parent=st)

                st.children.append(child)
        else:
            assert(action_type == "gen")
            st.tokens=d["tokens"]
            st.tokens_type=d["tokens_type"]
            st.tokens_vocab_index = d["tokens_vocab_index"]
            st.tokens_query_index = d["tokens_query_index"]

        assert(st.is_well_built())
        return st

    def to_dict(self):
        d = dict()
        d["node_type"]=self.node_type
        d["label"]=self.label
        d["action_type"]=self.action_type
        if(self.action_type=="apply"):
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
        self.root_node = None
        self.need_to_move = True
        self.current_token_index = 0

    def move(self):
        # shift the node to the next one, and specify the action type associated to its frontier node
        # print(self.need_to_move)
        #print("old node :",self.current_node,self.current_node.node_type,self.current_node.time_step,self.current_node.parent, self.current_node.children, self.current_node.tokens)
        assert(self.current_node.is_well_built())
        # print(self.current_node.action_type)
        if(self.need_to_move):
            # print("move from 1")
            st = self.current_node.next()

            if st is None:
                # print("End of tree")
                return False
            self.current_node = st
            assert(self.current_node.action_type==self.grammar.action_type(self.current_node.node_type))
        # print("new node :",self.current_node,self.current_node.node_type,self.current_node.time_step,self.current_node.parent, self.current_node.children, self.current_node.tokens)
        return True

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

    def to_dict(self, grammar):
        d = self.current_node.toDict()
        return d

    def set_to_root(self):
        self.current_node = self.root_node
        self.current_node.time_step = 0

class BuildingTree(Tree):
    # Trees used in prediction
    def __init__(self, grammar, query):
        # create a tree with only a root node
        super(BuildingTree,self).__init__(grammar)
        self.current_node = SubTree.root(grammar)
        self.root_node = self.current_node
        self.current_node.action_type = "apply"
        self.sentence = query

    def pick_and_get_rule(self, rules_probs):
        # from the rule probabilities, find the best one conditionned to the frontier node type and update the tree
        assert(self.current_node.action_type == "apply")
        rule_choices = self.grammar.rules_from_node(self.current_node.node_type)
        # print(rule_choices,type(rule_choices), rules_probs, type(rules_probs))
        selected_probs = np.array(rules_probs)[rule_choices]
        pred_rule = rule_choices[np.argmax(selected_probs)]
        child_nodes = self.grammar.get_children(pred_rule)
        self.current_node.set_rule(pred_rule, child_nodes)
        self.need_to_move=True
        return pred_rule

    def set_token(self, tktype, tkindex):
        # print(tkindex)
        # set a token, and its child if it was not an eos token
        assert(self.current_node.action_type == "gen")
        end = (tkindex==self.grammar.get_vocab_index("'<eos>'"))
        if(tktype=="vocab"):
            token = self.grammar.get_vocab(tkindex)
            tk_vocab_index = tkindex
            tk_query_index = None
        else:
            assert(tktype=="copy")
            token = self.sentence[tkindex]
            tk_vocab_index = None
            tk_query_index = tkindex

        self.current_node.set_token(token, tktype, tk_vocab_index,tk_query_index)

        if(end):
            self.need_to_move = True
        else:
            self.need_to_move = False




class OracleTree(Tree):
    # Golden rees used in training
    def __init__(self, grammar):
        # create from an ast
        # print(type(grammar))
        super(OracleTree,self).__init__(grammar)
        self.sentence=None

    def set_query(self,sentence):
        self.sentence=sentence

    def get_oracle_rule(self):
        # returns the correct rule for loss computation in the model
        # print(self.get_action_type())
        assert(self.get_action_type() == "apply")
        assert(self.current_node.is_well_built())
        assert(self.current_node.rule is not None)
        self.need_to_move=True
        return self.current_node.rule

    @staticmethod
    def make_from_dict(grammar, d):
        t = OracleTree(grammar)
        t.current_node = SubTree.from_dict(d)
        t.current_node.time_step=0
        t.root_node = t.current_node
        return t

    def get_oracle_token(self):
        #print(self.current_node)
        # returns the correct token for loss computation in the model
        assert(self.current_node.action_type == "gen")
        tkvocindex,tkcopindex,tkinvocab = self.current_node.get_token_info(self.current_token_index, max_copy_index = len(self.sentence))
        #print(len(self.current_node.tokens),"tokens, at number",self.current_token_index,":",self.current_node.tokens[self.current_token_index])
        if(tkvocindex==self.grammar.get_vocab_index("'<eos>'")):
            self.need_to_move=True
            self.current_token_index=0
        else:
            self.need_to_move=False
            self.current_token_index+=1
        if(tkvocindex is None):
            tkvocindex=self.grammar.get_vocab_index("'<unk>'")
        return tkvocindex,tkcopindex,tkinvocab

    def set_query(self,query):
        self.sentence=query
