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

    def next(self):
        # recursive function to return the correct next node
        # assert(self.is_built())
        # print(len(self.children),self.child_to_explore)
        # if(old_time is None):
        #     old_time = self.time_step
        if(self.child_to_explore < len(self.children)):
            child = self.children[self.child_to_explore]
            self.child_to_explore += 1
            #child.time_step = old_time+1
            #print(child.node_type,child.parent,child.time_step, child.rule,child.tokens)
            # assert(child.is_well_built())
            return child
        else:
            self.child_to_explore = 0
            self.time_step = None
            if self.parent == None:
                return None
            else:
                return self.parent.next()

    @staticmethod
    def root(grammar):
        # create a subtree with only the root node
        st = SubTree(node_type=grammar.get_node_index("root"), label=None, parent=None)
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

    def copy(self, parent):
        st = SubTree(self.node_type,self.label,parent)
        st.action_type = self.action_type
        st.rule = self.rule
        if(st.action_type == "gen"):
            st.tokens= self.tokens[:]
            st.tokens_type = self.tokens_type[:]
            st.tokens_vocab_index = self.tokens_vocab_index[:]
            st.tokens_query_index = self.tokens_query_index[:]
        st.time_step = None
        st.children = [c.copy(st) for c in self.children]
        st.child_to_explore = 0
        return st


    @staticmethod
    def from_dict(d,parent=None):
        total_length = 0
        node_type = d["node_type"]
        label = d["label"]
        st = SubTree(parent=parent, node_type=node_type, label=label)
        st.children=[]
        action_type = d["action_type"]
        st.set_action_type(action_type)
        if(action_type=="apply"):
            total_length = 1
            assert(d["rule"] is not None)
            st.rule = d["rule"]
            children=d["children"]

            for child_d in children:
                child, length = SubTree.from_dict(child_d,parent=st)
                total_length+=length

                st.children.append(child)
        else:
            assert(action_type == "gen")
            st.tokens=d["tokens"]
            st.tokens_type=d["tokens_type"]
            st.tokens_vocab_index = d["tokens_vocab_index"]
            st.tokens_query_index = d["tokens_query_index"]
            total_length = len(st.tokens) - 1

        assert(st.is_well_built())
        return st, total_length

    def to_dict(self):
        d = dict()
        d["node_type"]=self.node_type
        d["label"]=self.label
        d["action_type"]=self.action_type
        if(self.action_type=="apply"):
            d["rule"]=self.rule
            # print(type(d["rule"]))
        else:
            assert(self.action_type=="gen")
            d["tokens_vocab_index"]=self.tokens_vocab_index
            d["tokens_query_index"]=self.tokens_query_index
            d["tokens_type"]=self.tokens_type
            d["tokens"]=self.tokens
            #assert(self.tokens[0] is not None)
        d["children"]=[]
        for child in self.children:
            d["children"].append(child.to_dict())
        # print(self.node_type,self.tokens)
        return d

class Tree:
    def __init__(self, grammar, verbose = False):
        # abstract class for trees
        self.grammar = grammar
        self.current_node = None
        self.root_node = None
        self.need_to_move = True
        self.current_token_index = 0
        self.verbose = verbose
        self.current_time_step=0


    def go_to_root(self):
        while(self.current_node.parent is not None):
            self.current_node = self.current_node.parent
        self.current_time_step=0



    def move(self):

        # shift the node to the next one, and specify the action type associated to its frontier node
        # print(self.need_to_move)
        #print("old node :",self.current_node,self.current_node.node_type,self.current_node.,self.current_node.parent, self.current_node.children, self.current_node.tokens)
        self.current_node.time_step = self.current_time_step
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

        self.current_time_step +=1
        return True

    def get_node_type(self):
        # value needed by the model
        return self.current_node.node_type

    def get_action_type(self):
        return self.current_node.action_type

    def get_parent_time(self):
        # value needed by the model
        if(self.current_node.action_type=="gen" and self.current_token_index > 0):
            return self.current_node.time_step
        else:
            return self.current_node.parent.time_step

    def has_ended(self):
        # to know if decoding is over (current node is None)
        return self.current_node == None

    def to_dict(self, grammar):
        d = self.current_node.to_dict()
        return d

class BuildingTree(Tree):
    # Trees used in prediction
    def __init__(self, grammar, query, verbose=False):
        # create a tree with only a root node
        super(BuildingTree,self).__init__(grammar,verbose)
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
        pred_rule = rule_choices[np.argmax(selected_probs)].item()
        child_nodes = self.grammar.get_children(pred_rule)
        self.current_node.set_rule(pred_rule, child_nodes)
        self.need_to_move=True
        if(self.verbose):
            print("new rule :",self.grammar.get_rule(pred_rule))
        return pred_rule

    def set_token(self, tktype, tkindex):
        # print(tkindex)
        # set a token, and its child if it was not an eos token
        tkindex = tkindex.item()
        assert(self.current_node.action_type == "gen")
        end = (tkindex==self.grammar.get_vocab_index("<eos>"))
        if(tktype=="vocab"):
            token = self.grammar.get_vocab(tkindex)
            tk_vocab_index = tkindex
            tk_query_index = None
        else:
            assert(tktype=="copy")
            token = self.sentence[tkindex]
            tk_vocab_index = None
            tk_query_index = tkindex
        if(self.grammar.get_node_type(self.current_node.node_type)=='int' and not end): # we need a number
            #print(self.current_node.node_type)
            try:
                token = int(token)
            except:
                token=0

        self.current_node.set_token(token, tktype, tk_vocab_index,tk_query_index)
        if(self.verbose):
            print("new token :",token)
        if(end):
            self.need_to_move = True
            self.current_token_index=0
        else:
            self.need_to_move = False
            self.current_token_index+=1




class OracleTree(Tree):
    # Golden rees used in training
    def __init__(self, grammar, verbose = False):
        # create from an ast
        # print(type(grammar))
        super(OracleTree,self).__init__(grammar,verbose)
        self.sentence=None
        self.length = 0

    def set_query(self,sentence):
        self.sentence=sentence

    def get_oracle_rule(self):
        # returns the correct rule for loss computation in the model
        # print(self.get_action_type())
        assert(self.get_action_type() == "apply")
        assert(self.current_node.is_well_built())
        assert(self.current_node.rule is not None)
        self.need_to_move=True
        if(self.verbose):
            print("new rule :",self.grammar.get_rule(self.current_node.rule))
        return self.current_node.rule

    @staticmethod
    def make_from_dict(grammar, d):
        t = OracleTree(grammar)
        t.current_node, length = SubTree.from_dict(d)
        t.current_node.time_step=0
        t.root_node = t.current_node
        t.length = length
        return t

    def get_oracle_token(self):
        #print(self.current_node)
        # returns the correct token for loss computation in the model
        assert(self.current_node.action_type == "gen")
        tkvocindex,tkcopindex,tkinvocab = self.current_node.get_token_info(self.current_token_index, max_copy_index = len(self.sentence))
        if(self.verbose):
            print("new token :",self.current_node.tokens[self.current_token_index],", voc index ",self.current_node.tokens_vocab_index[self.current_token_index])

        if(tkvocindex==self.grammar.get_vocab_index("<eos>")):
            self.need_to_move=True
            self.current_token_index=0
        else:
            self.need_to_move=False
            self.current_token_index+=1
        if(tkvocindex is None):
            tkvocindex=self.grammar.get_vocab_index("<unk>")
        return tkvocindex,tkcopindex,tkinvocab

    def set_query(self,query):
        self.sentence=query


    def copy(self, verbose = False):
        tc = OracleTree(self.grammar, verbose)
        tc.current_node = self.current_node.copy(None)
        tc.current_node.time_step =0
        tc.length = self.length
        return tc
