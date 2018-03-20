class SubTree:
    def __init__(self, frontier_node_type, parent):
        self.frontier_node_type = frontier_node_type
        self.action_type = None
        self.rule = None
        self.token = None
        self.token_type = None
        self.token_index = None
        self.time_step = None
        self.parent = None
        self.children = None
        self.child_to_explore = 0

    def next(self, old_time=None):
        # recursive function to return the correct next node
        assert(self.is_built())
        if(old_time == None):
            old_time = self.time_step
        if(self.child_to_explore < len(children))
            child = self.children[self.child_to_explore]
            self.child_to_explore += 1
            child.time_step = old_time+1
            return child
        else if self.parent == None:
            return None
        else return self.parent.next(old_time)

    @static
    def root():
        # create a subtree with only the root node
        st = SubTree(frontier_node_type=grammar.index("root"), parent=None)
        st.time_step = 0
        return st

    def is_built(self):
        # check that the node has an action
        return self.children != None and (self.rule != None or self.token != None)

    def set_rule(self, rule, child_nodes):
        # set a rule and children
        self.rule = rule
        self.children = []
        for node in child_nodes:
            self.children = SubTree(parent=self, frontier_node_type=node)

    def set_token(self, token, tktype=None, tkindex=None, end=False):
        # set a token, and child if any
        self.token = token
        self.token_type = tktype
        self.token_index = tkindex
        if end:
            self.children = []
        else:
            self.children = [SubTree(parent=self, frontier_node_type=self.frontier_node_type)]

    @static
    def fromDict(d,parent=None):
        node_type = d["node_type"]
        st = SubTree(parent=parent, frontier_node_type=node_type)
        action_type = d["action_type"]
        if(action_type=="apply_rule"):
            st.rule = d["rule"]
        elif(action_type=="gen_vocab"):
            st.action_type="gen"
            st.token_type="vocab"
            st.token_index = d["token_index"]

        else:
            assert(action_type=="gen_copy")
            st.action_type="gen"
            st.token_type="copy"
            st.token_index = d["token_index"]
            st.token=d["token"]

        children=d["children"]
        if(children==[]):
            st.children==[]
        else:
            st.children==[]
            for child_d in children:
                st.children.append(fromDict(child_d,parent=st))
        return st

    def toDict(self):
        d = dict()
        d["node_type"]=self.frontier_node_type
        if(self.action_type=="apply_rule"):
            d["action_type"]="apply_rule"
            d["rule"]=self.rule
        else:
            assert(self.action_type=="gen"):
            if(self.token_type=="vocab"):
                d["action_type"]="gen_vocab"
                d["token_index"]=self.token_index
            else:
                assert(self.token_type=="copy")
                d["action_type"]="gen_copy"
                d["token_index"]=self.token_index
                d["token"]=self.token
        d["children"]=[]
        for child in self.children:
            d["children"].append(child.toDict())

        return d

class Tree:
    def __init__(self, grammar):
        # abstract class for trees
        self.grammar = grammar
        self.current_node = None

    def move(self):
        # shift the node to the next one, and specify the action type associated to its frontier node
        self.current_node = self.current_node.next()
        self.current_node.action_type = grammar.action_type(self.current_node.frontier_node_type)

    def get_node_type(self):
        # value needed by the model
        return self.current_node.frontier_node_type

    def get_action_type(self):
        return self.current_node.action_type

    def get_parent_time(self):
        # value needed by the model
        return self.current_node.parent.time_step

    def has_ended(self):
        # to know if decoding is over (current node is None)
        return self.current_node == None

    @static
    def make_from_dict(grammar, d):
        t = Tree(grammar)
        t.current_node = SubTree.from_dict(d)
        t.current_action_type = grammar.action_type(t.current_node.frontier_node_type)

    def to_json(self, grammar):
        # TODO
        pass


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
        rule_choices = self.grammar.rules_from_frontier_node()
        selected_probs = rule_probs[rule_choices]
        pred_rule = rule_choices[np.argmax(selected_probs)]
        child_nodes = self.grammar.get_children(r)
        self.current_node.set_rule(pred_rule, child_nodes)

    def set_token(self, tktype, tkindex, token):
        # set a token, and its child if it was not an eos token
        assert(self.current_node.action_type == "gen")
        end = (tkindex==grammar.index("eos"))
        self.current_node.set_token(token, end)
        self.current_node.set_token(token, tktype, tkindex)


class OracleTree(Tree):
    # Golden rees used in training
    def __init__(self, grammar, indexedString):
        # create from an ast
        super.__init__(self, grammar)
        Tree.make_from_indexed_string(grammar, indexedString)

    def get_oracle_rule(self):
        # returns the correct rule for loss computation in the model
        assert(self.current_action_type == "apply")
        return self.current_node.rule

    def get_oracle_token(self):
        # returns the correct token for loss computation in the model
        assert(self.current_action_type == "gen")
        return (self.current_node.token_type, self.current_node.token_index)
