class SubTree:
    def __init__(self, frontier_node_type, parent):
        self.frontier_node_type = frontier_node_type
        self.rule = None
        self.token = None
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

    def set_token(self, token, end=False):
        # set a token, and child if any
        self.token = token
        if end:
            self.children = []
        else:
            self.children = [SubTree(parent=self, frontier_node_type=self.frontier_node_type)]


class Tree:
    def __init__(self, grammar):
        # abstract class for trees
        self.grammar = grammar
        self.current_action_type = None
        self.current_node = None

    def move(self):
        # shift the node to the next one, and specify the action type associated to its frontier node
        self.current_node = self.current_node.next()
        self.current_action_type = grammar.action_type(self.current_node.frontier_node_type)

    def get_node_type(self):
        # value needed by the model
        return self.current_node.frontier_node_type

    def get_action_type(self):
        return self.current_action_type

    def get_parent_time(self):
        # value needed by the model
        return self.current_node.parent.time_step

    def has_ended(self):
        # to know if decoding is over (current node is None)
        return self.current_node == None

    @static
    def make_from_ast(grammar, astTree):
        # TODO
        pass

    def to_ast(self):
        # TODO
        pass


class BuildingTree(Tree):
    # Trees used in prediction
    def __init__(self, grammar):
        # create a tree with only a root node
        super.__init__(self, grammar)
        self.current_node = SubTree.root()
        self.current_action_type = "apply"

    def pick_and_set_rule(self, rules_probs):
        # from the rule probabilities, find the best one conditionned to the frontier node type and update the tree
        assert(self.current_action_type == "apply")
        rule_choices = self.grammar.rules_from_frontier_node()
        selected_probs = rule_probs[rule_choices]
        pred_rule = rule_choices[np.argmax(selected_probs)]
        child_nodes = self.grammar.get_children(r)
        self.current_node.set_rule(pred_rule, child_nodes)

    def set_token(self, token):
        # set a token, and its child if it was not an eos token
        assert(self.current_action_type == "gen")
        end = (token=grammar.index("eos"))
        self.current_node.set_token(token, end)


class OracleTree(Tree):
    # Golden tees used in training
    def __init__(self, grammar, astTree):
        # create from an ast
        super.__init__(self, grammar)
        self.current_node = Tree.make_from_ast(grammar, astTree)

    def get_oracle_rule(self):
        # returns the correct rule for loss computation in the model
        assert(self.current_action_type == "apply")
        return self.current_node.rule

    def get_oracle_token(self):
        # returns the correct token for loss computation in the model
        assert(self.current_action_type == "gen")
        return self.current_node.token
