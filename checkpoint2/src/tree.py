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
        st = SubTree(frontier_node_type=grammar.index("root"), parent=None)
        st.time_step = 0
        return st

    def is_built(self):
        return self.children != None and self.rule != None


class Tree:
    def __init__(self, grammar):
        self.grammar = grammar
        self.current_action_type = None
        self.current_node = None

    def move(self):
        self.current_node = self.current_node.next()

    def get_node_type(self):
        return self.current_node.frontier_node_type

    def get_action_type(self):
        return self.current_action_type

    def get_parent_time(self):
        return self.current_node.parent.time_step

    def has_ended(self):
        return self.current_node == None

    @static
    def make_from_ast(grammar, astTree):
        # TODO
        pass


class BuildingTree(Tree):
    # Tree of dynet vectors for nodes and time steps
    def __init__(self, grammar):
        super.__init__(self, grammar)
        self.current_node = SubTree.root()
        self.current_action_type = "apply"

    def set_rule(self, rule_index):
        assert(self.current_action_type == "apply")
        # TODO
        pass

    def set_token(self, token):
        assert(self.current_action_type == "gen")
        # TODO
        pass


class OracleTree(Tree):
    # Tree of dynet vectors for nodes and time steps
    def __init__(self, grammar, astTree):
        super.__init__(self, grammar)
        self.current_node = Tree.make_from_ast(grammar, astTree)

    def get_oracle_rule(self):
        assert(self.current_action_type == "apply")
        return self.current_node.rule

    def get_oracle_token(self):
        assert(self.current_action_type == "gen")
        return self.current_node.token
