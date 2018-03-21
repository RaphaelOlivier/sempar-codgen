import dynet as dy
import numpy as np
from collections import namedtuple
import dynet as dy
import time
import random
import math
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
import tree as Tree

args = namedtuple('args', [
	'vocabLengthSource',
	'vocabLengthTarget',
	'targetIndexer',
	'targetRuleDictionnary',
	'targetGenDictionary',
	'numLayer',
	'embeddingApplySize',
	'embeddingGenSize',
	'embeddedNodeType'
	'hiddenSize',
	'attSize',
	'dropout',
	'learningRate',
	])(
	#not written,
	#vocab_target,
	#targetIndexer,
	#targetDict,
	1, # check when num_layers is 50
	128,
	128,
	64,
	256,
	32,
	0,
	0.001,
	)

class ASTNNModule:
	def __init__(self,args):

		self.targetIndexer = args.targetIndexer
		self.targetRuleDictionnary = args.targetRuleDictionnary
		self.targetGenDictionary = args.targetGenDictionary
		self.vocabLengthSource = args.vocabLengthSource
		self.vocabLengthTarget = args.vocabLengthTarget

		self.unkTarget = self.targetIndexer["<unk>"]
		self.eosTarget = self.targetIndexer["</s>"]

		# parameters for the model
		self.numLayer = args.numLayer
		self.embeddingApplySize = args.embeddingApplySize
		self.embeddingGenSize = args.embeddingGenSize
		self.embeddingNodeType = args.embeddingNodeType
        self.hiddenSize = args.hiddenSize
        self.attSize = args.attSize
        self.dropout = args.dropout
		self.embeddingRuletypeSize = 2
		self.learningRate= args.learningRate
        self.ASTmodel = dy.ParameterCollection()
		self.trainer = dy.AdamTrainer(self.ASTmodel, alpha=self.learningRate)

		# creating source embedding matrix
        self.sourceLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthSource, self.embeddingApplySize))
        # creating target embedding matrix 
        #self.targetLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthTarget, self.embeddingGenSize))
        # action embeddging matrix
        self.actionRuleLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthActionRule, self.embeddingApplySize))
        # for node type
        self.nodeTypeLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthNodes, self.embeddingNodeType))

        self.gentokenLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthTarget, self.embeddingGenSize))
        # adding paramteters to the AST Neural Network
		self.attentionSource = self.model.add_parameters((self.attSize, self.hiddenSize * 2))
        self.attentionTarget = self.model.add_parameters((self.attSize, self.numLayer*self.hiddenSize * 2))
        self.attentionParameter = self.model.add_parameters((1, self.att_size)) 
        self.w_softmax = self.model.add_parameters((self.vocab_length_target, self.hidden_size)) # should change whe hidden layers increase 
        self.b_softmax = self.model.add_parameters((self.vocab_length_target))

        # initializing the encoder and decoder
        self.forward_encoder = dy.LSTMBuilder(self.num_layer, self.embedding_size, self.hidden_size, self.model)
        self.backward_encoder = dy.LSTMBuilder(self.num_layer, self.embedding_size, self.hidden_size, self.model)

        self.eos_target = self.targetIndexer['root']
        self.unk_target = self.targetIndexer['<unk>']
        # embedding size + (previous action embedding + context vector + node type mebedding + pare
        #	net feeding )
		# parent feeding - hidden states of parent action + embedding of parent action 

        self.decoder = dy.VanillaLSTMBuilder(self.num_layer, self.hidden_size * 2 +self.embedding_size*5, self.hidden_size, self.model)

    def encoder(self, nl):
    	# nl - natural langauge
    	# forward LSTM
    	forward_hidden_state = self.forward_encoder.initial_state()
    	forward_vectors = []

    	for word in nl:
    		forward_hidden_state = forward_hidden_state.add_input(word)
    		# next time step in the RNN
    		output = forward_hidden_state.output()
    		forward_vectors.append(output)

        # backward LSTM

        backward_hidden_state = self.backward_encoder.initial_state()
        reversed_nl = list(reversed(nl))
        backward_vectors = []

        for word in reversed_nl:
        	backward_hidden_state = backward_hidden_state.add_input(word)
        	# next time step
        	output = backward_hidden_state.output()
        	backward_vectors.append(output)

        # to match the input sequence
        backward_vectors = list(reversed(backward_vectors))

        encoder_hidden_states = [dy.concatenate(list(x)) for x in zip(forward_vectors, backward_vectors)]

        return encoder_hidden_states

    def forward_prop(self, input_sentence, output_sentence=None, mode="predict"):
        dy.renew_cg()

        embedded_input_sentence = []
        for word in sentence:
        	embedded_input_sentence.append(self.source_lookup[word])
        encoded = self.encoder(embedded_input_sentence)

        if(mode == "train"):
            self.set_dropout()
            loss = self.decode_to_loss(encoded, output_sentence)
            return loss

        if(mode == "validate"):
            self.disable_dropout()
            loss = self.decode_to_loss(encoded, output_sentence)
            return loss

        if(mode == "predict"):
            self.disable_dropout()
            output_sentence = self.decode_to_prediction(encoded, 2*len(input_sentence))
            return output_sentence

	def set_dropout(self):
        self.forward_encoder.set_dropout(self.dropout)
        self.backward_encoder.set_dropout(self.dropout)
        self.decoder.set_dropout(self.dropout)

    def disable_dropout(self):
        self.forward_encoder.disable_dropout()
        self.backward_encoder.disable_dropout()
        self.decoder.disable_dropout()

	def backward_prop_and_update_parameters(self, loss):
        loss.backward()
        self.trainer.update()

    def save(self, path):
        self.model.save(path)

    def get_learning_rate(self):
    	print ("learning rate" + str(self.learning_rate))
        return self.learning_rate

    def reduce_learning_rate(self, factor):
        self.learning_rate = self.learning_rate/factor
        self.trainer.learning_rate = self.trainer.learning_rate/factor

    def parent_feed(self, parent_action_hidden_state, parent_action_embedding ):

    	return [parent_action_hidden_state, parent_action_embedding]

    def decoder_state(self, previous_action, context_vector, parent_action, current_frontier_node_type  ):
    	new_input = dy.concatenate(
            [previous_action, context_vector, parent_action, current_frontier_node_type])
        new_state = previous_state.add_input(new_input)
        return new_state

    def get_att_context_vector(self, scr_output_matrix , current_state, fixed_attentional_component):

    	w1_att_tgt = dy.parameter(self.attentionTarget)
    	w2_att = dy.parameter(self.attentionParameter)

    	target_output_embedding = dy.concatenate(list(current_state.s()))

        a_t = dy.transpose(
            w2_att * dy.tanh(dy.colwise_add(fixed_attentional_component, w1_att_tgt * tgt_output_embedding)))
        alignment = dy.softmax(a_t)

        context_vector = src_output_matrix * alignment 

        return context_vector

    def decode_to_loss(self, vectors, output):

    	goldenTree = Tree.OracleTree(output) # check this too


    	w = dy.parameter(self.w_softmax) # the weight matrix 
    	b = dy.parameter(self.b_softmax)

    	w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

    	attentional_component = w1 * encoded_states

    	encoded_states = dy.concatenate_cols(encoded) # used for context vecor

    	prev_action_embedding = self.actionRuleLookup(self.eos_target)

    	# parent_action - 2* 256
    	# context vector - 2*256 
    	# node type - 64  - need to change this 

    	decoder_states = [] # used in LSTM models for parent feed
    	decoder_action = [] 

    	current_state = self.decoder.initial_state().add_input(dy.concatenate \
    		([dy.vecInput(self.hiddenSize*3 + self.embeddingNodeType + self.embeddingApplySize), prev_action_embedding]))

    	decoder_states.append(current_state)
    	resultant_parse_tree = ""

    	decoder_state.append(prev_action_embedding) # storing the hidden states for further prediction

    	for i in range(max_length):
    		context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
    		parent_time =  tree.get_parent_time()
    		frontier_node_type_embedding = self.nodeTypeLookup(tree.get_frontier_node_type())
    		parent_action = self.parent_feed(decoder_states[parent_time], decoder_action[parent_time])

            current_state = self.decoder_state(
                current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)

        decoder_states.append(current_state)

        action_type = tree.get_action_type()  # assuming the tree module manages to get the node type

        if action_type == "apply":
            current_apply_action_embedding = self.get_apply_action_embedding(current_state)  # affine tf
            golden_next_rule = goldentree.get_oracle_rule()
            item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, golden_next_rule)
            prev_action_embedding = self.actionRuleLookup(golden_next_rule)
            decoder_action.append(prev_action_embedding)

        if action_type == "gen":
            current_action_embedding = self.get_gen_embedding(current_state)  # affine tf over gen vocab
            # todo
            goldentoken = goldenTree.get_oracle_token()
            item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, goldentoken)
            prev_action_embedding = self.gentokenLookup(goldentoken)
            decoder_action.append(prev_action_embedding)

        	if(token == self.eos_target):
            	return 




    def decoder_to_predict(self, encoded, max_length):


    	# start witht the root node at time to
        # it either generate apply rule - chooses a rule  - from closed set
        # it picks terminal - and generates a terminal rule - from open set
        # expand the current node - fronteir noder at step t 
        # given a set of rules (r) - it chooses a rule r from subset that has 
        # head matching the type nft.
        # then appends all child nodes specified by selected production
        # if a variable node is added to derivation - switches to GENTOKEN
        # is applu rule nodes -  add more children to the derivation
        # reach a fronteir node that corresponds to a variable type - GEN TOKEN 
        # used to fill the nodes with values
        # <'\n'> used to close the terminal node - apply gen token can happen multiple times
        # terminal nodes can be copied or choosen from the terminal vocab
        # use bi-directional LSTM - encoder
        # RNN - decoder 
        # each action step in the grammar model - grounds to time step in
        # decoder RNN 
        # sequence of time steps - unrolling RNN time steps
        # Vaniall LSTM - decoder
        # hidden state - st -> vector concatenation -> 
        # st -> internal hidden state
        # a function of - previous action, context vectore, input encodings via 
        # soft attention and parent action , note type embedding of current 
        # frontier type 
        # Action embeddingd - Wr and WG
        # ct - context vectore using for current action prediction
        # Parent action - also used pt - concat of hidden state of parent action
        # spt and embedding of parent action 
        # the probability of applying a rule r is softmax over the formula given
        # the probability of gen token - marginal probability - 
        # all embeddings is 128
        # one of them is 64 -  node one
        # RNN - 256 states and 50 hidden layers
        # lots of dropouts 


    	tree = Tree.BuildingTree()

    	w = dy.parameter(self.w_softmax) # the weight matrix 
    	b = dy.parameter(self.b_softmax)

    	w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

    	attentional_component = w1 * encoded_states

    	encoded_states = dy.concatenate_cols(encoded) # used for context vecor

    	prev_action_embedding = self.actionRuleLookup(self.eos_target)

    	# parent_action - 2* 256
    	# context vector - 2*256 
    	# node type - 64  - need to change this 

    	decoder_states = []
    	decoder_action = []

    	current_state = self.decoder.initial_state().add_input(dy.concatenate \
    		([dy.vecInput(self.hiddenSize*3 + self.embeddingNodeType + self.embeddingApplySize), prev_action_embedding]))

    	decoder_states.append(current_state)
    	resultant_parse_tree = ""

    	decoder_state.append(prev_action_embedding) # storing the hidden states for further prediction

    	for i in range(max_length):
    		context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
    		parent_time =  tree.get_parent_time()
    		frontier_node_type_embedding = self.nodeTypeLookup(tree.get_frontier_node_type())
    		parent_action = self.parent_feed(decoder_states[parent_time], decoder_action[parent_time])

            current_state = self.decoder_state(
                current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)

        decoder_states.append(current_state)

        action_type = tree.get_action_type()  # assuming the tree module manages to get the node type

        if action_type == "apply":
            current_apply_action_embedding = self.get_apply_action_embedding(current_state)  # affine tf
            rule_probs = (dy.log_softmax(current_apply_action_embedding)).value()
            action = to_apply_rule(action_embedding)  # argmax
            next_rule = tree.pick_and_set_rule(list(rule_probs))
            prev_action_embedding = self.actionRuleLookup(next_rule)
            decoder_action.append(prev_action_embedding)

        if action_type == "gen":
            current_action_embedding = self.get_gen_embedding(current_state)  # affine tf over gen vocab
            # todo
            token = tree.set_token(rule_probs)
            prev_action_embedding = self.gentokenLookup(token)
            decoder_action.append(prev_action_embedding)

        	if(token == self.eos_target):
            	return 
