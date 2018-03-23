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

args = namedtuple('args', ['numLayer','embeddingSourceSize','embeddingApplySize','embeddingGenSize','embeddingNodeSize',
				'hiddenSize','attSize','dropout','learningRate'])(50,128,128,128,64,256,32,0,0.001)

class ASTNet:
	def __init__(self, args, vocabLengthSource, vocabLengthActionRule, vocabLengthNodes, vocabLengthTarget):

		self.vocabLengthSource = vocabLengthSource
		self.vocabLengthActionRule = vocabLengthActionRule
		self.vocabLengthNodes = vocabLengthNodes
		self.vocabLengthTarget = vocabLengthTarget

		# parameters for the model
		self.numLayer = args.numLayer
		self.embeddingApplySize = args.embeddingApplySize
		self.embeddingGenSize = args.embeddingGenSize
		self.embeddingNodeSize = args.embeddingNodeSize
		self.embeddingSourceSize = args.embeddingSourceSize
		self.hiddenSize = args.hiddenSize
		self.attSize = args.attSize
		self.dropout = args.dropout
		self.embeddingRuletypeSize = 2
		self.learningRate= args.learningRate


		self.ASTmodel = dy.ParameterCollection()
		self.trainer = dy.AdamTrainer(self.ASTmodel, alpha=self.learningRate)

		# source lookup
		self.source_lookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthSource, self.embeddingSourceSize))

		# action embeddging matrix
		self.actionRuleLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthActionRule, self.embeddingApplySize))

		# for node type lookup
		self.nodeTypeLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthNodes, self.embeddingNodeSize))

		# gor gen type lookup
		self.gentokenLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthTarget, self.embeddingGenSize))


		# adding paramteters to the AST Neural Network
		self.attentionSource = self.ASTmodel.add_parameters((self.attSize, self.hiddenSize * 2))
		self.attentionTarget = self.ASTmodel.add_parameters((self.attSize, self.numLayer*self.hiddenSize * 2))
		self.attentionParameter = self.ASTmodel.add_parameters((1, self.attSize))


		self.w_softmax = self.ASTmodel.add_parameters((self.numLayer, self.hiddenSize)) # should change whe hidden layers increase
		self.b_softmax = self.ASTmodel.add_parameters((self.vocabLengthTarget))

		# initializing the encoder and decoder
		self.forward_encoder = dy.LSTMBuilder(self.numLayer, self.embeddingSourceSize, self.hiddenSize, self.ASTmodel)
		self.backward_encoder = dy.LSTMBuilder(self.numLayer, self.embeddingSourceSize, self.hiddenSize, self.ASTmodel)

		# check this
		# embedding size + (previous action embedding + context vector + node type mebedding + parnnet feeding )
		# parent feeding - hidden states of parent action + embedding of parent action

		self.decoder = dy.VanillaLSTMBuilder(self.numLayer, self.hiddenSize * 2 +self.embeddingSourceSize*5, self.hiddenSize, self.ASTmodel)

		# adding the selection matrix
		self.w_selection_gen_softmax = self.ASTmodel.add_parameters((2, self.hiddenSize))


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
		self.ASTmodel.save(path)

	def get_learning_rate(self):
		print ("learning rate" + str(self.learning_rate))
		return self.learning_rate

	def reduce_learning_rate(self, factor):
		self.learning_rate = self.learning_rate/factor
		self.trainer.learning_rate = self.trainer.learning_rate/factor

	def parent_feed(self, parent_action_hidden_state, parent_action_embedding ):

		return [parent_action_hidden_state, parent_action_embedding]

	def decoder_state(self, previous_action, context_vector, parent_action, current_frontier_node_type  ):
		new_input = dy.concatenate([previous_action, context_vector, parent_action, current_frontier_node_type])
		new_state = previous_state.add_input(new_input)
		return new_state

	def get_att_context_vector(self, scr_output_matrix , current_state, fixed_attentional_component):

		w1_att_tgt = dy.parameter(self.attentionTarget)
		w2_att = dy.parameter(self.attentionParameter)

		target_output_embedding = dy.concatenate(list(current_state.s()))

		a_t = dy.transpose(w2_att * dy.tanh(dy.colwise_add(fixed_attentional_component, w1_att_tgt * tgt_output_embedding)))
		alignment = dy.softmax(a_t)

		context_vector = src_output_matrix * alignment

		return context_vector

	def decode_to_loss(self, vectors, output):

		goldenTree = Tree.OracleTree(output) # check this too

		sel_gen = dy.parameter(self.w_selection_gen_softmax)

		w = dy.parameter(self.w_softmax) # the weight matrix
		b = dy.parameter(self.b_softmax)

		w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

		attentional_component = w1 * encoded_states

		encoded_states = dy.concatenate_cols(encoded) # used for context vecor

		prev_action_embedding = self.actionRuleLookup(dy.vecInput(self.embeddingApplySize))

		# parent_action - 2* 256
		# context vector - 2*256
		# node type - 64  - need to change this

		decoder_states = [] # used in LSTM models for parent feed

		decoder_actions = []

		current_state = self.decoder.initial_state().add_input(dy.concatenate([dy.vecInput(self.hiddenSize*3 + self.embeddingNodeSize + self.embeddingApplySize), prev_action_embedding]))

		decoder_states.append(current_state)

		resultant_parse_tree = ""

		losses = []

		decoder_actions.append(prev_action_embedding) # storing the hidden states for further prediction

		while(tree.has_ended()):

			context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)

			parent_time =  tree.get_parent_time()

			frontier_node_type_embedding = self.nodeTypeLookup(tree.get_frontier_node_type())

			parent_action = self.parent_feed(decoder_states[parent_time], decoder_action[parent_time])

			current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)

			decoder_states.append(current_state)

			action_type = tree.get_action_type()  # assuming the tree module manages to get the node type

			if action_type == "apply":

				current_apply_action_embedding = self.get_apply_action_embedding(current_state)  # affine tf

				golden_next_rule = goldentree.get_oracle_rule()

				item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, golden_next_rule)

				prev_action_embedding = self.actionRuleLookup(golden_next_rule)

				decoder_action.append(prev_action_embedding)

				losses.append(item_loss)

				if action_type == "gen":

					selection_prob = (dy.log_softmax(sel_gen * current_state)).value()

					selected_action = np.argmax(selection_prob)

					if selected_action == 0:

						current_gen_action_embedding = self.get_gen_embedding(current_state, w, b)  # affine tf over gen vocab

						goldentoken = goldenTree.get_oracle_token()

						item_loss = dy.pickneglogsoftmax(current_gen_action_embedding, goldentoken)

						losses.append(item_loss)

						prev_action_embedding = self.gentokenLookup(goldentoken)

						decoder_action.append(prev_action_embedding)

					elif selected_action == 1:

						copy_probs = self.get_gen_copy_embedding(current_state, context_vector, encoded_states)

						pred_token = np.argmax(copy_probs)

						tree.set_token("copy",pred_token)

			tree.move()

		return losses

	def get_apply_action_embedding(self, current_state):

		s = dy.affine_transform([b, w, current_state.output()])
		g = dy.tanh(s)
		s = self.actionRuleLookup * g
		return s

	def get_gen_embedding(current_state, w, b, context_vector):

		current_state = dy.concatenate([current_state, context_vector])
		s = dy.affine_transform([b, w, current_state.output()])
		g = dy.tanh(s)
		s = self.gentokenLookup * g
		return s

	def decoder_to_predict(self, encoded, max_length):

		tree = Tree.BuildingTree()

		w = dy.parameter(self.w_softmax) # the weight matrix
		b = dy.parameter(self.b_softmax)

		sel_gen = dy.parameter(self.w_selection_gen_softmax)

		w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

		attentional_component = w1 * encoded_states

		encoded_states = dy.concatenate_cols(encoded) # used for context vecor

		prev_action_embedding = self.actionRuleLookup(dy.vecInput(self.embeddingApplySize)) # starts with the root

		# parent_action - 2* 256
		# context vector - 2*256
		# node type - 64  - need to change this

		decoder_states = [] # for parent feeding
		decoder_actions = [] # for parent feeding

		current_state = self.decoder.initial_state().add_input(dy.concatenate([dy.vecInput(self.hiddenSize*3 + self.embeddingNodeSize + self.embeddingApplySize), prev_action_embedding]))

		decoder_states.append(current_state)

		resultant_parse_tree = "" # is this needed or the tree model takes care of it?

		decoder_actions.append(prev_action_embedding) # storing the hidden states for further prediction

		while(tree.has_ended()):

			context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)

			parent_time =  tree.get_parent_time()

			frontier_node_type_embedding = self.nodeTypeLookup(tree.get_frontier_node_type())

			parent_action = self.parent_feed(decoder_states[parent_time], decoder_actions[parent_time])

			current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)

			decoder_states.append(current_state)

			action_type = tree.get_action_type()  # assuming the tree module manages to get the node type

			if action_type == "apply":

				current_apply_action_embedding = self.get_apply_action_embedding(current_state, w, b) # output of the lstm

				rule_probs = (dy.log_softmax(current_apply_action_embedding)).value() # check if transpose needed

				next_rule = tree.pick_and_set_rule((rule_probs))

				prev_action_embedding = self.actionRuleLookup(next_rule)

				decoder_action.append(prev_action_embedding)

			if action_type == "gen":

				pred_token = ''
				# for generating from the vocabulary
				selection_prob = (dy.log_softmax(sel_gen * current_state)).value()

				selected_action = np.argmax(selection_prob)

				if selected_action == 0:

					current_gen_action_embedding = self.get_gen_embedding(current_state, w, b)  # affine tf over gen vocab

					rule_probs = (dy.log_softmax(current_gen_apply_action_embedding)).value() # check if transpose needed

					pred_token = np.argmax(selected_probs)

					tree.set_token("vocab",pred_token)

				elif selected_action == 1:
					copy_probs = self.get_gen_copy_embedding(current_state, context_vector, encoded_states)

					pred_token = np.argmax(copy_probs)

					tree.set_token("copy",pred_token)
					current_gen_action_embedding = self.get_gen_copy_embedding(current_state, context_vector, encoded_states)
					# to do
					copy_tk = np.argmax()

				prev_action_embedding = self.gentokenLookup(pred_token)

				decoder_action.append(prev_action_embedding)

				tree.move()

			# check if this is the way to return
