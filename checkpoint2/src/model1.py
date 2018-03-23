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
				'hiddenSize','attSize','dropout','learningRate'])(1,128,128,128,64,256,32,0,0.001)

class ASTNet:
	def __init__(self, args, vocabLengthSource, vocabLengthActionRule, vocabLengthNodes, vocabLengthTarget):

		self.vocabLengthSource = vocabLengthSource
		self.vocabLengthActionRule = vocabLengthActionRule
		self.vocabLengthNodes = vocabLengthNodes
		self.vocabLengthTarget = vocabLengthTarget

		# parameters for the model
		self.numLayer = args.numLayer
		self.embeddingSourceSize = args.embeddingSourceSize
		self.embeddingApplySize = args.embeddingApplySize
		self.embeddingGenSize = args.embeddingGenSize
		self.embeddingNodeSize = args.embeddingNodeSize
		self.hiddenSize = args.hiddenSize
		self.attSize = args.attSize
		self.dropout = args.dropout
		self.embeddingRuletypeSize = 2
		self.learningRate= args.learningRate


		self.model = dy.ParameterCollection()
		self.trainer = dy.AdamTrainer(self.model, alpha=self.learningRate)

		# source lookup
		self.sourceLookup = self.model.add_lookup_parameters((self.vocabLengthSource, self.embeddingSourceSize))

		# action embeddging matrix
		self.actionRuleLookup = self.model.add_lookup_parameters((self.vocabLengthActionRule, self.embeddingApplySize))

		# for node type lookup
		self.nodeTypeLookup = self.model.add_lookup_parameters((self.vocabLengthNodes, self.embeddingNodeSize))

		# gor gen type lookup
		self.gentokenLookup = self.model.add_lookup_parameters((self.vocabLengthTarget, self.embeddingGenSize))


		# adding paramteters to the AST Neural Network
		self.attentionSource = self.model.add_parameters((self.attSize, self.hiddenSize * 2))
		self.attentionTarget = self.model.add_parameters((self.attSize, self.numLayer*self.hiddenSize * 2))
		self.attentionParameter = self.model.add_parameters((1, self.attSize))

		self.w_outrnn = self.model.add_parameters((self.embeddingApplySize, self.hiddenSize)) # should change whe hidden layers increase
		self.b_outrnn = self.model.add_parameters((self.embeddingApplySize))

		# initializing the encoder and decoder
		self.forward_encoder = dy.LSTMBuilder(self.numLayer, self.embeddingSourceSize, self.hiddenSize, self.model)
		self.backward_encoder = dy.LSTMBuilder(self.numLayer, self.embeddingSourceSize, self.hiddenSize, self.model)

		# check this
		# embedding size + (previous action embedding + context vector + node type mebedding + parnnet feeding )
		# parent feeding - hidden states of parent action + embedding of parent action
		self.inputDecoderSize = self.embeddingApplySize + self.hiddenSize * 2 + self.hiddenSize + self.embeddingApplySize + self.embeddingNodeSize
		self.decoder = dy.VanillaLSTMBuilder(self.numLayer, self.inputDecoderSize, self.hiddenSize, self.model)

		# adding the selection matrix
		self.w_selection_gen_softmax = self.model.add_parameters((2, self.hiddenSize))


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

	def forward_prop(self, input_sentence, output_tree=None, mode="predict"):

		dy.renew_cg()

		embedded_input_sentence = []
		for word in input_sentence:
			embedded_input_sentence.append(self.sourceLookup[word])

		encoded = self.encoder(embedded_input_sentence)

		if(mode == "train"):
			self.set_dropout()
			loss = self.decode_to_loss(encoded, output_tree)
			return loss

		if(mode == "validate"):
			self.disable_dropout()
			loss = self.decode_to_loss(encoded, output_tree)
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

		return dy.concatenate([parent_action_hidden_state, parent_action_embedding])

	def decoder_state(self, previous_state, previous_action, context_vector, parent_action, current_frontier_node_type  ):

		new_input = dy.concatenate([previous_action, context_vector, parent_action, current_frontier_node_type])
		new_state = previous_state.add_input(new_input)
		return new_state

	def get_att_context_vector(self, src_output_matrix , current_state, fixed_attentional_component):

		w1_att_tgt = dy.parameter(self.attentionTarget)

		w2_att = dy.parameter(self.attentionParameter)

		target_output_embedding = dy.concatenate(list(current_state.s()))
		# print(w1_att_tgt.value().shape)
		a_t = dy.transpose(w2_att * dy.tanh(dy.colwise_add(fixed_attentional_component, w1_att_tgt * target_output_embedding)))

		alignment = dy.softmax(a_t)

		context_vector = src_output_matrix * alignment

		return context_vector

	def get_gen_copy_embedding(current_state, context_vector, encoded_states):

			copy_vectors = []

			for encoded_state in encoded_states:
				copy_vectors.append(dy.concatenate(encoded_state, current_state, context_vector))

			c_t = dy.tanh(np.array(copy_vectors))

			copy_probs = dy.softmax(c_t)

			return copy_probs

	def decode_to_loss(self, encoded_vectors, goldenTree):
		# initializing decoder state
		# src_output = encoded_vectors[-1]

		# current_state = self.decoder.initial_state().set_s([src_output, dy.tanh(src_output)])

		sel_gen = dy.parameter(self.w_selection_gen_softmax)

		w = dy.parameter(self.w_outrnn) # the weight matrix
		b = dy.parameter(self.b_outrnn)

		w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

		encoded_states = dy.concatenate_cols(encoded_vectors) # used for context vecor

		attentional_component = w1 * encoded_states

		decoder_states = [] # used in LSTM models for parent feed

		decoder_actions = []

		# parent_action - 2* 256
		# context vector - 2*256
		# node type - 64  - need to change this
		current_state = self.decoder.initial_state().add_input(dy.vecInput(self.inputDecoderSize))
		resultant_parse_tree = ""

		losses = []

		#first timestep - specific due to the absence of parent and previous action
		prev_action_embedding = dy.vecInput(self.embeddingApplySize)
		context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
		# no parent time
		frontier_node_type_embedding = self.nodeTypeLookup[goldenTree.get_node_type()]
		parent_action = dy.vecInput(self.hiddenSize+self.embeddingApplySize)
		current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)
		decoder_states.append(current_state)
		# action_type = apply_rule
		current_apply_action_embedding = self.get_apply_action_embedding(current_state, w, b)  # affine tf
		golden_next_rule = goldenTree.get_oracle_rule()
		item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, golden_next_rule)
		prev_action_embedding = self.actionRuleLookup[golden_next_rule]
		decoder_actions.append(prev_action_embedding)
		losses.append(item_loss)
		goldenTree.move()

		while(not goldenTree.has_ended()):

			context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)

			parent_time =  goldenTree.get_parent_time()

			frontier_node_type_embedding = self.nodeTypeLookup[goldenTree.get_node_type()]

			parent_action = self.parent_feed(decoder_states[parent_time].output(), decoder_actions[parent_time])

			current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)

			decoder_states.append(current_state)

			action_type = goldenTree.get_action_type()  # assuming the tree module manages to get the node type

			if action_type == "apply_rule":

				current_apply_action_embedding = self.get_apply_action_embedding(current_state, w, b)  # affine tf

				golden_next_rule = goldenTree.get_oracle_rule()

				item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, golden_next_rule)

				prev_action_embedding = self.actionRuleLookup[golden_next_rule]

				decoder_actions.append(prev_action_embedding)

				losses.append(item_loss)

			elif action_type == "gen":

				item_loss = dynet.scalarInput(0)

				selection_prob = (dy.log_softmax(sel_gen * current_state))

				# words generated from vocabulary

				goldentoken = goldenTree.get_oracle_token("vocab")

				if(goldentoken is not None):

					current_gen_action_embedding = self.get_gen_embedding(current_state, w, b)  # affine tf over gen vocab

					item_loss += dy.pickneglogsoftmax(selection_prob(0) * current_gen_action_embedding, goldentoken)

					prev_action_embedding = self.gentokenLookup[goldentoken]

					decoder_actions.append(prev_action_embedding)

				# words copied from the sentence

				goldentoken = goldenTree.get_oracle_token("copy")

				if(goldentoken is not None):

					copy_probs = self.get_gen_copy_embedding(current_state, context_vector, encoded_states)

					item_loss += dy.pickneglogsoftmax(selection_prob(1) * copy_probs, goldentoken)

				losses.append(item_loss)

			goldenTree.move()

		return losses

	def get_apply_action_embedding(self, current_state, w, b):

		s = dy.affine_transform([b, w, current_state.output()])
		g = dy.tanh(s)
		s = dy.transpose(self.actionRuleLookup) * g
		return s

	def get_gen_vocab_embedding(current_state, context_vector, w, b):

		current_state = dy.concatenate([current_state, context_vector])
		s = dy.affine_transform([b, w, current_state.output()])
		g = dy.tanh(s)
		s = dy.transpose(self.gentokenLookup) * g
		return s

	def decode_to_prediction(self, encoded_vectors, max_length):
		# initializing decoder state
		# src_output = encoded_vectors[-1]
		# current_state = self.decoder.initial_state().set_s([src_output, dy.tanh(src_output)])
		self.decoder.initial_state()

		tree = Tree.BuildingTree()

		w = dy.parameter(self.w_softmax) # the weight matrix
		b = dy.parameter(self.b_softmax)

		sel_gen = dy.parameter(self.w_selection_gen_softmax)

		w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

		attentional_component = w1 * encoded_states

		encoded_states = dy.concatenate_cols(encoded) # used for context vecor

		decoder_states = [] # for parent feeding

		decoder_actions = [] # for parent feeding

		# parent_action - 2* 256
		# context vector - 2*256
		# node type - 64  - need to change this

		current_state = self.decoder.initial_state().add_input(dy.vecInput(self.hiddenSize * 2 +self.embeddingSourceSize*5))

		resultant_parse_tree = "" # is this needed or the tree model takes care of it?

		while(tree.has_ended()):

			context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)

			parent_time =  tree.get_parent_time()

			prev_action_embedding = dy.vecInput(self.embeddingApplySize)

			node_type_embedding = self.nodeTypeLookup[tree.get_node_type()]

			parent_action = self.parent_feed(decoder_states[parent_time], decoder_actions[parent_time])

			current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, node_type_embedding)

			decoder_states.append(current_state)

			action_type = tree.get_action_type()  # assuming the tree module manages to get the node type

			if action_type == "apply_rule":

				current_apply_action_embedding = self.get_apply_action_embedding(current_state, w, b) # output of the lstm

				rule_probs = (dy.log_softmax(current_apply_action_embedding)).value() # check if transpose needed

				next_rule = tree.pick_and_set_rule((rule_probs))

				prev_action_embedding = self.actionRuleLookup[next_rule]

				decoder_actions.append(prev_action_embedding)

			if action_type == "gen":

				pred_token = ''
				# for generating from the vocabulary
				selection_prob = (dy.log_softmax(sel_gen * current_state)).value()

				selected_action = np.argmax(selection_prob)

				if selected_action == 0:

					current_gen_action_embedding = self.get_gen_vocab_embedding(current_state, w, b)  # affine tf over gen vocab

					rule_probs = (dy.log_softmax(current_gen_apply_action_embedding)).value() # check if transpose needed

					pred_token = np.argmax(selected_probs)

					tree.set_token("vocab",pred_token)

				elif selected_action == 1:

					copy_probs = self.get_gen_copy_embedding(current_state, context_vector, encoded_states)

					pred_token = np.argmax(copy_probs)

					tree.set_token("copy", pred_token)

				prev_action_embedding = self.gentokenLookup[pred_token]

				decoder_actions.append(prev_action_embedding)

				tree.move()

			# check if this is the way to return
