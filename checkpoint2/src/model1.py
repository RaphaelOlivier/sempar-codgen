import dynet as dy
import numpy as np
from collections import namedtuple
import time
import random
import math
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
import tree as Tree


class ASTNet:
	def __init__(self, args, vocabLengthSource, vocabLengthActionRule, vocabLengthNodes, vocabLengthTarget):

		self.flag_copy = False

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

		self.w_selection_gen_softmax = self.model.add_parameters((2, self.hiddenSize))

		self.w_out_rule = self.model.add_parameters((self.embeddingApplySize, self.hiddenSize)) # should change whe hidden layers increase
		self.b_out_rule = self.model.add_parameters((self.embeddingApplySize))

		self.w_out_vocab = self.model.add_parameters((self.embeddingApplySize, self.hiddenSize + self.hiddenSize * 2)) # should change whe hidden layers increase
		self.b_out_vocab = self.model.add_parameters((self.embeddingApplySize))

		self.w_pointer_hidden = self.model.add_parameters((self.hiddenSize, 2*self.hiddenSize + 2*self.hiddenSize + self.hiddenSize))
		self.b_pointer_hidden = self.model.add_parameters((self.hiddenSize))
		self.w_pointer_out = self.model.add_parameters((1, self.hiddenSize))
		self.b_pointer_out = self.model.add_parameters((1))
		# initializing the encoder and decoder
		self.forward_encoder = dy.LSTMBuilder(self.numLayer, self.embeddingSourceSize, self.hiddenSize, self.model)
		self.backward_encoder = dy.LSTMBuilder(self.numLayer, self.embeddingSourceSize, self.hiddenSize, self.model)

		# check this
		# embedding size + (previous action embedding + context vector + node type mebedding + parnnet feeding )
		# parent feeding - hidden states of parent action + embedding of parent action
		self.inputDecoderSize = self.embeddingApplySize + self.hiddenSize * 2 + self.hiddenSize + self.embeddingApplySize + self.embeddingNodeSize
		self.decoder = dy.VanillaLSTMBuilder(self.numLayer, self.inputDecoderSize, self.hiddenSize, self.model)


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

	def forward_prop(self, input_sentence, output_tree, mode="predict"):

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
			output_sentence = self.decode_to_prediction(encoded,output_tree, 2*len(input_sentence))
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

	def load(self, path):
		self.model.populate(path)

	def get_learning_rate(self):
		print ("learning rate" + str(self.learningRate))
		return self.learningRate

	def reduce_learning_rate(self, factor):
		self.learningRate = self.learningRate/factor
		self.trainer.learningRate = self.trainer.learningRate/factor

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
		a_t = dy.transpose(w2_att * dy.tanh(dy.colwise_add(fixed_attentional_component, w1_att_tgt * target_output_embedding)))
		alignment = dy.softmax(a_t)
		context_vector = src_output_matrix * alignment

		return context_vector

	def get_gen_copy_embedding(self,current_state, context_vector, encoded_states, w1, b1, w2, b2):

			copy_vectors = []
			for encoded_state in dy.transpose(encoded_states):
				#print(len(encoded_state.value()))
				#print(len(current_state.output().value()))
				#print(len(context_vector.value()))
				copy_input = dy.concatenate([encoded_state, current_state.output(), context_vector])
				copy_hidden = dy.rectify(dy.affine_transform([b1, w1, copy_input]))
				copy_output = dy.affine_transform([b2, w2, copy_hidden])
				copy_vectors.append(copy_output)
			c_t = dy.concatenate(copy_vectors)
			copy_probs = dy.softmax(c_t)

			return copy_probs

	def get_apply_action_embedding(self, current_state, w, b):

		s = dy.affine_transform([b, w, current_state.output()])
		g = dy.tanh(s)
		s = dy.transpose(self.actionRuleLookup) * g
		return s

	def get_gen_vocab_embedding(self,current_state, context_vector, w, b):

		# state = dy.concatenate([current_state.output(), context_vector])
		state = dy.concatenate([current_state.output(),context_vector])
		s = dy.affine_transform([b, w, state])
		g = dy.tanh(s)
		s = dy.transpose(self.gentokenLookup) * g
		return s

	def decode_to_loss(self, encoded_vectors, goldenTree):
		# initializing decoder state
		# src_output = encoded_vectors[-1]

		# current_state = self.decoder.initial_state().set_s([src_output, dy.tanh(src_output)])
		#print("hey")
		sel_gen = dy.parameter(self.w_selection_gen_softmax)
		wr = dy.parameter(self.w_out_rule) # the weight matrix
		br = dy.parameter(self.b_out_rule)
		wg = dy.parameter(self.w_out_vocab) # the weight matrix
		bg = dy.parameter(self.b_out_vocab)
		w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

		wp1 = dy.parameter(self.w_pointer_hidden)
		bp1 = dy.parameter(self.b_pointer_hidden)
		wp2 = dy.parameter(self.w_pointer_out)
		bp2 = dy.parameter(self.b_pointer_out)

		encoded_states = dy.concatenate_cols(encoded_vectors) # used for context vecor
		attentional_component = w1 * encoded_states
		decoder_states = [] # used in LSTM models for parent feed
		decoder_actions = []
		losses = []

		current_state = self.decoder.initial_state().add_input(dy.vecInput(self.inputDecoderSize))
		resultant_parse_tree = ""

		#first timestep - specific due to the absence of parent and previous action
		prev_action_embedding = dy.vecInput(self.embeddingApplySize)
		context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
		# no parent time
		frontier_node_type_embedding = self.nodeTypeLookup[goldenTree.get_node_type()]
		parent_action = dy.vecInput(self.hiddenSize+self.embeddingApplySize)
		current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)
		decoder_states.append(current_state)
		# action_type = apply
		current_apply_action_embedding = self.get_apply_action_embedding(current_state, wr, br)  # affine tf
		golden_next_rule = goldenTree.get_oracle_rule()
		#print(len(current_apply_action_embedding.value()),golden_next_rule)
		item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, golden_next_rule)
		prev_action_embedding = self.actionRuleLookup[golden_next_rule]
		decoder_actions.append(prev_action_embedding)
		losses.append(item_loss)

		while(goldenTree.move()):

			context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
			parent_time =  goldenTree.get_parent_time()
			# print(parent_time)
			frontier_node_type_embedding = self.nodeTypeLookup[goldenTree.get_node_type()]
			parent_action = self.parent_feed(decoder_states[parent_time].output(), decoder_actions[parent_time])
			current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)
			decoder_states.append(current_state)

			action_type = goldenTree.get_action_type()  # assuming the tree module manages to get the node type
			#print("action type :",action_type)
			if action_type == "apply":

				current_apply_action_embedding = self.get_apply_action_embedding(current_state, wr, br)  # affine tf
				golden_next_rule = goldenTree.get_oracle_rule()
				item_loss = dy.pickneglogsoftmax(current_apply_action_embedding, golden_next_rule)
				#print(len(current_apply_action_embedding.value()),golden_next_rule)
				prev_action_embedding = self.actionRuleLookup[golden_next_rule]
				decoder_actions.append(prev_action_embedding)
				losses.append(item_loss)

			else:
				assert(action_type == "gen")
				item_prob = dy.scalarInput(0)
				selection_prob = (dy.softmax(sel_gen * current_state.output()))
				goldentoken_vocab, goldentoken_copy, in_vocab = goldenTree.get_oracle_token()

				# words generated from vocabulary

				current_gen_action_embedding = self.get_gen_vocab_embedding(current_state, context_vector, wg, bg)  # affine tf over gen vocab
				if(self.flag_copy):
					item_prob += selection_prob[0] * dy.softmax(current_gen_action_embedding)[goldentoken_vocab]
				else:
					item_prob += dy.softmax(current_gen_action_embedding)[goldentoken_vocab]
				#print(len(current_gen_action_embedding.value()),goldentoken_vocab)
				prev_action_embedding = self.gentokenLookup[goldentoken_vocab]
				decoder_actions.append(prev_action_embedding)

				# words copied from the sentence
				if(goldentoken_copy is not None and self.flag_copy):

					copy_probs = self.get_gen_copy_embedding(current_state, context_vector, encoded_states, wp1, bp1, wp2, bp2)
					# print(len(copy_probs.value()),goldentoken_copy)
					item_prob += selection_prob[1] * dy.softmax(copy_probs)[goldentoken_copy]

				losses.append(-dy.log(item_prob))

		return dy.sum_batches(dy.esum(losses))

	def decode_to_prediction(self, encoded_vectors, tree, max_length):
		# initializing decoder state

		wr = dy.parameter(self.w_out_rule) # the weight matrix
		br = dy.parameter(self.b_out_rule)
		wg = dy.parameter(self.w_out_vocab) # the weight matrix
		bg = dy.parameter(self.b_out_vocab)
		sel_gen = dy.parameter(self.w_selection_gen_softmax)
		w1 = dy.parameter(self.attentionSource) # the weight matrix for context vector

		wp1 = dy.parameter(self.w_pointer_hidden)
		bp1 = dy.parameter(self.b_pointer_hidden)
		wp2 = dy.parameter(self.w_pointer_out)
		bp2 = dy.parameter(self.b_pointer_out)

		encoded_states = dy.concatenate_cols(encoded_vectors) # used for context vecor
		attentional_component = w1 * encoded_states
		decoder_states = [] # for parent feeding
		decoder_actions = [] # for parent feeding

		# parent_action - 2* 256
		# context vector - 2*256
		# node type - 64  - need to change this

		current_state = self.decoder.initial_state().add_input(dy.vecInput(self.inputDecoderSize))

		#first timestep - specific due to the absence of parent and previous action
		prev_action_embedding = dy.vecInput(self.embeddingApplySize)
		context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
		# no parent time
		frontier_node_type_embedding = self.nodeTypeLookup[tree.get_node_type()]
		parent_action = dy.vecInput(self.hiddenSize+self.embeddingApplySize)
		current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, frontier_node_type_embedding)
		decoder_states.append(current_state)
		# action_type = "apply"
		current_apply_action_embedding = self.get_apply_action_embedding(current_state, wr, br) # output of the lstm
		rule_probs = dy.log_softmax(current_apply_action_embedding).value() # check if transpose needed
		next_rule = tree.pick_and_get_rule((rule_probs))
		prev_action_embedding = self.actionRuleLookup[next_rule]
		decoder_actions.append(prev_action_embedding)

		while(tree.move()):

			context_vector = self.get_att_context_vector(encoded_states, current_state, attentional_component)
			parent_time =  tree.get_parent_time()
			# pprintrint(parent_time)
			prev_action_embedding = dy.vecInput(self.embeddingApplySize)
			node_type_embedding = self.nodeTypeLookup[tree.get_node_type()]


			parent_action = self.parent_feed(decoder_states[parent_time].output(), decoder_actions[parent_time])
			current_state = self.decoder_state(current_state, prev_action_embedding, context_vector, parent_action, node_type_embedding)
			decoder_states.append(current_state)
			action_type = tree.get_action_type()  # assuming the tree module manages to get the node type

			if action_type == "apply":

				current_apply_action_embedding = self.get_apply_action_embedding(current_state, wr, br) # output of the lstm
				rule_probs = dy.log_softmax(current_apply_action_embedding).value() # check if transpose needed
				next_rule = tree.pick_and_get_rule(rule_probs)
				prev_action_embedding = self.actionRuleLookup[next_rule]
				decoder_actions.append(prev_action_embedding)

			if action_type == "gen":

				pred_token = ''
				# for generating from the vocabulary
				selection_prob = dy.log_softmax(sel_gen * current_state.output()).value()


				current_gen_action_embedding = self.get_gen_vocab_embedding(current_state, context_vector, wg, bg)  # affine tf over gen vocab
				rule_probs_vocab = dy.log_softmax(current_gen_action_embedding).value() # check if transpose needed
				pred_vocab = np.argmax(rule_probs_vocab)
				if(self.flag_copy):
					pred_vocab_prob =rule_probs_vocab[pred_vocab] + selection_prob[0]
				else:
					pred_vocab_prob =rule_probs_vocab[pred_vocab]
					pred_token = pred_vocab
					tree.set_token("vocab",pred_token)

				if(self.flag_copy):
					copy_probs = self.get_gen_copy_embedding(current_state, context_vector, encoded_states, wp1, bp1, wp2, bp2).value()
					#print(len(copy_probs))
					pred_copy = np.argmax(copy_probs)
					pred_copy_prob = copy_probs[pred_copy] + selection_prob[1]

					# FOR NOW NOT AS IN THE PAPER !!
					if(pred_vocab_prob > pred_copy):
						pred_token = pred_vocab
						tree.set_token("vocab",pred_token)
					else:
						pred_token = pred_copy
						tree.set_token("copy", pred_token)

				prev_action_embedding = self.gentokenLookup[pred_token]
				decoder_actions.append(prev_action_embedding)

		return tree

	def backward_and_update(self,loss):
		loss.backward()
		self.trainer.update()

			# check if this is the way to return
