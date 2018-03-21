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

args = namedtuple('args', [
	'vocabLengthSource',
	'vocabLengthTarget',
	'targetIndexer',
	'targetDictionnary',
	'numLayer',
	'embeddingApplySize',
	'embeddingGenSize',
	'hiddenSize',
	'attSize',
	'dropout',
	'learningRate',
	])(
	#not written,
	#vocab_target,
	#targetIndexer,
	#targetDict,
	1,
	128,
	128,
	64,
	32,
	0,
	0.001,
	)

class ASTNNModule:
	def __init__(self,args):

		self.targetIndexer = args.targetIndexer
		self.targetDictionnary = args.targetDictionnary
		self.vocabLengthSource = args.vocabLengthSource
		self.vocabLengthTarget = args.vocabLengthSource

		self.unkTarget = self.targetIndexer["<unk>"]
		self.eosTarget = self.targetIndexer["</s>"]

		# parameters for the model
		self.numLayer = args.numLayer
		self.embeddingApplySize = args.embeddingApplySize
		self.embeddingGenSize = args.embeddingGenSize
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
        self.targetLookup = self.ASTmodel.add_lookup_parameters((self.vocabLengthTarget, self.embeddingGenSize))


        self.attention_source = self.model.add_parameters(
            (self.att_size, self.hidden_size * 2))
        self.attention_target = self.model.add_parameters(
            (self.att_size, self.num_layer*self.hidden_size * 2))
        self.attention_parameter = self.model.add_parameters(
            (1, self.att_size))

        
        self.w_softmax = self.model.add_parameters(
            (self.vocab_length_target, self.hidden_size))
        self.b_softmax = self.model.add_parameters((self.vocab_length_target))

        self.forward_encoder = dy.LSTMBuilder(
            self.num_layer, self.embedding_size, self.hidden_size, self.model)
        self.backward_encoder = dy.LSTMBuilder(
            self.num_layer, self.embedding_size, self.hidden_size, self.model)
        self.decoder = dy.LSTMBuilder(self.num_layer, self.hidden_size * 2 +
                                      self.embedding_size, self.hidden_size, self.model)
