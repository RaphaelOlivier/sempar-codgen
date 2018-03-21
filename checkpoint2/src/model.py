import dynet as dy
import numpy as np
from collections import namedtuple

class ASTNet:
    def __init__(self, vocab_length_source, vocab_length_target, targetIndexer, targetDictionnary, num_layer=1, embedding_apply_size=128,
                 embedding_gen_size=128, hidden_size=64, att_size=32, dropout=0, learning_rate=0.001):

        self.targetIndexer = targetIndexer
        self.targetDictionnary = targetDictionnary
        self.vocab_length_source = vocab_length_source
        self.vocab_length_target = vocab_length_target

        self.unk_target = self.targetIndexer["<unk>"]
        self.eos_target = self.targetIndexer['</s>']

        self.num_layer = num_layer
        self.embedding_apply_size = embedding_apply_size
        self.embedding_gen_size = embedding_gen_size
        self.embedding_ruletype_size = 2
        self.hidden_size = hidden_size
        self.att_size = att_size
        self.dropout = dropout

        self.model = dy.ParameterCollection()

        self.learning_rate = learning_rate
        self.trainer = dy.AdamTrainer(self.model, alpha=self.learning_rate)

        self.source_lookup = self.model.add_lookup_parameters(
            (self.vocab_length_source, self.embedding_size))

        self.attention_source = self.model.add_parameters(
            (self.att_size, self.hidden_size * 2))
        self.attention_target = self.model.add_parameters(
            (self.att_size, self.num_layer*self.hidden_size * 2))
        self.attention_parameter = self.model.add_parameters(
            (1, self.att_size))

        self.target_lookup = self.model.add_lookup_parameters(
            (self.vocab_length_target, self.embedding_size))
        self.w_softmax = self.model.add_parameters(
            (self.vocab_length_target, self.hidden_size))
        self.b_softmax = self.model.add_parameters((self.vocab_length_target))

        self.forward_encoder = dy.LSTMBuilder(
            self.num_layer, self.embedding_size, self.hidden_size, self.model)
        self.backward_encoder = dy.LSTMBuilder(
            self.num_layer, self.embedding_size, self.hidden_size, self.model)
        self.decoder = dy.LSTMBuilder(self.num_layer, self.hidden_size * 2 +
                                      self.embedding_size, self.hidden_size, self.model)

    def embed(self, sentence):
        return [self.source_lookup[x] for x in sentence]

    def encode(self, sentence):

        # forward LSTM
        forward_state = self.forward_encoder.initial_state()
        forward_vectors = []

        for word in sentence:
            forward_state = forward_state.add_input(word)
            output = forward_state.output()
            forward_vectors.append(output)

        # backward LSTM
        reversed_sentence = list(reversed(sentence))
        backward_state = self.backward_encoder.initial_state()
        backward_vectors = []

        for word in reversed_sentence:
            backward_state = backward_state.add_input(word)
            output = backward_state.output()
            backward_vectors.append(output)

        backward_vectors = list(reversed(backward_vectors))

        encoder_states = [dy.concatenate(list(x)) for x in zip(
            forward_vectors, backward_vectors)]
        return encoder_states

    def attention(self, src_output_matrix, state, fixed_attentional_component):

        w1_att_tgt = dy.parameter(self.attention_target)
        w2_att = dy.parameter(self.attention_parameter)

        tgt_output_embedding = dy.concatenate(list(state.s()))

        a_t = dy.transpose(
            w2_att * dy.tanh(dy.colwise_add(fixed_attentional_component, w1_att_tgt * tgt_output_embedding)))
        alignment = dy.softmax(a_t)

        context_vector = src_output_matrix * alignment
        return context_vector

    def set_dropout(self):
        self.forward_encoder.set_dropout(self.dropout)
        self.backward_encoder.set_dropout(self.dropout)
        self.decoder.set_dropout(self.dropout)

    def disable_dropout(self):
        self.forward_encoder.disable_dropout()
        self.backward_encoder.disable_dropout()
        self.decoder.disable_dropout()

    def decoder_state(self, previous_state, previous_action, context_vector, parent_action, frontier_node):
        new_input = dy.concatenate(
            [previous_action, context_vector, parent_action, frontier_node])
        new_state = previous_state.add_input(new_input)
        return new_state

    def decode_to_loss(self, vectors, output):

            # STOP

            item_loss = dy.pickneglogsoftmax(s, next_word)
            losses.append(item_loss)
            prev_output_embeddings = self.target_lookup[next_word]

            tree.move(...)

        loss = dy.esum(losses)
        return loss

    def decode_to_prediction(self, encoded, max_length):

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



        tree = Tree()

        output = list(output)

        encoded_states = dy.concatenate_cols(vectors)

        prev_action_embedding = self.target_lookup[self.eos_target]
        current_state = self.decoder.initial_state().add_input(
            dy.concatenate([dy.vecInput(self.hidden_size * 2), prev_output_embeddings]))
        losses = []
        attentional_component = w1 * encoded_states
        for next_word in output:
            context_vector = self.attention(
                encoded_states, current_state, attentional_component)

            frontier_node_type_embedding = self.embed_node_type(
                tree.get_frontier_node_type())
            parent_time = tree.get_parent_time()
            parent_action_embedding = decoder_states[parent_time] + \
                decoder_action[parent_time]

            current_state = self.decoder_state(
                current_state, prev_action_embedding, context_vector, parent_action_embedding, frontier_node_type_embedding)

            action_type = self.get_action_type(current_state)  # affine tf
            if action_type == "apply":
                action_embedding = get_apply_embedding(current_state)  # affine tf
                action = to_apply_rule(action_embedding)  # argmax
            if action_type == "gen":
                action_embedding = get_gen_embedding(current_state)  # affine tf

            probs = (dy.log_softmax(s)).value()
            next_word = np.argmax(probs)
            prev_output_embeddings = self.target_lookup[next_word]

            if(next_word == self.eos_target):
                return result[:-1]
            if next_word in self.targetDictionnary.keys():
                result += self.targetDictionnary[next_word]+" "
            else:
                result += self.targetDictionnary[unk_target]+" "
        return result[:-1]

    def forward(self, input_sentence, output_sentence=None, mode="predict"):
        dy.renew_cg()

        embedded_input_sentence = self.embed(input_sentence)
        encoded = self.encode(embedded_input_sentence)

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
            pred = self.decode_to_prediction(encoded, 2*len(input_sentence))
            return pred

    def backward_and_update(self, loss):
        loss.backward()
        self.trainer.update()

    def save(self, path):
        self.model.save(path)

    def get_learning_rate(self):
        return self.learning_rate

    def reduce_learning_rate(self, factor):
        self.learning_rate = self.learning_rate/factor
        self.trainer.learning_rate = self.trainer.learning_rate/factor
