import dynet as dy
import time
import random
import math
import sys

from argparse import ArgumentParser
from collections import Counter, defaultdict
import numpy as np


parser = ArgumentParser(description='Baseline Code Generator')
parser.add_argument('--data', type=str, default='hs',
                    help='Dataset to be used')
parser.add_argument('--iter', type=int, default=20,
                    help='Training iteration')

args, _ = parser.parse_known_args()

mode = args.data
ITERATION = args.iter
start = time.time()

# description
w2i_source = defaultdict(lambda: len(w2i_source))

# code
w2i_target = defaultdict(lambda: len(w2i_target))

# read data set
# input file format is "word1 word2 ..."


def read_dataset(source_file, target_file):
    with open(source_file, "r", encoding='utf-8', errors='ignore') as s_file, open(target_file, "r", encoding='utf-8', errors='ignore') as t_file:
        for source_line, target_line in zip(s_file, t_file):
            sent_src = [w2i_source[x] for x in source_line.strip().split(" ")] + [eos_source]
            sent_trg = [w2i_target[x] for x in target_line.strip().split(" ")] + [eos_target]
            yield (sent_src, sent_trg)


# choose dataset
if mode == "django":
    modulo = 1000
    print("Using Django dataset...")
    train_source_file = "../data/django_dataset/django.train.desc"
    train_target_file = "../data/django_dataset/django.train.code"
    dev_source_file = "../data/django_dataset/django.dev.desc"
    dev_target_file = "../data/django_dataset/django.dev.code"
    test_source_file = "../data/django_dataset/django.test.desc"
    test_target_file = "../data/django_dataset/django.test.code"
else:
    modulo = 100
    print("Using HS dataset...")
    train_source_file = "../data/hs_dataset/hs.train.desc"
    train_target_file = "../data/hs_dataset/hs.train.code"
    dev_source_file = "../data/hs_dataset/hs.dev.desc"
    dev_target_file = "../data/hs_dataset/hs.dev.code"
    test_source_file = "../data/hs_dataset/hs.test.desc"
    test_target_file = "../data/hs_dataset/hs.test.code"

# Special words
unk_source = w2i_source["<unk>"]
eos_source = w2i_source['</s>']
unk_target = w2i_target["<unk>"]
eos_target = w2i_target['</s>']
#sos_target = w2i_target['<s>']

# Reading data
read_dataset(train_source_file, train_target_file)
train_data = list(read_dataset(train_source_file, train_target_file))


w2i_source = defaultdict(lambda: unk_source, w2i_source)

w2i_target = defaultdict(lambda: unk_target, w2i_target)
i2w_target = {v: k for k, v in w2i_target.items()}

source_vocab = len(w2i_source)
print("Source vocabulary size: {}".format(source_vocab))
target_vocab = len(w2i_target)
print("Target vocabulary size: {}".format(target_vocab))

dev_data = list(read_dataset(dev_source_file, dev_target_file))
test_data = list(read_dataset(test_source_file, test_target_file))

# start Dynet and define trainer
model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

num_layer = 1
embedding_size = 128
hidden_size = 64
att_size = 32

#  -------- create word embedding matrix  --------
source_lookup = model.add_lookup_parameters((source_vocab, embedding_size))

# -------- attention parameters --------
attention_source = model.add_parameters((att_size, hidden_size * 2))
attention_target = model.add_parameters((att_size, num_layer*hidden_size * 2))
attention_parameter = model.add_parameters((1, att_size))

target_lookup = model.add_lookup_parameters((target_vocab, embedding_size))
w_softmax = model.add_parameters((target_vocab, hidden_size))
b_softmax = model.add_parameters((target_vocab))

# LSTM
forward_encoder = dy.LSTMBuilder(num_layer, embedding_size, hidden_size, model)
backward_encoder = dy.LSTMBuilder(num_layer, embedding_size, hidden_size, model)
decoder = dy.LSTMBuilder(num_layer, hidden_size * 2 + embedding_size, hidden_size, model)


def embed(sentence):
    return [source_lookup[x] for x in sentence]


def lstm(initial_state, input_vectors):
    state = initial_state
    output_vectors = []

    for vector in input_vectors:
        state = state.add_input(vector)
        output_vector = state.output()
        output_vectors.append(output_vector)
    return output_vectors


def encode(forward_encoder, backward_encoder, sentence):
    reversed_sentence = list(reversed(sentence))

    forward_vectors = lstm(forward_encoder.initial_state(), sentence)
    b_vectors = lstm(backward_encoder.initial_state(), reversed_sentence)
    backward_vectors = list(reversed(b_vectors))
    output_vectors = [dy.concatenate(list(x)) for x in zip(forward_vectors, backward_vectors)]
    return output_vectors


def attend(input_matrix, state, w1dt):
    # input_matrix = encoder_state x sequence length = input vectors are concatenated as columns
    global attention_target
    global attention_parameter
    w_target = dy.parameter(attention_target)
    w_att = dy.parameter(attention_parameter)

    # w1dt: (attention dim x sequence length) -> parameter for source
    # w2dt: (attention dim x attention dim) -> parameter for attention
    w2dt = w_target * dy.concatenate(list(state.s()))

    # MLP for attention score
    unnormalized_att_score = dy.transpose(w_att * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_vector = dy.softmax(unnormalized_att_score)

    # context_vector = H_encoder * attention_vector
    context_vector = input_matrix * att_vector
    return context_vector


def decode(decoder, vectors, output):
    w = dy.parameter(w_softmax)
    b = dy.parameter(b_softmax)
    w1 = dy.parameter(attention_source)
    output = list(output)

    input_matrix = dy.concatenate_cols(vectors)
    w1dt = None

    prev_output_embeddings = target_lookup[eos_target]
    current_state = decoder.initial_state().add_input(
        dy.concatenate([dy.vecInput(hidden_size * 2), prev_output_embeddings]))
    losses = []

    for next_word in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_matrix

        #(embed, context), prev_word
        vector = dy.concatenate([attend(input_matrix, current_state, w1dt), prev_output_embeddings])

        current_state = current_state.add_input(vector)
        s = dy.affine_transform([b, w, current_state.output()])
        item_loss = dy.pickneglogsoftmax(s, next_word)
        losses.append(item_loss)
        prev_output_embeddings = target_lookup[next_word]

    loss = dy.esum(losses)
    return loss


def generate(input_sentence, encoder_f, encoder_b, decoder):
    dy.renew_cg()

    input_sentence = embed(input_sentence)
    encoded = encode(encoder_f, encoder_b, input_sentence)

    w = dy.parameter(w_softmax)
    b = dy.parameter(b_softmax)
    w1 = dy.parameter(attention_source)
    input_matrix = dy.concatenate_cols(encoded)
    w1dt = None

    prev_output_embeddings = target_lookup[eos_target]
    current_state = decoder.initial_state().add_input(
        dy.concatenate([dy.vecInput(hidden_size * 2), prev_output_embeddings]))

    result = ""
    for i in range(len(input_sentence)*2):
        w1dt = w1dt or w1 * input_matrix
        vector = dy.concatenate([attend(input_matrix, current_state, w1dt), prev_output_embeddings])

        current_state = current_state.add_input(vector)
        s = dy.affine_transform([b, w, current_state.output()])
        probs = (dy.log_softmax(s)).value()
        next_word = np.argmax(probs)
        prev_output_embeddings = target_lookup[next_word]

        if(next_word == eos_target):
            return result[:-1]
        if next_word in i2w_target.keys():
            result += i2w_target[next_word]+" "
        else:
            return result[:-1]
    return result[:-1]


def get_loss(input_sentence, output_sentence, encoder_f, encoder_b, decoder):
    dy.renew_cg()

    embedded_input_sentence = embed(input_sentence)
    output_sentence = output_sentence
    encoded = encode(encoder_f, encoder_b, embedded_input_sentence)

    return decode(decoder, encoded, output_sentence)


def set_dropout():
    p = 0.
    forward_encoder.set_dropout(p)
    backward_encoder.set_dropout(p)
    decoder.set_dropout(p)


def disable_dropout():
    forward_encoder.disable_dropout()
    backward_encoder.disable_dropout()
    decoder.disable_dropout()


def train(train_data, encoder_f, encoder_b, decoder, log_writer):
    random.shuffle(train_data)
    train_words, train_loss = 0, 0.0
    start = time.time()

    set_dropout()

    for sent_id, sent in enumerate(train_data):
        input_sent, output_sent = sent[0], sent[1]
        loss = get_loss(input_sent, output_sent, encoder_f, encoder_b, decoder)
        train_loss += loss.value()
        train_words += len(sent[1][1:])
        loss.backward()
        trainer.update()
        if (sent_id+1) % modulo == 0:
            print("--finished %r sentences" % (sent_id+1))
            print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
                  (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

            log_writer.write("--finished %r sentences\n" % (sent_id+1))
            log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                             (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                     (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    # Evaluate on development set
    disable_dropout

    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev_data):
        input_sent, output_sent = sent[0], sent[1]
        loss = get_loss(input_sent, output_sent, encoder_f, encoder_b, decoder)
        dev_loss += loss.value()
        dev_words += len(sent[1][1:])
        trainer.update()
    dev_loss_per_word = dev_loss/dev_words
    dev_perplexity = math.exp(dev_loss/dev_words)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, dev_loss_per_word, dev_perplexity, time.time()-start))

    return dev_loss_per_word, dev_perplexity


# Training
log_writer = open("../log/"+str(ITERATION)+"_iter_"+mode+".log", 'w')

print("iteration: " + str(ITERATION))
lowest_dev_loss = float("inf")
successive_decreasing_counter = 0
lowest_dev_perplexity = float("inf")
for ITER in range(ITERATION):
    # Perform training
    dev_loss_per_word, dev_perplexity = train(train_data, forward_encoder, backward_encoder, decoder, log_writer)
    if lowest_dev_perplexity > dev_perplexity:
        lowest_dev_perplexity = dev_perplexity
        print("----------------------------------")
        print("Saving lowest dev perplexity: " + str(lowest_dev_perplexity) + " at iteration: " + str(ITER) + "...")
        model.save("../model/"+mode+"_"+str(ITERATION)+"lowest_iter_AdamTrainer.model")
        print("----------------------------------")
    if lowest_dev_loss > dev_loss_per_word:
        lowest_dev_loss = dev_loss_per_word
        successive_decreasing_counter = 0
    else:
        print("old learning rate: " + str(trainer.learning_rate))
        trainer.learning_rate = trainer.learning_rate/5.0
        print("new learning rate: " + str(trainer.learning_rate))
        successive_decreasing_counter += 1

    if successive_decreasing_counter == 2:
        print("Early stopping...")
        break

log_writer.close()

model.save("../model/"+mode+"_"+str(ITERATION)+"_iter_AdamTrainer.model")

# generate result
writer = open("../results/test_"+mode+"_"+str(ITERATION)+"_iter.result", 'w')
sentences = []
print("Generating result...")
for sent_id, sent in enumerate(test_data):
    input_sent, output_sent = sent[0], sent[1]
    generated_sent = generate(input_sent, forward_encoder, backward_encoder, decoder)
    sentences.append(generated_sent)

print("Writing result...")
for sent in sentences:
    writer.write(sent+"\n")
writer.close()
