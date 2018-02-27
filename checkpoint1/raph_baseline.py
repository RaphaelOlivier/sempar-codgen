import dynet as dy
import random
from collections import defaultdict
import random
import math
import sys
import argparse
import numpy as np
import pdb
import time

mode = 'heartstone'
# mode='django'
train_src_file, train_trg_file, dev_src_file, dev_trg_file, test_src_file, test_trg_file = None, None, None, None, None, None

if(mode == 'heartstone'):
    train_src_file = "hs_dataset/hs.train.desc"
    train_trg_file = "hs_dataset/hs.train.code"
    dev_src_file = "hs_dataset/hs.dev.desc"
    dev_trg_file = "hs_dataset/hs.dev.code"
    test_src_file = "hs_dataset/hs.test.desc"
    test_trg_file = "hs_dataset/hs.test.code"
if(mode == 'django'):
    train_src_file = "django_dataset/django.train.desc"
    train_trg_file = "django_dataset/django.train.code"
    dev_src_file = "django_dataset/django.dev.desc"
    dev_trg_file = "django_dataset/django.dev.code"
    test_src_file = "django_dataset/django.test.code"
    test_trg_file = "django_dataset/django.test.code"


w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            # need to append EOS tags to at least the target sentence
            sent_src = [w2i_src[x] for x in line_src.strip().split() + ['</s>']]
            sent_trg = [w2i_trg[x] for x in ['<s>'] + line_trg.strip().split() + ['</s>']]
            yield (sent_src, sent_trg)


train_data = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
eos_src = w2i_src['</s>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}


dev_data = list(read(dev_src_file, dev_trg_file))
test_data = list(read(test_src_file, test_trg_file))

# Model parameters
VOCAB_SOURCE_SIZE = len(w2i_src)
VOCAB_TARGET_SIZE = len(w2i_trg)
print("Source vocabulary size: {}".format(VOCAB_SOURCE_SIZE))
print("Target vocabulary size: {}".format(VOCAB_TARGET_SIZE))
LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 128
STATE_SIZE = 128
ATTENTION_SIZE = 32


# Model definition
model = dy.Model()


LOOKUP_SRC = model.add_lookup_parameters((VOCAB_SOURCE_SIZE, EMBEDDINGS_SIZE))
LOOKUP_TRG = model.add_lookup_parameters((VOCAB_TARGET_SIZE, EMBEDDINGS_SIZE))

enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

input_lookup = model.add_lookup_parameters((VOCAB_SOURCE_SIZE, EMBEDDINGS_SIZE))
attention_w1 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = model.add_parameters((ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters((1, ATTENTION_SIZE))
decoder_w = model.add_parameters((VOCAB_TARGET_SIZE, STATE_SIZE))
decoder_b = model.add_parameters((VOCAB_TARGET_SIZE))
output_lookup = model.add_lookup_parameters((VOCAB_TARGET_SIZE, EMBEDDINGS_SIZE))

# Trainer definition
trainer = dy.AdamTrainer(model)
#trainer = dy.AdagradTrainer(model, 0.001)


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            # need to append EOS tags to at least the target sentence
            sent_src = [w2i_src[x] for x in line_src.strip().split() + ['</s>']]
            sent_trg = [w2i_trg[x] for x in ['<s>'] + line_trg.strip().split() + ['</s>']]
            yield (sent_src, sent_trg)


def run_lstm(init_state, input_vecs):
    s = init_state
    # print(input_vecs[0].value())
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):

    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(dec_lstm, vectors, output):
    output = [eos_trg] + list(output) + [eos_trg]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[eos_trg]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for word in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[word]
        loss.append(-dy.log(dy.pick(probs, word))*dy.pick(probs, word))
    loss = dy.esum(loss)
    return loss


def generate(input_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, renew=True):
    if(renew):
        dy.renew_cg()

    input_sentence = [LOOKUP_SRC[x] for x in input_sentence]
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, input_sentence)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[eos_trg]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(input_sentence)*2):
        if count_EOS == 2:
            break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_word = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_word]
        if next_word in i2w_trg.keys():
            if(i2w_trg[next_word] == eos_trg):
                count_EOS += 1
                continue

            out += i2w_trg[next_word]+" "
        else:
            return out
    return out


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    input_sentence = [LOOKUP_SRC[x] for x in input_sentence]
    output_sentence = output_sentence
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, input_sentence)
    return decode(dec_lstm, encoded, output_sentence)


def train(train_data, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, log_writer):
    random.shuffle(train_data)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train_data):
        # if sent_id == 1:
        #    break
        input_sent, output_sent = sent[0], sent[1]
        my_loss = get_loss(input_sent, output_sent, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        train_loss += my_loss.value()
        train_words += len(sent)
        my_loss.backward()
        trainer.update()
        if (sent_id+1) % 1000 == 0:
            print("--finished %r sentences" % (sent_id+1))
            log_writer.write("--finished %r sentences\n" % (sent_id+1))
            log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                             (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                     (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev_data):
        input_sent, output_sent = sent[0], sent[1]
        my_loss = get_loss(input_sent, output_sent, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        dev_loss += my_loss.value()
        dev_words += len(sent)
        trainer.update()
    dev_loss_per_word = dev_loss/dev_words
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, dev_loss_per_word, math.exp(dev_loss/dev_words), time.time()-start))

    return dev_loss_per_word


# Training
ITERATION = 10

#1028 or 512
log_writer = open(str(ITERATION)+"_iter_"+mode+".log", 'w')

print("iteration: " + str(ITERATION))
lowest_dev_loss = float("inf")
decreasing_counter = 0
for ITER in range(ITERATION):
    # Perform training
    dev_loss_per_word = train(train_data, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, log_writer)
    if lowest_dev_loss > dev_loss_per_word:
        lowest_dev_loss = dev_loss_per_word
    else:
        print("old learning rate: " + str(trainer.learning_rate))
        trainer.learning_rate = trainer.learning_rate/2.0
        print("new learning rate: " + str(trainer.learning_rate))
        decreasing_counter += 1

    if decreasing_counter == 5:
        print("Early stopping")
        break

log_writer.close()

model.save("model/"+mode+"_"+str(ITERATION)+"_iter_AdamTrainer.model")

# this is how you generate, can replace with desired sentenced to generate
writer = open("results/test_"+mode+"_"+str(ITERATION)+"_iter.result", 'w')
sentences = []
for sent_id, sent in enumerate(test_data):
    input_sent, output_sent = sent[0], sent[1]
    generated_sent = generate(input_sent, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
    sentences.append(generated_sent)
for sent in sentences:
    writer.write(sent+"\n")
writer.close()
