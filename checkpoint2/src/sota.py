import dynet as dy
import time
import random
import math
import sys
import numpy as np
import datetime

from model1 import ASTNet
import source_dataset
import target_dataset

from argparse import ArgumentParser
from collections import Counter, defaultdict, namedtuple
import tree

parser = ArgumentParser(description='Checkpoint2 Code Generator')
parser.add_argument('--data', type=str, default='hs',
                    help='Dataset to be used')
parser.add_argument('--iter', type=int, default=20,
                    help='Training iteration')

args, _ = parser.parse_known_args()

# flag_copy = False
mode = args.data
ITERATION = args.iter
start = time.time()


sourceDataset = source_dataset.SourceDataset(mode)
targetDataset = target_dataset.TargetDataset(mode)
sourceIndexer = sourceDataset.indexer
targetIndexer = targetDataset.indexer

vocab_length_source = sourceDataset.vocab_length
vocab_length_target = targetIndexer.vocab_length
vocab_length_nodes = targetIndexer.node_length
vocab_length_rules = targetIndexer.rule_length
today = datetime.datetime.now().strftime("%Y-%m-%d")
# Reading data

# start Dynet and define trainer
modulo = 10
if mode == "django":
    modulo = 100

args_model = namedtuple('args', ['numLayer', 'embeddingSourceSize', 'embeddingApplySize', 'embeddingGenSize',
                        'embeddingNodeSize','hiddenSize', 'attSize', 'pointerSize', 'dropout',
                         'learningRate'])(1, 128, 128, 128, 64, 256, 50, 50, 0.3, 0.001)

net = ASTNet(args=args_model, vocabLengthSource=vocab_length_source,
             vocabLengthActionRule=vocab_length_rules, vocabLengthNodes=vocab_length_nodes,
             vocabLengthTarget=vocab_length_target)
# net.load("../../data/exp/models/hs_15_iter_AdamTrainer.model")


def train(log_writer):
    target_train_dataset = targetDataset.target_train_dataset
    train_words, train_loss = 0, 0.0
    a = np.arange(len(target_train_dataset))
    np.random.shuffle(a)
    batch_size = 8
    for k in range(0, len(a)-batch_size, batch_size):
        dy.renew_cg()
        losses = []
        for j in range(k, k+batch_size):
            i = a[j]
            input_s, real_s, goldenTree = sourceDataset.train_index[i], sourceDataset.train_str[i], target_train_dataset[i].copy(
                verbose=False)
            # print(i)
            #print(" ".join(real_s))
            goldenTree.set_query(real_s)

            train_words += goldenTree.length
            # print(input_s,output_s.current_node)
            # print(type(output_s))
            loss = net.forward_prop(input_s, goldenTree, mode="train")
            losses.append(loss)
            # print("here")
        # print("there")
        batch_loss = dy.esum(losses)
        loss_val = batch_loss.value()
        # print("batch loss :", loss_val)
        train_loss += loss_val
        # print("bonjour")
        net.backward_prop_and_update_parameters(batch_loss)
        if k % (modulo*batch_size) == 0:
            print("--finished %r sentences" % k)
            print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
                  (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

            log_writer.write("--finished %r sentences\n" % k)
            log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                             (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
    log_writer.write("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs\n" %
                     (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))

    # Evaluate on development set

    target_dev_dataset = targetDataset.target_dev_dataset
    dev_words, dev_loss = 0, 0.0
    dev_loss = 0
    for i in range(0, len(target_dev_dataset)):
        dy.renew_cg()
        input_s, real_s, goldenTree = sourceDataset.dev_index[i], sourceDataset.dev_str[i], target_dev_dataset[i].copy(
            verbose=False)

        goldenTree.set_query(real_s)

        dev_words += goldenTree.length

        loss = net.forward_prop(input_s, goldenTree, mode="validate")
        dev_loss += loss.value()

    dev_loss_per_word = dev_loss/dev_words
    dev_perplexity = math.exp(dev_loss/dev_words)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" %
          (ITER, dev_loss_per_word, dev_perplexity, time.time()-start))

    return dev_loss_per_word, dev_perplexity


# Training
log_writer = open("../../data/exp/log/" +str(ITERATION)+"_iter_"+mode+ "_"+str(today)+".log", 'w')
print("iteration: " + str(ITERATION))
lowest_dev_loss = float("inf")
successive_decreasing_counter = 0
lowest_dev_perplexity = float("inf")
for ITER in range(ITERATION):
    # Perform training
    dy.renew_cg()
    dev_loss_per_word, dev_perplexity = train(log_writer)
    if lowest_dev_perplexity > dev_perplexity:
        lowest_dev_perplexity = dev_perplexity
        print("----------------------------------")
        print("Saving lowest dev perplexity: " +
              str(lowest_dev_perplexity) + " at iteration: " + str(ITER) + "...")
        net.save("../../data/exp/models/"+mode+"_"+str(ITER) +"_"+str(today) +
                 "_lowest_iter_AdamTrainer.model")
        print("----------------------------------")
    if lowest_dev_loss > dev_loss_per_word:
        lowest_dev_loss = dev_loss_per_word
        successive_decreasing_counter = 0
    else:
        print("old learning rate: " + str(net.get_learning_rate()))
        net.reduce_learning_rate(5)
        print("new learning rate: " + str(net.get_learning_rate()))
        successive_decreasing_counter += 1

    #if successive_decreasing_counter == 3:
    #    print("Early stopping...")
    #    break

log_writer.close()

net.save("../../data/exp/models/"+mode+"_"+str(ITERATION)+"_iter_AdamTrainer.model")
# generate result
trees = []
print("Generating result...")
target_test_dataset = targetDataset.target_test_dataset
test_words, test_loss = 0, 0.0
test_loss = 0
for i in range(0, len(target_test_dataset)):

    input_s, real_s = sourceDataset.test_index[i], sourceDataset.test_str[i]
    #print(" ".join(real_s))
    generated_tree = tree.BuildingTree(targetDataset.indexer, real_s, verbose=False)

    generated_tree = net.forward_prop(input_s, generated_tree, mode="predict")
    generated_tree.go_to_root()
    trees.append(generated_tree)

print("Writing result...")

#path = "../../data/exp/results/test_"+mode+"_" + str(ITERATION)+"_iter.json"
suffix = str(ITERATION)+"_iter"+"_"+str(today)
targetDataset.export(trees, suffix)
