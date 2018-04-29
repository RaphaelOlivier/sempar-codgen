# -*- coding: UTF-8 -*-

from __future__ import division
import os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import logging
import traceback

from nn.utils.generic_utils import init_logging

from model import *


#DJANGO_ANNOT_FILE = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
DJANGO_ANNOT_FILE = "anno.txt"


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens


def evaluate(model, dataset, verbose=True):
    if verbose:
        logging.info('evaluating [%s] dataset, [%d] examples' % (dataset.name, dataset.count))

    exact_match_ratio = 0.0

    for example in dataset.examples:
        logging.info('evaluating example [%d]' % example.eid)
        hyps, hyp_scores = model.decode(example, max_time_step=config.decode_max_time_step)
        gold_rules = example.rules

        if len(hyps) == 0:
            logging.warning('no decoding result for example [%d]!' % example.eid)
            continue

        best_hyp = hyps[0]
        predict_rules = [dataset.grammar.id_to_rule[rid] for rid in best_hyp]

        assert len(predict_rules) > 0 and len(gold_rules) > 0

        exact_match = sorted(gold_rules, key=lambda x: x.__repr__()) == sorted(predict_rules, key=lambda x: x.__repr__())
        if exact_match:
            exact_match_ratio += 1

        # p = len(predict_rules.intersection(gold_rules)) / len(predict_rules)
        # r = len(predict_rules.intersection(gold_rules)) / len(gold_rules)

    exact_match_ratio /= dataset.count

    logging.info('exact_match_ratio = %f' % exact_match_ratio)

    return exact_match_ratio


def evaluate_decode_results(dataset, decode_results, verbose=True):
    from lang.py.parse import tokenize_code, de_canonicalize_code
    # tokenize_code = tokenize_for_bleu_eval
    import ast
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()

        counter = 0
        if config.data_type == 'django':
            for raw_id, line in enumerate(open(DJANGO_ANNOT_FILE)):
                eid_to_annot[raw_id] = line.strip()
                counter += 1

        print("counter: " + str(counter))

        f_bleu_eval_ref = open(dataset.name + '.ref', 'w')
        f_bleu_eval_hyp = open(dataset.name + '.hyp', 'w')
        f_generated_code = open(dataset.name + '.geneated_code', 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        # refer_source = ref_code
        refer_tokens = tokenize_code(refer_source)
        cur_example_correct = False

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        decode_cand = decode_cands[0]

        cid, cand, ast_tree, code = decode_cand
        code = astor.to_source(ast_tree).strip()

        # simple_url_2_re = re.compile('_STR:0_', re.))
        try:
            predict_tokens = tokenize_code(code)
        except:
            logging.error('error in tokenizing [%s]', code)
            continue

        if refer_tokens == predict_tokens:
            cum_acc += 1
            cur_example_correct = True

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(code + '\n')
                f.write('-' * 60 + '\n')

        if config.data_type == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif config.data_type == 'hs':
            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        # The if-chunk below is for debugging purpose, sometimes the reference cannot match with the prediction
        # because of inconsistent quotes (e.g., single quotes in reference, double quotes in prediction).
        # However most of these cases are solved by cannonicalizing the reference code using astor (parse the reference
        # into AST, and regenerate the code. Use this regenerated one as the reference)
        weired = False
        if refer_tokens_for_bleu == pred_tokens_for_bleu and refer_tokens != predict_tokens:
            # cum_acc += 1
            weired = True
        elif refer_tokens == predict_tokens:
            # weired!
            # weired = True
            pass

        shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

        all_references.append([refer_tokens_for_bleu])
        all_predictions.append(pred_tokens_for_bleu)

        # try:
        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
        cum_bleu += bleu_score
        # except:
        #    pass

        if verbose:
            print 'raw_id: %d, bleu_score: %f' % (example.raw_id, bleu_score)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            if config.data_type == 'django':
                f_decode.write(eid_to_annot[example.raw_id] + '\n')
            elif config.data_type == 'hs':
                f_decode.write(' '.join(example.query) + '\n')

            f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
            f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

            f_decode.write('canonicalized reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('canonicalized prediction: \n')
            f_decode.write(code + '\n')
            f_decode.write('reference code for bleu calculation: \n')
            f_decode.write(ref_code_for_bleu + '\n')
            f_decode.write('predicted code for bleu calculation: \n')
            f_decode.write(pred_code_for_bleu + '\n')
            f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
            f_decode.write('weired: %s\n' % weired)
            f_decode.write('-' * 60 + '\n')

            # for Hiro's evaluation
            f_generated_code.write(pred_code_for_bleu.replace('\n', '#NEWLINE#') + '\n')


        # compute oracle
        best_score = 0.
        cur_oracle_acc = 0.
        for decode_cand in decode_cands[:config.beam_size]:
            cid, cand, ast_tree, code = decode_cand
            try:
                code = astor.to_source(ast_tree).strip()
                predict_tokens = tokenize_code(code)

                if predict_tokens == refer_tokens:
                    cur_oracle_acc = 1

                if config.data_type == 'django':
                    pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
                    # convert canonicalized code to raw code
                    for literal, place_holder in example.meta_data['str_map'].iteritems():
                        pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                elif config.data_type == 'hs':
                    pred_code_for_bleu = code

                # we apply Ling Wang's trick when evaluating BLEU scores
                pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

                ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
                bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu,
                                           weights=ngram_weights,
                                           smoothing_function=sm.method3)

                if bleu_score > best_score:
                    best_score = bleu_score

            except:
                continue

        cum_oracle_bleu += best_score
        cum_oracle_acc += cur_oracle_acc

    cum_bleu /= dataset.count
    cum_acc /= dataset.count
    cum_oracle_bleu /= dataset.count
    cum_oracle_acc /= dataset.count

    logging.info('corpus level bleu: %f', corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3))
    logging.info('sentence level bleu: %f', cum_bleu)
    logging.info('accuracy: %f', cum_acc)
    logging.info('oracle bleu: %f', cum_oracle_bleu)
    logging.info('oracle accuracy: %f', cum_oracle_acc)

    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

        f_bleu_eval_ref.close()
        f_bleu_eval_hyp.close()
        f_generated_code.close()

    return cum_bleu, cum_acc


def analyze_decode_results(dataset, decode_results, verbose=True):
    from lang.py.parse import tokenize_code, de_canonicalize_code
    # tokenize_code = tokenize_for_bleu_eval
    import ast
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()

        if config.data_type == 'django':
            for raw_id, line in enumerate(open(DJANGO_ANNOT_FILE)):
                eid_to_annot[raw_id] = line.strip()

        f_bleu_eval_ref = open(dataset.name + '.ref', 'w')
        f_bleu_eval_hyp = open(dataset.name + '.hyp', 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1

    binned_results_dict = defaultdict(list)
    def get_binned_key(ast_size):
        cutoff = 50 if config.data_type == 'django' else 250
        k = 10 if config.data_type == 'django' else 25 # for hs

        if ast_size >= cutoff:
            return '%d - inf' % cutoff

        lower = int(ast_size / k) * k
        upper = lower + k

        key = '%d - %d' % (lower, upper)

        return key


    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        # refer_source = ref_code
        refer_tokens = tokenize_code(refer_source)
        cur_example_acc = 0.0

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        decode_cand = decode_cands[0]

        cid, cand, ast_tree, code = decode_cand
        code = astor.to_source(ast_tree).strip()

        # simple_url_2_re = re.compile('_STR:0_', re.))
        try:
            predict_tokens = tokenize_code(code)
        except:
            logging.error('error in tokenizing [%s]', code)
            continue

        if refer_tokens == predict_tokens:
            cum_acc += 1
            cur_example_acc = 1.0

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(code + '\n')
                f.write('-' * 60 + '\n')

        if config.data_type == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif config.data_type == 'hs':
            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

        all_references.append([refer_tokens_for_bleu])
        all_predictions.append(pred_tokens_for_bleu)

        # try:
        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
        cum_bleu += bleu_score
        # except:
        #    pass

        if verbose:
            print 'raw_id: %d, bleu_score: %f' % (example.raw_id, bleu_score)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            if config.data_type == 'django':
                f_decode.write(eid_to_annot[example.raw_id] + '\n')
            elif config.data_type == 'hs':
                f_decode.write(' '.join(example.query) + '\n')

            f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
            f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

            f_decode.write('canonicalized reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('canonicalized prediction: \n')
            f_decode.write(code + '\n')
            f_decode.write('reference code for bleu calculation: \n')
            f_decode.write(ref_code_for_bleu + '\n')
            f_decode.write('predicted code for bleu calculation: \n')
            f_decode.write(pred_code_for_bleu + '\n')
            f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
            # f_decode.write('weired: %s\n' % weired)
            f_decode.write('-' * 60 + '\n')

        # compute oracle
        best_bleu_score = 0.
        cur_oracle_acc = 0.
        for decode_cand in decode_cands[:config.beam_size]:
            cid, cand, ast_tree, code = decode_cand
            try:
                code = astor.to_source(ast_tree).strip()
                predict_tokens = tokenize_code(code)

                if predict_tokens == refer_tokens:
                    cur_oracle_acc = 1.

                if config.data_type == 'django':
                    pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
                    # convert canonicalized code to raw code
                    for literal, place_holder in example.meta_data['str_map'].iteritems():
                        pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                elif config.data_type == 'hs':
                    pred_code_for_bleu = code

                # we apply Ling Wang's trick when evaluating BLEU scores
                pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

                ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
                cand_bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu,
                                                weights=ngram_weights,
                                                smoothing_function=sm.method3)

                if cand_bleu_score > best_bleu_score:
                    best_bleu_score = cand_bleu_score

            except:
                continue

        cum_oracle_bleu += best_bleu_score
        cum_oracle_acc += cur_oracle_acc

        ref_ast_size = example.parse_tree.size
        binned_key = get_binned_key(ref_ast_size)
        binned_results_dict[binned_key].append((bleu_score, cur_example_acc, best_bleu_score, cur_oracle_acc))

    cum_bleu /= dataset.count
    cum_acc /= dataset.count
    cum_oracle_bleu /= dataset.count
    cum_oracle_acc /= dataset.count

    logging.info('corpus level bleu: %f', corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3))
    logging.info('sentence level bleu: %f', cum_bleu)
    logging.info('accuracy: %f', cum_acc)
    logging.info('oracle bleu: %f', cum_oracle_bleu)
    logging.info('oracle accuracy: %f', cum_oracle_acc)

    keys = sorted(binned_results_dict, key=lambda x: int(x.split(' - ')[0]))

    Y = [[], [], [], []]
    X = []

    for binned_key in keys:
        entry = binned_results_dict[binned_key]
        avg_bleu = np.average([t[0] for t in entry])
        avg_acc = np.average([t[1] for t in entry])
        avg_oracle_bleu = np.average([t[2] for t in entry])
        avg_oracle_acc = np.average([t[3] for t in entry])
        print binned_key, avg_bleu, avg_acc, avg_oracle_bleu, avg_oracle_acc, len(entry)

        Y[0].append(avg_bleu)
        Y[1].append(avg_acc)
        Y[2].append(avg_oracle_bleu)
        Y[3].append(avg_oracle_acc)

        X.append(int(binned_key.split(' - ')[0]))

    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 6, 2.5

    if config.data_type == 'django':
        fig, ax = plt.subplots()
        ax.plot(X, Y[0], 'bs--', label='BLEU', lw=1.2)
        # ax.plot(X, Y[2], 'r^--', label='oracle BLEU', lw=1.2)
        ax.plot(X, Y[1], 'r^--', label='acc', lw=1.2)
        # ax.plot(X, Y[3], 'r^--', label='oracle acc', lw=1.2)
        ax.set_ylabel('Performance')
        ax.set_xlabel('Reference AST Size (# nodes)')
        plt.legend(loc='upper right', ncol=6)
        plt.tight_layout()
        # plt.savefig('django_acc_ast_size.pdf', dpi=300)
        # os.system('pcrop.sh django_acc_ast_size.pdf')
        plt.savefig('django_perf_ast_size.pdf', dpi=300)
        os.system('pcrop.sh django_perf_ast_size.pdf')
    else:
        fig, ax = plt.subplots()
        ax.plot(X, Y[0], 'bs--', label='BLEU', lw=1.2)
        # ax.plot(X, Y[2], 'r^--', label='oracle BLEU', lw=1.2)
        ax.plot(X, Y[1], 'r^--', label='acc', lw=1.2)
        # ax.plot(X, Y[3], 'r^--', label='oracle acc', lw=1.2)
        ax.set_ylabel('Performance')
        ax.set_xlabel('Reference AST Size (# nodes)')
        plt.legend(loc='upper right', ncol=6)
        plt.tight_layout()
        # plt.savefig('hs_bleu_ast_size.pdf', dpi=300)
        # os.system('pcrop.sh hs_bleu_ast_size.pdf')
        plt.savefig('hs_perf_ast_size.pdf', dpi=300)
        os.system('pcrop.sh hs_perf_ast_size.pdf')
    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

        f_bleu_eval_ref.close()
        f_bleu_eval_hyp.close()

    return cum_bleu, cum_acc

if __name__ == '__main__':
    eid_to_annot = dict()

    counter = 0
    for raw_id, line in enumerate(open(DJANGO_ANNOT_FILE)):
        eid_to_annot[raw_id] = line.strip()
        counter += 1

    print(len(eid_to_annot))
    print("counter: " + str(counter))
    dataset = "test_data"
    model = "model.best_bleu.npz"
    print("MAIN")
    #decode_results_file = "decode_results.bin"
    #decode_results = deserialize_from_file(decode_results_file)
    #print(dataset.count)