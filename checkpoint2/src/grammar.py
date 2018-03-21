import nltk
import ast


class Grammar:
    def __init__(self, freqmin):
        self.freqmin = freqmin

    def preprocess_code(self, c):
        # TODO with nltk
        return c

    def code2ast(self, c):
        # TODO with ast
        return c

    def inferenceAlgorithm(self, ut):
        # TODO, see pseudocode
