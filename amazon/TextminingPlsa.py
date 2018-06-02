import re
import sys
import numpy as np

from string import punctuation
from utils import normalize
# from sklearn.preprocessing import normalize
from stopwords import stopwords
from replacers import RegexpReplacer

"""
Author: 
Chris Cha (https://github.com/ryan-chris)
Reference:
https://github.com/hitalex
"""

class PreprocessingCorpus(object):
    '''
    Pre-processing documents
    '''

    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        self.words = []

    def create_corpus(self):
        '''
        Create a corpus from document.
        '''
        symbols = punctuation.replace('\'', '')
        self.file = open(self.filepath)
        try:
            #self.lines = [line for line in self.file]
            content = self.file.read().strip()
            for symbol in symbols:
                content = content.replace(symbol, '')
            self.words = self.words + content.split()

        finally:
            self.file.close()

        return self.words

    def lower_text(self):
        """
        Converts the text of all imported files to lowercase.
        """
        self.lower_corpus_list = []

        for char in self.unset_apostrophe_list:
            lower_character = char.lower()
            self.lower_corpus_list.append(lower_character)

        return self.lower_corpus_list

    def unset_apostrophe(self):
        """
        This function needs to import RegexpReplacer.
        """
        replacer = RegexpReplacer()

        self.unset_apostrophe_list = []

        for element in self.words:
            temp_elem = replacer.replace(element)
            self.unset_apostrophe_list.append(temp_elem.replace('\'', ''))

        return self.unset_apostrophe_list

    def remove_stopword(self):
        """
        This function needs to import stopwords.
        Stopwords can be added or removed.
        """
        self.final_list = self.lower_corpus_list.copy()

        for word in self.lower_corpus_list:
            if word in stopwords:
                self.final_list.remove(word)

        return self.final_list

class Corpus(object):
    '''
    A collection of documents.
    '''

    def __init__(self):
        '''
        Initialize empty document list.
        '''
        self.documents = []
        self.listLoglikelihood=[]

    def add_document(self, document):
        '''
        Add a document to the corpus.
        '''
        self.documents.append(document)

    def build_vocabulary(self):
        '''
        Construct a list of unique words in the corpus.
        '''
        # ** ADD ** #
        # exclude words that appear in 90%+ of the documents
        # exclude words that are too (in)frequent
        discrete_set = set()
        for document in self.documents:
            for word in document:
                discrete_set.add(word)
        self.vocabulary = list(discrete_set)
        #print(self.vocabulary)

        return self.vocabulary

    def plsa(self, number_of_topics, max_iter, lambda_b=0):

        '''
        Topic Modeling
        '''
        print("EM iteration begins...")
        # Get vocabulary and number of documents.
        self.build_vocabulary()
        number_of_documents = len(self.documents)
        vocabulary_size = len(self.vocabulary)

        # build term-doc matrix
        self.term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype=np.int)
        for d_index, doc in enumerate(self.documents):
            term_count = np.zeros(vocabulary_size, dtype=np.int)
            for word in doc:
                if word in self.vocabulary:
                    w_index = self.vocabulary.index(word)
                    term_count[w_index] += 1
            self.term_doc_matrix[d_index] = term_count

        # Create the counter arrays.
        self.document_topic_prob = np.zeros([number_of_documents, number_of_topics], dtype=np.float)  # P(z | d)
        self.topic_word_prob = np.zeros([number_of_topics, len(self.vocabulary)], dtype=np.float)  # P(w | z)
        self.topic_prob = np.zeros([number_of_documents, len(self.vocabulary), number_of_topics], dtype=np.float)  # P(z | d, w)

        self.background_word_prob = np.sum(self.term_doc_matrix, axis=0)/np.sum(self.term_doc_matrix) # Background words probability (1 X W)
        self.background_prob = np.zeros([number_of_documents, len(self.vocabulary)], dtype=np.float) # initialize background probability

        # Initialize
        print("Initializing...")
        # randomly assign values
        self.document_topic_prob = np.random.random(size=(number_of_documents, number_of_topics))
        for d_index in range(len(self.documents)):
            normalize(self.document_topic_prob[d_index])  # normalize for each document
        self.topic_word_prob = np.random.random(size=(number_of_topics, len(self.vocabulary)))
        for z in range(number_of_topics):
            normalize(self.topic_word_prob[z])  # normalize for each topic


        # Run the EM algorithm
        temp = 0
        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            #print("===E step===")
            for d_index, document in enumerate(self.documents):
                for w_index in range(vocabulary_size):
                    denominator = ((lambda_b * self.background_word_prob[w_index]) + ((1 - lambda_b) * np.sum(self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index])))
                    prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]

                    if sum(prob) == 0.0:
                        print("d_index = " + str(d_index) + ",  w_index = " + str(w_index))
                        print("self.document_topic_prob[d_index, :] = " + str(self.document_topic_prob[d_index, :]))
                        print("self.topic_word_prob[:, w_index] = " + str(self.topic_word_prob[:, w_index]))
                        print("topic_prob[d_index][w_index] = " + str(prob))
                        exit(0)
                    else:
                        normalize(prob)
                    self.topic_prob[d_index][w_index] = prob
                    self.background_prob[d_index][w_index] = (lambda_b * self.background_word_prob[w_index]) / denominator

            #print("===M step===")
            # update P(w | z); word-distribution
            for z in range(number_of_topics):
                for w_index in range(vocabulary_size):
                    s = 0
                    for d_index in range(len(self.documents)):
                        count = self.term_doc_matrix[d_index][w_index]
                        s = s + count * (1 - self.background_prob[d_index][w_index]) * self.topic_prob[d_index, w_index, z]
                    self.topic_word_prob[z][w_index] = s
                normalize(self.topic_word_prob[z])

            # update P(z | d); lamda(coverage)
            for d_index in range(len(self.documents)):
                for z in range(number_of_topics):
                    s = 0
                    for w_index in range(vocabulary_size):
                        count = self.term_doc_matrix[d_index][w_index]
                        s = s + count * (1 - self.background_prob[d_index][w_index]) * self.topic_prob[d_index, w_index, z]
                    self.document_topic_prob[d_index][z] = s
                normalize(self.document_topic_prob[d_index])

            self.para = lambda_b
            if abs(self.loglikelihood() - temp) < 0.0001:
                break
            else:
                temp = self.loglikelihood()
                self.listLoglikelihood.append(temp)

        return self.para, self.listLoglikelihood, self.topic_word_prob, self.document_topic_prob
        #(self.listLoglikelihood, word_prob_dist, topic_prob_dist)


    def loglikelihood(self):

        log_value = np.log((self.para * self.background_word_prob) + (1-self.para) * np.dot(self.document_topic_prob, self.topic_word_prob))
        cGivenlamda = self.term_doc_matrix * log_value
        sumLoglikelihood = 0
        for iter in range(len(self.vocabulary)):
            sumLoglikelihood += sum(cGivenlamda)[iter,]

        return sumLoglikelihood
