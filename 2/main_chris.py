# External API
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

# Part of External API
from operator import itemgetter
from pandas import Series, DataFrame

# Own code API
import TextminingPlsa


def visualization_likelihod(value_list):
    plt.plot(value_list, c="b", lw=2, ls="--", marker="o", ms=5, mec="g", mew=2, mfc="r")

    plt.xlabel('$Iteration$', fontsize=14)
    plt.title('Log-Likelihood for Optimization', fontsize=20)
    plt.ylabel('$Value$', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():

    corpus = TextminingPlsa.Corpus()  # instantiate corpus
    document_paths = ['./test/']


    for document_path in document_paths:
        for document_file in glob.glob(os.path.join(document_path, '*.txt')):
            #print(document_file)
            document = TextminingPlsa.PreprocessingCorpus(document_file)  # instantiate document
            document.create_corpus()
            document.unset_apostrophe()
            document.lower_text()
            predoc = document.remove_stopword()
            corpus.add_document(predoc)  # push onto corpus documents list

    print(corpus.build_vocabulary())
    voca = corpus.build_vocabulary()

    result = corpus.plsa(number_of_topics=5, max_iter=100)
    print(result[1].shape, result[2].shape)

    topic_word_list = []
    for row in range(result[1].shape[0]):
        for col in range(result[1].shape[1]):
            topic_word_list.append((voca[col], result[1][row][col]))

    #print(len(topic_word_list), topic_word_list)

    topic1 = topic_word_list[:1930]
    #topic2 = topic_word_list[1930:3860]
    #topic3 = topic_word_list[3860:5790]
    #topic4 = topic_word_list[5790:7720]
    #topic5 = topic_word_list[7720:9650]

    print(topic1)
    #print(topic2)
    #print(len(topic1), len(topic2), len(topic3), len(topic4), len(topic5))

    topic1= sorted(topic1, key=itemgetter(1))
    topic1_word = [wordFreq[0] for wordFreq in topic1]
    #topic2 =sorted(topic2, key=itemgetter(1))
    #topic2_word = [wordFreq[0] for wordFreq in topic2]
    #topic3 =sorted(topic3, key=itemgetter(1))
    #topic3_word = [wordFreq[0] for wordFreq in topic3]
    #topic4 =sorted(topic4, key=itemgetter(1))
    #topic4_word = [wordFreq[0] for wordFreq in topic4]
    #topic5 =sorted(topic5, key=itemgetter(1))
    #topic5_word = [wordFreq[0] for wordFreq in topic5]

    making_data = {
        'Topic1': [topic1_word[0], topic1_word[1], topic1_word[2], topic1_word[3], topic1_word[4]]
        #'Topic2': [topic2_word[0], topic2_word[1], topic2_word[2], topic2_word[3], topic2_word[4]],
        #'Topic3': [topic3_word[0], topic3_word[1], topic3_word[2], topic3_word[3], topic3_word[4]],
        #'Topic4': [topic4_word[0], topic4_word[1], topic4_word[2], topic4_word[3], topic4_word[4]],
        #'Topic5': [topic5_word[0], topic5_word[1], topic5_word[2], topic5_word[3], topic5_word[4]]
    }
    visualization_likelihod(result[0])
    """
    making_data ={
        'Word':[corpus.build_vocabulary()[0],corpus.build_vocabulary()[1],corpus.build_vocabulary()[2],'Log-Likelihood'],
        'Iteration1_P(w|topic1)': [result[1][0][0],result[1][0][1],result[1][0][2],result[0][0]],
        'Iteration2_P(w|topic1)': [result[1][1][0],result[1][1][0],result[1][1][0],result[0][1]],
        'Iteration3_P(w|topic1)': [result[1][2][0],result[1][2][0],result[1][2][0],result[0][2]],
        'Iteration4_P(w|topic1)': [result[1][-3][0],result[1][-3][0],result[1][-3][0],result[0][-3]],
        'Iteration5_P(w|topic1)': [result[1][-2][0],result[1][-2][0],result[1][-2][0],result[0][-2]],
        'Iteration6_P(w|topic1)': [result[1][-1][0],result[1][-1][0],result[1][-1][0],result[0][-1]],
    }
    """

    result_data = DataFrame(making_data)
    print(result_data)


if __name__ == "__main__":
    main()