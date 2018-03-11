
# coding: utf-8

# ## Environment Setting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from collections import Counter
from string import punctuation
from nltk.stem import PorterStemmer
from operator import itemgetter
from math import floor, log

from stopwords import stopwords
from replacer import RegexpReplacer

# ## 1. Pre-Processing

k_files = ['k{}.txt'.format(number) for number in range(1, 6)]
N_files = ['N{}.txt'.format(number) for number in range(1, 6)]


# Tokenization: tokenize each document into tokens. 

def import_corpus(filename):
    
    words = []

    symbols = punctuation.replace('\'','')
    
    with open(filename) as file:
        content = file.read().strip()
        for symbol in symbols:
            content = content.replace(symbol, '')
        words = words + content.split()

    return words

def create_corpus(filenames):
    # 단어를 저장할 리스트를 생성합니다.
    words = []

    # 여러 파일에 등장하는 모든 단어를 모두 words에 저장합니다.
    # 이 때 문장부호를 포함한 모든 특수기호를 제거합니다.
    symbols = punctuation.replace('\'','')
    
    for filename in filenames:
        with open(filename) as file:
            content = file.read().strip()
            for symbol in symbols:
                content = content.replace(symbol, '')
            words = words + content.split()

    return words


# Normalization: normalize the tokens from step 1. 

def lower_text(document_object):
    
    lower_corpus_list = []
    
    for char in document_object:
        lower_character = char.lower()
        lower_corpus_list.append(lower_character)
        
    return lower_corpus_list

def unset_apostrophe(lower_corpus_list):
    
    replacer = RegexpReplacer()
    
    unset_apostrophe_list = []
    
    for element in lower_corpus_list:
        temp_elem = replacer.replace(element)
        unset_apostrophe_list.append(temp_elem.replace('\'',''))
        
    return unset_apostrophe_list


# Stopword removal: remove stopwords.

def remove_stopword(duplicate_list):
    
    final_list = duplicate_list.copy()
    
    for word in duplicate_list:
        if word in stopwords:
            final_list.remove(word)
            
    return final_list


# Stemming: stem the tokens back to their root form. 

def stem_corpus(before_stem_final_list):
    
    ps = PorterStemmer()
    corpus_list = [ps.stem(stem_word) for stem_word in before_stem_final_list]
    
    corpus = Counter(corpus_list)
    
    return list(corpus.items())


# ## 2. Understand Zipf's Law 

# Zipf's curve

def zipfs_curve(stem_corpus_list):
    #pos = range(len(stem_corpus_list))
    
    words = [tup[0] for tup in stem_corpus_list]
    freqs = [tup[1] for tup in stem_corpus_list]
    
    ranks = [rank+1 for rank in range(len(words))]
    
    font = fm.FontProperties(fname='./NanumBarunGothic.ttf')
    plt.figure(figsize=(16,6))
    plt.plot(ranks, freqs)
    
    plt.xlabel('rank', fontproperties=font, fontsize=14)
    plt.title('Zipf\'s law', fontproperties=font, fontsize=20)
    plt.ylabel('frequency', fontproperties=font, fontsize=14)
    plt.tight_layout()


# Decide to cut-off

def cut_zipfs_curve(stem_corpus_list, cut=0.1):
    #pos = range(len(stem_corpus_list))
    
    words = [tup[0] for tup in stem_corpus_list]
    freqs = [tup[1] for tup in stem_corpus_list]
    ranks = [rank+1 for rank in range(len(words))]
    
    cutoff = floor(len(words)*cut)
    re_words = words[cutoff:-cutoff]
    re_freqs = freqs[cutoff:-cutoff]
    re_ranks = [re+1 for re in range(len(re_words))]
    
    font = fm.FontProperties(fname='./NanumBarunGothic.ttf')
    plt.figure(figsize=(16,6))
    plt.plot(re_ranks, re_freqs)
    
    plt.xlabel('rank', fontproperties=font, fontsize=14)
    plt.title('Zipf\'s law(cut-off)', fontproperties=font, fontsize=20)
    plt.ylabel('frequency', fontproperties=font, fontsize=14)
    plt.tight_layout()


# 추가) cut off 결정해서.. 다시 그린 그림 // 멀티플랏(cut off) // rank 1 2 3 word. freq

# ## 3.  Construct a Vocabulary 

# List the resulting vocabulary.

def list_voca(stem_corpus_list, cutoff=False, cut=0.1):
    cutNum = floor(len(stem_corpus_list)*cut)

    if cutoff == True:
        return [tup[0] for tup in stem_corpus_list][cutNum:-cutNum]
      
    else:
        return [tup[0] for tup in stem_corpus_list]


# List top 50 and bottom 50 words according to TF in the resulting vocabulary, and represent their corresponding IDFs. 

def tf(stem_corpus_list):
    """
    input data structure has a list with tuples(word, counts)
    e.g. [(word1, counts), (word2, counts), ... ]
    """
    x = [tup[1] for tup in stem_corpus_list]
    tf_list = []
    
    for elem in x:
        y = .5 + .5*float(elem) / max(x)
        tf_list.append(y)
    return tf_list    


def idf(stem_corpus_list, doc_list):
    """
    input data structure has a list with tuples(word, counts).
    and doc_list is a list of document.
    e.g. stem_corpus_list = [(word1, counts), (word2, counts), ... ]
         if doument is 1, doc_list = [stem_corpus_list] 
    """
    M = len(doc_list)
    words = [tup[0] for tup in stem_corpus_list]
    
    idf_list = []
    word_count=0
    
    for doc in doc_list:
        for word in words: 
            if word in doc:
                word_count+=1
            idf_w = log((M+1) / (word_count+1))
            idf_list.append(idf_w)
            
    return idf_list


def list_N_top_bottom(word, tf, idf, num = 50):
    df = pd.DataFrame()
    
    df['Words'] = word[:num] + word[-num:]
    df['TF'] = tf[:num] + tf[-num:]
    df['IDF'] = idf[:num] + idf[-num:]
    
    return df


# ## 4. Compute Similarity between documents 

# Paste your implementation of similarity computation

def jaccard(s1, s2):
    
    set1, set2 = set(s1), set(s2)
    
    return len(set1 & set2) / float(len(set1 | set2))


# For each document, list the top 3 most similar documents and the corresponding similarity. Please define your similarity.

def loopUI(func, bunch):
    
    result = []
    for elem in bunch:
        para = func(elem)
        result.append(para)
        
    return result

def main():
    
    #1
    korea = loopUI(import_corpus, k_files)
    newyork = loopUI(import_corpus, N_files)
    document = korea + newyork
    total = k_files + N_files
    
    lower_doc = loopUI(lower_text, document)
    
    unset_doc = loopUI(unset_apostrophe, lower_doc)
    
    stop_doc = loopUI(remove_stopword, unset_doc)
    
    stem_doc = loopUI(stem_corpus, stop_doc)
    
    #2
    for doc in stem_doc:
        zipfs_curve(sorted(doc, key=itemgetter(1), reverse=True))

    for doc in stem_doc:
        cut_zipfs_curve(sorted(doc, key=itemgetter(1), reverse=True), cut=0.1)
    
    #3
    voca_doc = loopUI(list_voca, stem_doc)
    tf_doc = loopUI(tf, stem_doc)
    
    idf_doc = []
    for stem in stem_doc:
        temp = idf(stem, total)
        idf_doc.append(temp)
    
    doc_df1 = list_N_top_bottom(voca_doc[0], tf_doc[0], idf_doc[0], num = 50)
    doc_df2 = list_N_top_bottom(voca_doc[1], tf_doc[1], idf_doc[1], num = 50)
    doc_df3 = list_N_top_bottom(voca_doc[2], tf_doc[2], idf_doc[2], num = 50)
    doc_df4 = list_N_top_bottom(voca_doc[3], tf_doc[3], idf_doc[3], num = 50)
    doc_df5 = list_N_top_bottom(voca_doc[4], tf_doc[4], idf_doc[4], num = 50)
    
    doc_df1 = list_N_top_bottom(voca_doc[0], tf_doc[0], idf_doc[0], num = 50)
    doc_df2 = list_N_top_bottom(voca_doc[1], tf_doc[1], idf_doc[1], num = 50)
    doc_df3 = list_N_top_bottom(voca_doc[2], tf_doc[2], idf_doc[2], num = 50)
    doc_df4 = list_N_top_bottom(voca_doc[3], tf_doc[3], idf_doc[3], num = 50)
    doc_df5 = list_N_top_bottom(voca_doc[4], tf_doc[4], idf_doc[4], num = 50)
    
    #4
    doc_mat = np.zeros((10,10))
    for row in range(len(voca_doc)):
        for col in range(len(voca_doc)):
            doc_mat[row][col] = jaccard(voca_doc[row], voca_doc[col])
    
    final = pd.DataFrame(doc_mat)
    final.columns = ['Doc{}'.format(number) for number in range(1,11)]
    final.index = ['Doc{}'.format(number) for number in range(1,11)]
    print(final)

    
if __name__ == '__main__':
    main()