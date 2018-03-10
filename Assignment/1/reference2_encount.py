# 프로젝트에 필요한 패키지를 import합니다.
from operator import itemgetter
from collections import Counter
from string import punctuation
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from stopwords import stopwords

from elice_utils import EliceUtils

elice_utils = EliceUtils()


def import_corpus(filename):
    '''
    단어와 빈도수 데이터가 담긴 파일을 읽어 리스트 형식으로 변환합니다.
    리스트의 각 원소는 (단어, 빈도수) 꼴의 튜플입니다.
    '''
    corpus = []

    # 함수를 완성합니다.
    with open(filename) as file:
        for line in file:
            word, freq = line.strip().split(',')
            tuples = (word, int(freq))
            corpus.append(tuples)

    return corpus


# print(import_corpus('corpus.txt'))

def create_corpus(filenames):
    '''
    텍스트가 담긴 파일 여러 개를 읽어 분석할 수 있는 corpus 형태(튜플의 리스트)로 만듭니다.
    '''
    # 단어를 저장할 리스트를 생성합니다.
    words = []

    # 여러 파일에 등장하는 모든 단어를 모두 words에 저장합니다.
    # 이 때 문장부호를 포함한 모든 특수기호를 제거합니다.
    for filename in filenames:
        with open(filename) as file:
            content = file.read().strip()
            for symbol in punctuation:
                content = content.replace(symbol, '')
            words = words + content.split()

    # words 리스트의 데이터를 corpus 형태로 변환합니다.
    corpus = Counter(words)

    # optional code line
    for stopword in stopwords:
        corpus[stopword] = 0
    return list(corpus.items())


# print(create_corpus('corpus.txt'))

def filter_by_prefix(corpus, prefix):
    '''
    corpus의 데이터 중 prefix로 시작하는 단어 데이터만 추립니다.
    '''

    '''
    #1
    filtered = []
    for word, freq in corpus:
        if word.startswith(prefix):
            filtered.append((word, freq))
    return filtered

    #2
    filtered = []
    for word_data in corpus:
        if word_data[0].startswith(prefix):
            filtered.append(word_data)
    return filtered
    '''

    # 3
    return [word_data for word_data in corpus if word_data[0].startswith(prefix)]


def most_frequent_words(corpus, number):
    '''
    corpus의 데이터 중 가장 빈도가 높은 number개의 데이터만 추립니다.
    '''
    return sorted(corpus, key=itemgetter(1), reverse=True)[:number]


def draw_frequency_graph(corpus):
    # 막대 그래프의 막대 위치를 결정하는 pos를 선언합니다.
    pos = range(len(corpus))

    # 튜플의 리스트인 corpus를 단어의 리스트 words와 빈도의 리스트 freqs로 분리합니다.
    words = [tup[0] for tup in corpus]
    freqs = [tup[1] for tup in corpus]

    # 한국어를 보기 좋게 표시할 수 있도록 폰트를 설정합니다.
    font = fm.FontProperties(fname='./NanumBarunGothic.ttf')

    # 막대의 높이가 빈도의 값이 되도록 설정합니다.
    plt.bar(pos, freqs, align='center')

    # 각 막대에 해당되는 단어를 입력합니다.
    plt.xticks(pos, words, rotation='vertical', fontproperties=font)

    # 그래프의 제목을 설정합니다.
    plt.title('단어 별 사용 빈도', fontproperties=font)

    # Y축에 설명을 추가합니다.
    plt.ylabel('빈도', fontproperties=font)

    # 단어가 잘리지 않도록 여백을 조정합니다.
    plt.tight_layout()

    # 그래프를 표시합니다.
    plt.savefig('graph.png')
    elice_utils.send_image('graph.png')


def main(prefix=''):
    # import_corpus() 함수를 통해 튜플의 리스트를 생성합니다.
    corpus = import_corpus('corpus.txt')

    # head로 시작하는 단어들만 골라 냅니다.
    prefix_words = filter_by_prefix(corpus, prefix)

    # 주어진 prefix로 시작하는 단어들을 빈도가 높은 순으로 정렬한 뒤 앞의 10개만 추립니다.
    top_ten = most_frequent_words(prefix_words, 10)

    # 단어 별 빈도수를 그래프로 나타냅니다.
    draw_frequency_graph(top_ten)

    # 'Alice in Wonderland' 책의 단어를 corpus로 바꿉니다.
    alice_files = ['alice/chapter{}.txt'.format(chapter) for chapter in range(1, 6)]
    alice_corpus = create_corpus(alice_files)

    top_ten_alice = most_frequent_words(alice_corpus, 10)
    draw_frequency_graph(top_ten_alice)


if __name__ == '__main__':
    main('t')
