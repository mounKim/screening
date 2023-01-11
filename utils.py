import os
from os.path import join


def return_embedding():
    f = open('data/all.review.vec.txt', 'r')
    data = f.readlines()[1:]
    embedding = {}
    for d in data:
        word = d.split()[0]
        emb = d.split()[1:]
        embedding[word] = emb
    return embedding


def return_alltext():
    positive = sorted(list(map(int, os.listdir('./data/train/positive'))))
    # negative = sorted(list(map(int, os.listdir('./data/train/negative'))))
    text_data = []
    len_data = []
    for d in range(2000):
        f = open(join('./data/train/all', str(d)), 'r')
        c = f.readlines()[0]
        if d in positive:
            text_data.append([c, 1])
        else:
            text_data.append([c, 0])
        len_data.append(len(c))
    return text_data


def return_alltext_test():
    positive = sorted(list(map(int, os.listdir('./data/test_answer/positive'))))
    # negative = sorted(list(map(int, os.listdir('./data/test_answer/negative'))))
    text_data = []
    len_data = []
    for d in range(2000):
        f = open(join('./data/test', str(d)), 'r')
        c = f.readlines()[0]
        if d in positive:
            text_data.append([c, 1])
        else:
            text_data.append([c, 0])
        len_data.append(len(c))
    return text_data


def return_top10000():
    text = return_alltext()
    word_dict = {}
    for i in text:
        word_list = i[0].split()
        for w in word_list:
            if w not in word_dict:
                word_dict[w] = 1
            else:
                word_dict[w] += 1
    sorted_dict = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
    words = []
    counts = []
    for i in sorted_dict[:10000]:
        words.append(i[0])
        counts.append(i[1])
    return words, counts
