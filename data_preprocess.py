import json

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Symbol removal, stop word filtering, part of speech reduction, part of speech tagging
# POS = ['JJ', 'JJR', 'JJS', 'NN', 'NNS']
english_punctuations = [',', '.', ':', ';', '``', '?', '（', '）', '(', ')', '[', ']',
                        '&', '!', '*', '@', '#', '$', '%', '\\', '\"', '}', '{']
# english_punctuations_not_period = [',', ':', ';', '``', '?', '（', '）', '(', ')',
#                                    '[', ']', '&', '!', '*', '@', '#', '$', '%', '\\', '\"', '}', '{']
stops_words = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()


def read_data(path, present=True, mode=1):
    '''
    读取语料
    Args:
        path: 语料路径
        present: 是否只取在摘要中存在的关键短语
        mode: 文本清洗模式
    Returns:
        document_list: 摘要的list
        gold_list: 关键短语的list
    '''
    document_list = []
    gold_list = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            abstract = example['abstract']
            abstract = text_preprocess(abstract, mode)
            keyphrases = example['keyword'].split(';')
            clean_keyphrases = []
            for kp in keyphrases:
                kp = text_preprocess(kp)
                if present and (kp not in abstract):
                    continue
                clean_keyphrases.append(kp)
            gold_list.append(clean_keyphrases)
            document_list.append(abstract)
    return document_list, gold_list

def text_preprocess(text, mode=1):
    '''
    文本简单的预处理
    Args:
        text: 原始文本
        mode: 模式
            有三种可选模式：
            1. 提取词干
            2. 提取词干+去除标点
            3. 提取词干+去除标点+去除停用词
    Returns:
        数据清洗后的文本
    '''
    words = []
    if mode == 1:
        new_text = text.replace('\\', '').lower()
        word_list = nltk.word_tokenize(new_text)  # Tokenizing
        for word in word_list:
            w = porter_stemmer.stem(word)  # Stemming
            words.append(w)
    elif mode == 2:
        new_text = text.replace('\\', '').lower()
        word_list = nltk.word_tokenize(new_text)  # Tokenizing
        for word in word_list:
            # Choose words whose pos are nouns and adjectives.
            if word not in english_punctuations:  # Symbol removal
                w = porter_stemmer.stem(word)  # Stemming
                words.append(w)
    elif mode == 3:
        new_text = text.replace('\\', '').lower()
        word_list = nltk.word_tokenize(new_text)  # Tokenizing
        for word in word_list:
            # Choose words whose pos are nouns and adjectives.
            if word not in english_punctuations and word not in stops_words:  # Symbol removal
                w = porter_stemmer.stem(word)  # Stemming
                words.append(w)
    return " ".join(words)



def dump_doc_kw(document_list, gold_list):
    '''把kp数据集的摘要和关键词分别存储到data/document.txt和data/present_kp.txt中'''
    with open('data/document.txt', 'w', encoding='utf-8') as f:
        for doc in document_list:
            print(doc, file=f)
    with open('data/present_kp.txt', 'w', encoding='utf-8') as f:
        for gold in gold_list:
            print(gold, file=f)
