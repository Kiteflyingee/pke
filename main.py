import logging

import pke
from nltk.corpus import stopwords
from tqdm import tqdm
import string

from data_preprocess import read_data
from evaluate import eval
from pke import load_document_frequency_file


def run_text_rank(extractor, n):
    """
    运行textrank算法抽取关键短语
    Args:
        extractor:TextRank抽取器，由调用方法创建
        n: 抽取关键短语个数
    Returns:
        抽取的关键短语
    """
    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor.candidate_weighting(window=2,
                                  pos=pos,
                                  top_percent=0.33)
    keyphrases = extractor.get_n_best(n=n)
    return keyphrases

def run_tf_idf(extractor, n, df):
    """
    运行tf-idf算法抽取关键短语
    Args:
        extractor:tf-idf抽取器，由调用方法创建
        n: 抽取关键短语个数
    Returns:
        抽取的关键短语
    """
    extractor.candidate_selection(n=3, stoplist=list(string.punctuation))
    extractor.candidate_weighting(df=df)
    keyphrases = extractor.get_n_best(n=n)
    return keyphrases

def run_yake(extractor, n):
    """
    运行yake算法抽取关键短语
    Args:
        extractor:yake抽取器，由调用方法创建
        n: 抽取关键短语个数
    Returns:
        抽取的关键短语
    """
    stoplist = stopwords.words('english')
    extractor.candidate_selection(n=3, stoplist=stoplist)
    window = 2
    use_stems = False  # use stems instead of words for weighting
    extractor.candidate_weighting(window=window,
                                  stoplist=stoplist,
                                  use_stems=use_stems)
    threshold = 0.8
    keyphrases = extractor.get_n_best(n=n, threshold=threshold)
    return keyphrases

def run_single_rank(extractor, n):
    """
    运行single_rank算法抽取关键短语
    Args:
        extractor:single_rank抽取器，由调用方法创建
        n: 抽取关键短语个数
    Returns:
        抽取的关键短语
    """
    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10,
                                  pos=pos)
    keyphrases = extractor.get_n_best(n=n)
    return keyphrases


def run_keyword_extract(algo, document_list, gold_list, n=5):
    """
    关键词抽取主方法，可以运行不同算法的实验
    Args:
        algo: 传入的算法名，决定使用对应算法，可选的有YAKE，SingleRank，TextRank，TF_IDF
        document_list:文档的list
        gold_list: 真实的关键短语list
        n: 抽取的短语个数

    Returns:
        抽取的关键短语list
    """
    keyword_list = []
    df = load_document_frequency_file("pke/models/df-semeval2010.tsv.gz", delimiter='\t')
    for doc_text in tqdm(document_list):
        if algo=='YAKE':
            extractor = pke.unsupervised.YAKE()
            # 2. load the content of the document.
            extractor.load_document(input=doc_text,
                                    language='en',
                                    normalization=None)
            keywords = run_yake(extractor, n)
        elif algo=="SingleRank":
            extractor = pke.unsupervised.SingleRank()
            # 2. load the content of the document.
            extractor.load_document(input=doc_text,
                                    language='en',
                                    normalization=None)
            keywords = run_single_rank(extractor, n)
        elif algo == 'TextRank':
            extractor = pke.unsupervised.TextRank()
            extractor.load_document(input=doc_text,
                                    language='en',
                                    normalization=None)
            keywords = run_text_rank(extractor, n)
        else:
            extractor = pke.unsupervised.TfIdf()
            extractor.load_document(input=doc_text,
                                    language='en',
                                    normalization=None)
            keywords = run_tf_idf(extractor, n, df)

        keyword_list.append(keywords)
        precision, recall, f1 = eval(gold_list[:len(keyword_list)], keyword_list)
        # test_result = {'precision': precision,
        #                'recall': recall,
        #                'f1': f1}
        # print(test_result)
    return keyword_list

def exp_main(data_path):
    """
    自动化实验主程序
    """
    # document_list, gold_list = read_data('a.json')
    document_list, gold_list = read_data(data_path, present=True, mode=1)
    for algo in ['YAKE', 'SingleRank', 'TextRank', 'TfIdf']:
        for n in [5, 10]:
            keyword_list = run_keyword_extract(algo, document_list, gold_list, n)
            precision, recall, f1 = eval(gold_list, keyword_list)
            print('\n当前是{}算法，n取{}的结果'.format(algo, n))
            print('precision:{}'.format(precision))
            print('recall:{}'.format(recall))
            print('f1:{}'.format(f1))
            print('**********************')
            test_result = {'precision': precision,
                           'recall': recall,
                           'f1': f1}
            write_result(keyword_list, algo, n, test_result)

def exp_with_arg(data_path):
    """
    命令行实验主程序
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", type=str)
    parser.add_argument("-n", type=int)
    parser.add_argument("-md", default=1, type=int)
    parser.add_argument("-p", action="store_true")
    args = parser.parse_args()
    document_list, gold_list = read_data(data_path, present=args.p, mode=args.md)
    keyword_list = run_keyword_extract(args.algo, document_list, gold_list, args.n)
    precision, recall, f1 = eval(gold_list, keyword_list)
    print('\n当前是{}算法，n取{}的结果'.format(args.algo, args.n))
    print('precision:{}'.format(precision))
    print('recall:{}'.format(recall))
    print('f1:{}'.format(f1))
    print('**********************')
    test_result = {'precision': precision,
                   'recall': recall,
                   'f1': f1}
    write_result(keyword_list, args.algo, args.n, test_result)

def write_result(keyword_list, algo, n, test_result):
    '''
    将抽取的关键短语结果持久化
    Args:
        keyword_list: 抽取的关键短语list
        algo: 使用的算法
        n: 抽取的个数
        test_result: 评测的结果

    '''
    with open('result/' + algo + '_'+str(n) + '.txt', 'w') as f:
        for keyword in keyword_list:
            print(keyword, file=f)
        print('===========================', file=f)
        print(test_result, file=f)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    data_path = 'data/kp20k_testing.json'
    # data_path = 'data/sem_eval_test.json'
    # exp_main(data_path)
    exp_with_arg(data_path)
