import pke
from nltk.corpus import stopwords
from tqdm import tqdm
import string

from data_preprocess import read_data
from evaluate import eval

def run_text_rank(extractor, n):
    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor.candidate_weighting(window=2,
                                  pos=pos,
                                  top_percent=0.33)
    keyphrases = extractor.get_n_best(n=n)
    return keyphrases

def run_tf_idf(extractor, n):
    extractor.candidate_selection(n=3, stoplist=list(string.punctuation))
    df = pke.load_document_frequency_file(input_file='tf_idf.tsv.gz')
    extractor.candidate_weighting(df=df)
    keyphrases = extractor.get_n_best(n=n)
    return keyphrases

def run_yake(extractor, n):
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
    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10,
                                  pos=pos)
    keyphrases = extractor.get_n_best(n=n)
    return keyphrases


def run_keyword_extract(algo, document_list, gold_list, n=5):
    # 1. create a YAKE extractor.
    keyword_list = []
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

        keyword_list.append(keywords)
        precision, recall, f1 = eval(gold_list[:len(keyword_list)], keyword_list)
        test_result = {'precision': precision,
                       'recall': recall,
                       'f1': f1}
        print(test_result)
    return keyword_list

def exp_main():
    # document_list, gold_list = read_data('a.json')
    document_list, gold_list = read_data('data/kp20k_testing.json')
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

def write_result(keyword_list, algo, n, test_result):
    with open('result/' + algo + '_'+str(n) + '.txt', 'w') as f:
        for keyword in keyword_list:
            print(keyword, file=f)
        print('===========================', file=f)
        print(test_result, file=f)

def exp_with_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", type=str)
    parser.add_argument("-n", type=int)
    args = parser.parse_args()
    document_list, gold_list = read_data('data/kp20k_testing.json', present=True)
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


if __name__ == '__main__':
    # exp_main()
    exp_with_arg()
