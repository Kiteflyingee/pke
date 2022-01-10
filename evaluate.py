import logging


def eval(gold_list, pred_list, n=10):
    '''
    计算评价指标， 使用micro平均
    Args:
        gold_list: 真实关键短语
        pred_list: 预测关键短语

    Returns:
        micro_precision: 准确率
        micro_recall: 召回率
        micro_f1: f1值
    '''
    num_predict = 0
    num_gold = 0
    num_match = 0
    macro_recalls = []
    macro_precisions = []
    assert len(gold_list) == len(pred_list)
    for i, gold in enumerate(gold_list):
        pred = set([kw for kw, score in pred_list[i][:n]])
        gold = set(gold)
        hits = len(pred & gold)
        num_match += hits
        num_predict += len(pred)
        num_gold += len(gold)
        precision = hits / len(pred) if len(pred) > 0 else 0.0
        recall = hits / len(gold) if len(gold) > 0 else 0.0
        macro_precisions.append(precision)
        macro_recalls.append(recall)
    # micro
    micro_precision = 1. * num_match / num_predict
    micro_recall = 1. * num_match / num_gold
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0.0
    # macro
    macro_precision = sum(macro_precisions) / len(macro_precisions)
    macro_recall = sum(macro_recalls) / len(macro_recalls)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) \
        if (macro_precision + macro_recall) > 0 else 0.0
    logging.info({'macro_precision': macro_precision,
                  'macro_recall': macro_recall,
                  'macro_f1': macro_f1})
    return micro_precision, micro_recall, micro_f1


def eval_v2(gold_list, pred_list):
    '''
    过时的评测，按词共现算准确率
    Args:
        gold_list: 真实关键短语
        pred_list: 预测关键短语

    Returns:
        precision: 准确率
        recall: 召回率
        f1: f1值
    '''
    num_predict = 0
    num_gold = 0
    num_match = 0
    macro_recalls = []
    assert len(gold_list) == len(pred_list)
    for i, gold in enumerate(gold_list):
        pred = ' '.join([kw for kw, score in pred_list[i]]).split()
        gold = ' '.join(gold).split()
        hits = len(set(pred) & set(gold))
        num_match += hits
        num_predict += len(pred)
        num_gold += len(gold)
        macro_recalls.append(hits / len(gold))
    precision = 1. * num_match / num_predict
    recall = 1. * num_match / num_gold
    macro_r = sum(macro_recalls) / len(macro_recalls)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def read_pred(path):
    '''
    读取存储的预测数据
    Args:
        path: 路径

    Returns:
        pred_list: 预测的关键短语list
    '''
    import ast
    pred_list = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip() == '===========================':
                break
            example = ast.literal_eval(line.strip())
            pred_list.append(example)
    return pred_list


if __name__ == '__main__':
    from data_preprocess import read_data

    doc_list, g_list = read_data('data/kp20k_testing.json', True)
    # files = ['YAKE_5.txt', 'YAKE_10.txt', 'SingleRank_5.txt', 'SingleRank_10.txt', 'Tf_idf_10.txt']
    files = ['Tf_idf_10.txt']
    for file in files:
        p_list = read_pred('result/' + file)
        precision, recall, f1 = eval(g_list, p_list, n=10)
        # precision, recall, f1 = eval_v2(g_list, p_list)
        print('当前是{}'.format(file))
        print({
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
