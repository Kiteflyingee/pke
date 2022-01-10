import json


def read_data(path, present=True):
    document_list = []
    gold_list = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            abstract = example['abstract']
            keyword = example['keyword'].split(';')
            if present:
                present_kw = []
                for kw in keyword:
                    if kw in abstract:
                        present_kw.append(kw)
                gold_list.append(present_kw)
            else:
                gold_list.append(keyword)
            document_list.append(abstract)
    return document_list, gold_list


def dump_doc_kw(document_list, gold_list):
    with open('data/document.txt', 'w', encoding='utf-8') as f:
        for doc in document_list:
            print(doc, file=f)
    with open('data/present_kp.txt', 'w', encoding='utf-8') as f:
        for gold in gold_list:
            print(gold, file=f)


if __name__ == '__main__':
    d_list, g_list = read_data('data/kp20k_testing.json')
    dump_doc_kw(d_list, g_list)