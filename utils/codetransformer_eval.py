
from collections import OrderedDict, Counter

def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def calculate(predict, original):
    '''
    idx already remove special token (unk,pad,...)
    :param predict: the list of list of idx
    :param original: the list of list of idx
    :return: p,r,f
    '''
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(predict, original):
        p, o = sorted(p), sorted(o)
        common = Counter(p) & Counter(o)
        true_positive += sum(common.values())
        false_positive += (len(p) - sum(common.values()))
        false_negative += (len(o) - sum(common.values()))
    return calculate_results(true_positive, false_positive, false_negative)


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1



import os
root_path = ''
all_file_path = [e for e in os.listdir(root_path) if 'output' in e]
all_file_path = [os.path.join(root_path, file_path)
                 for file_path in all_file_path]
all_rst = []
for e in tqdm(all_file_path):
    with open(e) as f:
        all_data = f.readlines()
    predict_list = []
    ref_list = []
    for i, line in enumerate(all_data):
        if i % 3 == 0:
            ref_list.append(eval(line))
        if i % 3 == 1:
            predict_list.append(eval(line)[0])
    tmp = calculate(predict_list, ref_list)
    # print(tmp)
    all_rst.append(tmp)

all_rst.sort(key=lambda x: x[2], reverse=True)
print(all_rst[:10])
