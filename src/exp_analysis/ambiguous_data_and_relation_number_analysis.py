"""
获取ambiguous data中关系共现的数量

给出模糊数据中，按照关系数目大小的分布情况。

"""
from operator import invert
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, "/data3/jjyu/work/STAD_copy/STAD/src")
from data_processor.data_loader_and_dumper import JsonDataLoader, JsonDataDumper


class DataAnalysis(object):
    def __init__(self, input_file):
        self.input_file = input_file
    
    def load_data(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_raw_instance()
        return data


class AmbiguousDataAnalysis():
    def __init__(self, input_file, drop_NA=False):
        self.input_file = input_file

    def load_data(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_raw_instance()
        return data

    def superset_of_accumulate_top_n_prob(self, data, label2id=None, easy_prob_threshold=None, ambiguity_prob_threshold=None, topN=None):
        """
        topN个预测的概率总和大于等于prob_threshold，这样的句子作为ambiguity labeling。
        因此，将每个句子的superset提取出来
        """
        superset_list = []  # a list of list
        for inst in data:
            if 'gold_relation' in inst:     #   确认下确实是预测过后的句子，否则没有gold_relation的项目，只有relation
                pred_rel_prob = inst['prob']
                if pred_rel_prob > easy_prob_threshold: # or pred_rel_id == 0:
                    # 这些是easy example
                    # 我们这边输入的是ambiguous data，原代码是针对所有pseudo data，先留着吧
                    continue
                ambiguity_labels = []
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                sum_of_prob = 0.0
                for prob, index in prob_index_tuple[:topN]:
                    ambiguity_labels.append(index)
                    sum_of_prob += prob
                    if sum_of_prob >= ambiguity_prob_threshold:
                        # 提前满足了条件，就不再往后走
                        break
                if sum_of_prob >= ambiguity_prob_threshold:
                    # 这是一个ambiguous data，收集下他的label set，即superset
                    superset_list.append(ambiguity_labels)
        return superset_list
    
    def count_relation_co_occurrence(self, superset_list):
        # 输入一个含有各种superset的list，统计关系两两共现的频次
        # define a 10*10 list, to count co-occurrence
        #for superset in superset_list:
        #    print(superset)
        relation_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        relation_pair_count = [[0]*10]*10       #   这种做法，会导致每一行的都是相同地址，因此，修改第一行就相当于修改其他所有行。很奇怪，值得研究下
        relation_pair_count = [
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                ]
        for superset in superset_list:
            # 开始遍历
            for i in range(len(superset)):
                for j in range(i+1, len(superset)):
                    a = superset[i]
                    b = superset[j]
                    # 对称的，最后上角或下角就行就行
                    relation_pair_count[a][b] += 1/len(superset)    # 算下占比，总不能[0,1] 和[1,2,3,4]中的一样吧
                    relation_pair_count[b][a] += 1/len(superset)
            for rel_id in superset:
                #relation_count[rel_id] += 1
                relation_count[rel_id] += 1/len(superset)     # 带权重
        #for item in relation_pair_count:
        #    print(item)
        #    print(sum(item))
        print(relation_count)
        return relation_count
    
    def count_relation_pair(self, superset_list):
        pair2count = {}
        for superset in superset_list:
            for i in range(len(superset)):
                for j in range(i+1, len(superset)):
                    a = superset[i]
                    b = superset[j]
                    relation_pair = a + "#" + b
                    if relation_pair not in pair2count:
                        pair2count[relation_pair] = 1
                    else:
                        pair2count[relation_pair] += 1
                    # 反过来计算下，省的搞错
                    relation_pair = b + "#" + a
                    if relation_pair not in pair2count:
                        pair2count[relation_pair] = 1
                    else:
                        pair2count[relation_pair] += 1
        pair2count = {k: v for k, v in sorted(pair2count.items(), key=lambda item: item[1], reverse=True)}
        return pair2count
                    
                    
def draw_barh(data):
    """Draw a beautiful barh for show the data distribution on confident data and ambiguous data,
    Especially the distribution on relation number of ambiguous data.


    Args:
        data (_type_): _description_
    """
    # init with confident data info
    labels = [str(data[0][0])]
    numbers = [data[0][1]]
    """
    for i in range(1, 11, 2):
        label = f"{data[i][0]}-{data[i+1][0]}"
        labels.append(label)
        numbers.append(data[i][1] + data[i+1][1])
    """

    step = 5
    splits = [[2, 2], [3, 3], [4,4], [5,5],[6,10], [11, 15], [16, 20], [21, 25], [26, 30]]
    splits = [[2, 2], [3, 3], [4,4], [5,5],[6,6], [7,7], [8,8],[9,9], [10,10], [11,11],[12,12],[13,13],[14,14], [15, 15], [16, 16], [17, 17], [18,18], [19,19], [20,20], [21, 28]]
    for split in splits:
        label = f"{split[0]}-{split[1]}"
        label = f"{split[0]}"
        if split == splits[-1]:
            label = ">20"

        number = 0
        for i in range(split[0], split[1]+1):
            print(i-1)
            print("---")
            print(data[i-1])
            number += data[i-1][1]
        labels.append(label)
        numbers.append(number)
            
    """
    for i in range(1, 11):
        label = str(data[i][0])
        number = data[i][1]

        labels.append(label)
        numbers.append(number)
    others = 0
    for i in range(11, len(data)):
        others += data[i][1]
    labels.append('>')
    numbers.append(others)
    """

    
    labels.reverse()
    numbers.reverse()
    plt.barh(labels, numbers)
    plt.title("Distribution of Auto-Annotated Instances")
    plt.savefig('./sentence_with_answer_size_v2.png')



if __name__ == "__main__":
    # For draw the ambiguous size -> sentence number
    prob_threshold = 0.9
    ambiguous_data_file = "/data/jjy/ST_RE_micro_accumulate_prob_20_low_resource/base/re-tacred_exclude_NA/low_resource_exp01/batch32_epoch20_fix_lr5e-05_seed3/pseudo_hard_example_accumulate_prob0.9_top38.txt"

    ambiguous_data_file = f"/data4/jjyunlp/rc_output/STAD_diff_data/base/re-tacred_exclude_NA/low_resource_exp_3/batch32_epoch20_fix_lr5e-05_seed3/pseudo_hard_example_accumulate_prob{prob_threshold}_top38.txt"

    analysis = AmbiguousDataAnalysis(ambiguous_data_file)
    ambiguous_data = analysis.load_data()
    superset_list = analysis.superset_of_accumulate_top_n_prob(ambiguous_data, easy_prob_threshold=prob_threshold, ambiguity_prob_threshold=prob_threshold, topN=38)
    print(superset_list[:10])

    partial_size_2_count = {}
    for partial_labels in superset_list:
        size = len(partial_labels)
        if size not in partial_size_2_count:
            partial_size_2_count[size] = 1
        else:
            partial_size_2_count[size] += 1
    print(partial_size_2_count)
    partial_size_2_count[1] = 8416
    info = sorted(partial_size_2_count.items(), key=lambda item: item[0], reverse=False)
    print(info)
    
    draw_barh(info)


