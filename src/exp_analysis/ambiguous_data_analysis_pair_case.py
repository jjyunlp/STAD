"""
获取ambiguous data中关系共现的数量


First, collect the most related relation pairs.
Give top-K relation pairs
Find the sentences that top-2 relations are
For examples:
if a confusion relation pair 是

"""
from hashlib import new
import sys
import json
import numpy as np
sys.path.insert(1, "/home/jjy/work/ST_RE/src/")
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

    def dump_data(self, filename, data):
        with open(filename, 'w') as writer:
            for inst in data:
                writer.write(json.dumps(inst, indent=4)+"\n")

    def load_data(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_raw_instance()
        return data

    def superset_of_accumulate_top_n_prob(self, data, label2id, easy_prob_threshold, ambiguity_prob_threshold, topN):
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
    
    def count_relation_pair(self, superset_list, id2rel):
        pair2count = {}
        for superset in superset_list:
            for i in range(len(superset)):
                for j in range(i+1, len(superset)):
                    a = id2rel[superset[i]]
                    b = id2rel[superset[j]]
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
    
    def get_cases(self, data, rel_a, rel_b, topN, id2rel, prob):
        """根据模糊的两个关系，去找sentence

        Args:
            data ([type]): [description]
            rel_a ([type]): [description]
            rel_b ([type]): [description]
        """
        easy_prob_threshold = prob
        ambig_prob_threshold = prob
        cases = []
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
                    if sum_of_prob >= ambig_prob_threshold:
                        # 提前满足了条件，就不再往后走
                        break
                if sum_of_prob >= ambig_prob_threshold:
                    # 这是一个ambiguous data，收集下他的label set，即superset
                    ambiguity_label_names = [id2rel[x] for x in ambiguity_labels]
                    gold_relation = inst['gold_relation']   #  并不需要parital的情况
                    cond1 = ambiguity_label_names[0] == rel_a
                    cond2 = ambiguity_label_names[1] == rel_b
                    #if cond1 and cond2:
                    if gold_relation != ambiguity_label_names[0]:
                        # 整理下格式
                        new_inst = {}
                        new_inst['head'] = inst['h']['name']
                        new_inst['tail'] = inst['t']['name']
                        new_inst['sen'] = " ".join(inst['token'])
                        new_inst['rel_a'] = ambiguity_label_names[0]
                        new_inst['rel_a_prob'] = prob_index_tuple[0][0]
                        new_inst['rel_b'] = ambiguity_label_names[1]
                        new_inst['rel_b_prob'] = prob_index_tuple[1][0]
                        new_inst['gold_relation'] = inst['gold_relation']
                        cases.append(new_inst)
        print(cases)
        self.dump_data(f"all_cases.json", cases)

        


                    




if __name__ == "__main__":

    dataset = "semeval"
    dataset = "re-tacred_exclude_NA"
    label_file = f"/home/jjy/work/ST_RE/data/{dataset}/label2id.json"
    with open(label_file) as reader:
        label2id = json.load(reader)
    id2label = {}
    for k,v in label2id.items():
        id2label[v] = k

    seeds = [0, 1, 2, 3, 4]
    seeds = [1]
    easy_prob_threshold = 0.9
    ambig_prob_threshold = 0.9
    relation_pair_count = {}
    avg_relation_count = []
    topN = 2    #  为了方便，目前只取前两个关系满足的情况
    for seed in seeds:
        ambiguous_data_file = f"/data/jjy/ST_RE_micro_accumulate_prob_10_low_resource/base/{dataset}/low_resource_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/pseudo_hard_example_accumulate_prob0.95_top38.txt"
        analysis = AmbiguousDataAnalysis(ambiguous_data_file)
        ambiguous_data = analysis.load_data()
        superset_list = analysis.superset_of_accumulate_top_n_prob(ambiguous_data, label2id, easy_prob_threshold, ambig_prob_threshold, topN)
        pair2count = analysis.count_relation_pair(superset_list, id2label)

        index = 8
        rel_a, rel_b = list(pair2count.keys())[index].split("#")
        rel_a = "org:top_members/employees"
        rel_b = "org:founded_by"
        analysis.get_cases(ambiguous_data, rel_a, rel_b, topN, id2label, easy_prob_threshold)
    print(pair2count)


