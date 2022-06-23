"""
分析实验中每个关系的性能变化情况
"""
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


class RelationResultAnalysis():
    """分析每一个关系的性能
    """
    def __init__(self, input_file):
        self.input_file = input_file

    def load_data(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_raw_instance()
        return data
    
    def read_results(self):
        with open(self.input_file) as reader:
            result = json.load(reader)
            #print(result)
        return result
    
    def extract_each_relation_f1(self, result, mode='micro'):
        # result 是包含所有结果信息的，包括val/test，overall和各个关系的一个json
        relation_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if mode == 'micro':
            for rel_id, prf in result['test']['verbose'].items():
                relation_result[int(rel_id)] = prf['f1']
        else:
            relation_result = result['test']['categoried_macro_f1']
        # print(relation_result)
        return relation_result


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
        relation_pair_count = [[0]*10]*10       #   这种做法，会导致每一行的都是相同地址，因此，修改第一行就相当于修改其他所有行
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
                    relation_pair_count[a][b] += 1
                    relation_pair_count[b][a] += 1
            for rel_id in superset:
                relation_count[rel_id] += 1
        for item in relation_pair_count:
            print(item)
            print(sum(item))
        print(relation_count)
        return relation_pair_count



if __name__ == "__main__":

    dataset = "semeval"
    #dataset = "re-tacred"
    mode = 'micro'
    #mode = 'macro'
    dataset_name = f"top10-{dataset}"
    label_file = f"/home/jjy/work/ST_RE/data/{dataset}_top10_label_excluding_NA/label2id.json"
    with open(label_file) as reader:
        label2id = json.load(reader)

    part = 10
    seeds = [0, 1, 2, 3, 4]
    easy_prob_threshold = 0.95
    ambig_prob_threshold = 0.95
    topN = 5
    relation_pair_count = {}
    our_easy_minus = []     # 提升的比例
    our_easy_abs_minus = []     # 提升的绝对值
    for seed in seeds:
        base_result_file = f"/data/jjy/ST_RE_{mode}_accumulate_prob_10_all_new/base/top10-{dataset}/balanced_small_data_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/results"
        base_analysis = RelationResultAnalysis(base_result_file)
        base_result = base_analysis.read_results()
        #print(base_result)
        base_relation_f1 = base_analysis.extract_each_relation_f1(base_result, mode)

        easy_result_file = f"/data/jjy/ST_RE_{mode}_accumulate_prob_10_all_new/merge_easy_example_prob0.95/top10-{dataset}/balanced_small_data_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/epoch_0/results"
        easy_analysis = RelationResultAnalysis(easy_result_file)
        easy_result = easy_analysis.read_results()
        easy_relation_f1 = base_analysis.extract_each_relation_f1(easy_result, mode)

        easy_and_ambig2_result_file = f"/data/jjy/ST_RE_{mode}_accumulate_prob_10_all_new/merge_easy_and_ambig2_prob0.95_top5_one_loss_batch32_by_sum_negative_loss/top10-{dataset}/balanced_small_data_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/epoch_0/results"
        easy_and_ambig2_analysis = RelationResultAnalysis(easy_and_ambig2_result_file)
        easy_and_ambig2_result = easy_and_ambig2_analysis.read_results()
        easy_and_ambig2_relation_f1 = base_analysis.extract_each_relation_f1(easy_and_ambig2_result, mode)

        our_result_file = f"/data/jjy/ST_RE_{mode}_accumulate_prob_10_all_new/two_stage_easy_and_ambig2_to_gold_lr1_5e-05_lr2_5e-05_prob0.95_top5_batch32_by_sum_negative_loss/top10-{dataset}/balanced_small_data_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/epoch_0/second/results"
        our_result_file = f"/data/jjy/ST_RE_{mode}_accumulate_prob_10_all_new/two_stage_easy_and_ambig2_to_gold_lr1_5e-05_lr2_5e-05_prob0.95_top5_batch32_by_random_one_negative_loss/top10-{dataset}/balanced_small_data_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/epoch_0/second/results"
        our_analysis = RelationResultAnalysis(our_result_file)
        our_result = our_analysis.read_results()
        #print(our_result)
        our_relation_f1 = our_analysis.extract_each_relation_f1(our_result, mode)
        #minus = [round((x-y)*100, 1) for (x, y) in zip(our_relation_f1, base_relation_f1)]
        #print(minus)
        minus = [round(((x-y)/y)*100, 1) for (x, y) in zip(our_relation_f1, easy_relation_f1)]
        abs_minus = [round(((x-y))*100, 1) for (x, y) in zip(our_relation_f1, easy_relation_f1)]
        #print(minus)
        our_easy_minus.append(minus)
        our_easy_abs_minus.append(abs_minus)
        #minus = [round((x-y)*100, 1) for (x, y) in zip(easy_and_ambig2_relation_f1, easy_relation_f1)]
        #print(minus)
        print("base f1")
        print(easy_relation_f1)
        print("NTPL f1")
        print(our_relation_f1)
    our_easy_minus = np.mean(np.array(our_easy_minus), axis=0)
    print("percentage improvement")
    print(our_easy_minus)
    print("Absolute Improvment")
    our_easy_abs_minus = np.mean(np.array(our_easy_abs_minus), axis=0)
    print(our_easy_abs_minus)
