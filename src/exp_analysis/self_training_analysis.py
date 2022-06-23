"""
在小数据实验上，我们使用剩余的training set作为unlabel data，因此，我们可以验证新增数据的准确率。
"""

import sys
import json
from sklearn.metrics import accuracy_score
sys.path.insert(1, "/home/jjy/work/ST_RE/src/")
from data_processor.data_loader_and_dumper import JsonDataLoader, JsonDataDumper
from utils.compute_metrics import EvaluationAndAnalysis


class DataAnalysis(object):
    def __init__(self, input_file):
        self.input_file = input_file
    
    def load_data(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_raw_instance()
        return data


class SelfTrainingAnalysis(DataAnalysis):
    def check_accuacy(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_instance()
        correct, wrong = 0, 0
        correct_dict = {}
        wrong_dict = {}
        for inst in data:
            if inst.inst['relation'] == inst.inst['original_label']:
                correct += 1
                print(inst.inst['original_label'], inst.inst['relation'])
                if inst.label not in correct_dict:
                    correct_dict[inst.label] = 1
                else:
                    correct_dict[inst.label] += 1
            else:
                wrong += 1
                #print(inst.inst['original_label'], inst.inst['relation'])
                if inst.label not in wrong_dict:
                    wrong_dict[inst.label] = 1
                else:
                    wrong_dict[inst.label] += 1
        print(correct, wrong, correct_dict, len(correct_dict))
        print(wrong_dict, len(wrong_dict))
    
    def get_avg_entity_distance(self, data):
        """Get the average entity distance of this data.
        Entity distance: how many tokens between two entities in a sentence (tokens or words are fine)

        Args:
            data (list):  a list of json-based examples
        """
        avg_distance = 0
        i = 0
        for inst in data:
            # [E1] .. [/E1] .. [E2] ..[/E2].. or [E2] .. [/E2] .. [E1] ..[/E1].. 
            distance = max(inst['t']['pos'][0] - inst['h']['pos'][1], inst['h']['pos'][0] - inst['t']['pos'][1])
            avg_distance += distance
        avg_distance /= len(data)
        avg_distance = round(avg_distance, 1)
        print("avg entity distance:", avg_distance, len(data))

    def get_avg_sen_length(self, data):
        """Get the average sen length of this data.
        Args:
            data (list):  a list of json-based examples
        """
        avg_sen_len = 0
        i = 0
        for inst in data:
            # [E1] .. [/E1] .. [E2] ..[/E2].. or [E2] .. [/E2] .. [E1] ..[/E1].. 
            distance = max(inst['t']['pos'][0] - inst['h']['pos'][1], inst['h']['pos'][0] - inst['t']['pos'][1])
            sen_len = len(inst['token'])
            avg_sen_len += sen_len

        avg_sen_len /= len(data)
        avg_sen_len = round(avg_sen_len, 1)
        print("avg sen length:", avg_sen_len, len(data))
    
    def get_top_prob_examples(self, data, N):
        """sort the data and then return top-N example in read-friendly mode.

        Args:
            data ([type]): [description]
        """
        data.sort(key=lambda k: (float(k['prob'])), reverse=True)
        top_data = data[:N]
        bottom_data = data[-N:]
        top_data = self.read_friendly_change(top_data)
        bottom_data = self.read_friendly_change(bottom_data)
        return (top_data, bottom_data)

    def read_friendly_change(self, data):
        new_data = []
        for inst in data:
            new_inst = {}
            sen = [] 
            for i, token in enumerate(inst['token']):
                if i == inst['h']['pos'][0]:
                    sen.append("[E1]")
                if i == inst['h']['pos'][1]:
                    sen.append("[/E1]")
                if i == inst['t']['pos'][0]:
                    sen.append("[E2]")
                if i == inst['t']['pos'][1]:
                    sen.append("[/E2]")
                sen.append(token)
            new_inst['sen'] = " ".join(sen)
            new_inst['pred_relation'] = inst['relation']
            new_inst['gold_relation'] = inst['gold_relation']
            new_inst['prob'] = inst['prob']
            new_data.append(new_inst)
        return new_data


    def read_friendly_change_for_gold_data(self, data):
        new_data = []
        for inst in data:
            new_inst = {}
            sen = [] 
            for i, token in enumerate(inst['token']):
                if i == inst['h']['pos'][0]:
                    sen.append("[E1]")
                if i == inst['h']['pos'][1]:
                    sen.append("[/E1]")
                if i == inst['t']['pos'][0]:
                    sen.append("[E2]")
                if i == inst['t']['pos'][1]:
                    sen.append("[/E2]")
                sen.append(token)
            new_inst['sen'] = " ".join(sen)
            new_inst['gold_relation'] = inst['relation']
            new_data.append(new_inst)
        return new_data


class PartialAnnotationAcc(DataAnalysis):
    """统计部分标注后的准确率，如果这个高了，那做这个才有希望。

    Args:
        object ([type]): [description]
    """
    def __init__(self, input_file, drop_NA=False):
        self.input_file = input_file
        self.drop_NA = drop_NA
    
    def load_data(self):
        loader = JsonDataLoader(self.input_file)
        data = loader.load_all_raw_instance()
        return data
    
    def accuracy(self, data, n):
        analysis = EvaluationAndAnalysis()
        gold_all = []   # except the padding inst
        pred_all = []
        for inst in data:
            if 'gold_relation' in inst:
                gold_all.append(label2id[inst['gold_relation']])
                pred_all.append(label2id[inst['relation']])     # the pred relation
        result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        acc = accuracy_score(gold_all, pred_all)
        print(n, acc)
    
    def accuracy_of_prob_threshold(self, data, label2id, prob_threshold):
        gold_all = []   # except the padding inst
        pred_all = []
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_id == 0 and self.drop_NA:
                    continue
                if pred_rel_prob > prob_threshold:
                    gold_rel_id = label2id[inst['gold_relation']]
                    gold_all.append(gold_rel_id)
                    pred_all.append(pred_rel_id)

        acc = accuracy_score(gold_all, pred_all)
        print(acc)
        print(len(gold_all))

    def accuracy_of_n_labels(self, data, label2id, n=1):
        gold_all = []   # except the padding inst
        pred_all = []
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_prob > 0.95:
                    continue
                #if pred_rel_id == 0:
                #    continue
                gold_rel_id = label2id[inst['gold_relation']]
                gold_all.append(gold_rel_id)
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                top_n_tuple = sorted(index_with_prob, reverse=True)[:n]
                top_n_prob, top_n_index = zip(*top_n_tuple)

                for index in top_n_index:
                    if index not in label_count_dict:
                        label_count_dict[index] = 1
                    else:
                        label_count_dict[index] += 1
                if gold_rel_id in top_n_index:
                    pred_all.append(gold_rel_id)
                else:
                    pred_all.append(pred_rel_id)
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        acc = accuracy_score(gold_all, pred_all)
        print(n, acc, n)
        print(label_count_dict)
        print(len(gold_all))

    def accuracy_of_ambiguity_prob(self, data, label2id, ambiguity_prob):
        """
        select all label that large than ambiguity_prob as one of labels
        """
        gold_all = []   # except the padding inst
        pred_all = []
        avg_label_num = 0
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_prob > 0.95 or pred_rel_id == 0:
                    continue
                ambiguity_labels = []
                gold_rel_id = label2id[inst['gold_relation']]
                gold_all.append(gold_rel_id)
                for index, prob in enumerate(inst['distri']):
                    if prob >= ambiguity_prob: # and index != 0:
                        ambiguity_labels.append(index)
                        if index not in label_count_dict:
                            label_count_dict[index] = 1
                        else:
                            label_count_dict[index] += 1
                avg_label_num += len(ambiguity_labels)
                if gold_rel_id in ambiguity_labels:
                    pred_all.append(gold_rel_id)
                else:
                    pred_all.append(pred_rel_id)
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        acc = accuracy_score(gold_all, pred_all)
        print(ambiguity_prob, acc, avg_label_num/len(gold_all))
        print(label_count_dict)
        print(len(gold_all))

    def accuracy_of_accumulate_ambiguity_prob(self, data, label2id, ambiguity_prob):
        """
        select all label that large than ambiguity_prob as one of labels
        """
        gold_all = []   # except the padding inst
        pred_all = []
        avg_label_num = 0
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_id == 0 and self.drop_NA:
                    continue
                if pred_rel_prob > 0.95: # or pred_rel_id == 0:
                    continue
                ambiguity_labels = []
                gold_rel_id = label2id[inst['gold_relation']]
                gold_all.append(gold_rel_id)
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                sum_of_prob = 0.0
                for prob, index in prob_index_tuple:
                    ambiguity_labels.append(index)
                    sum_of_prob += prob
                    if index not in label_count_dict:
                        label_count_dict[index] = 1
                    else:
                        label_count_dict[index] += 1
                    if sum_of_prob >= ambiguity_prob:
                        break
                avg_label_num += len(ambiguity_labels)
                if gold_rel_id in ambiguity_labels:
                    pred_all.append(gold_rel_id)
                else:
                    pred_all.append(pred_rel_id)
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        print(len(pred_all))
        acc = accuracy_score(gold_all, pred_all)
        print(ambiguity_prob, acc, avg_label_num/len(gold_all))
        print(label_count_dict)
        print(len(gold_all))

    def accuracy_of_accumulate_top_n_prob(self, data, label2id, easy_prob_threshold, ambiguity_prob_threshold, topN):
        """
        topN个预测的概率总和大于等于prob_threshold，这样的句子作为ambiguity labeling。
        剩余的还是被抛弃了。
        """
        gold_all = []   # except the padding inst
        pred_all = []
        hard_gold_all = []
        hard_pred_all = []
        hard_ambig_pred_all = []    # 也按照top排序
        top1_pred_all = []
        avg_label_num = 0
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_id == 0 and self.drop_NA:
                    continue
                if pred_rel_prob > easy_prob_threshold: # or pred_rel_id == 0:
                    # 这些是easy example
                    continue
                ambiguity_labels = []
                gold_rel_id = label2id[inst['gold_relation']]
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                sum_of_prob = 0.0
                for prob, index in prob_index_tuple[:topN]:
                    ambiguity_labels.append(index)
                    sum_of_prob += prob
                    if sum_of_prob >= ambiguity_prob_threshold:
                        # 提前满足了条件，就不再往后走
                        break
                if self.drop_NA and 0 in ambiguity_labels:
                    continue
                if sum_of_prob >= ambiguity_prob_threshold:
                    gold_all.append(gold_rel_id)
                    top1_pred_all.append(ambiguity_labels[0])   # 添加概率最高的
                    # topN约束下可用的数据
                    for prob, index in prob_index_tuple[:topN]:
                        if index not in label_count_dict:
                            label_count_dict[index] = 1
                        else:
                            label_count_dict[index] += 1
                    avg_label_num += len(ambiguity_labels)
                    if gold_rel_id in ambiguity_labels:
                        pred_all.append(gold_rel_id)
                    else:
                        pred_all.append(pred_rel_id)
                else:
                    hard_gold_all.append(gold_rel_id)
                    hard_pred_all.append(pred_rel_id)
                    if gold_rel_id in ambiguity_labels[:topN]:
                        hard_ambig_pred_all.append(gold_rel_id)
                    else:
                        hard_ambig_pred_all.append(pred_rel_id)
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        top1_acc = accuracy_score(gold_all, top1_pred_all)
        print(f"Top1 acc: {top1_acc}")
        acc = accuracy_score(gold_all, pred_all)
        avg_label_num /= len(gold_all)
        print(f"Ambi num = {len(gold_all)} and top1 acc={top1_acc} and ambig acc ={acc} avg label num={avg_label_num}")

        hard_acc = accuracy_score(hard_gold_all, hard_pred_all)
        hard_ambig_acc = accuracy_score(hard_gold_all, hard_ambig_pred_all)
        print(f"Hard num = {len(hard_gold_all)} and top1 acc={hard_acc} and ambig acc ={hard_ambig_acc}")

    def accuracy_of_accumulate_some_n_larger_than_prob(self, data, label2id, easy_prob_threshold, ambiguity_prob_threshold):
        """
        多个概率都大于比如0.1，全都标记为label
        还需要统计一个top1的准确率，如果这个很高，那我们的方法就很难超越直接添加的方法
        理想的场景：top-1准确率低，N个的准确率上去了。
        """
        gold_all = []   # except the padding inst
        pred_all = []
        top1_pred_all = []
        avg_label_num = 0
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_id == 0 and self.drop_NA:
                    continue
                if pred_rel_prob > easy_prob_threshold: # or pred_rel_id == 0:
                    # 这些是easy example
                    continue
                #if pred_rel_prob > 0.5:
                #    # 不是模糊数据
                #    continue
                ambiguity_labels = []
                gold_rel_id = label2id[inst['gold_relation']]
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                sum_of_prob = 0.0

                for prob, index in prob_index_tuple:
                    if prob < ambiguity_prob_threshold:
                        break
                    ambiguity_labels.append(index)
                    sum_of_prob += prob
                if len(ambiguity_labels) == 2 and sum_of_prob > easy_prob_threshold:  # 至少两个，且总的要大于一定阈值
                    # print(prob_index_tuple)
                    gold_all.append(gold_rel_id)
                    for index in ambiguity_labels:
                        if index not in label_count_dict:
                            label_count_dict[index] = 1
                        else:
                            label_count_dict[index] += 1
                    avg_label_num += len(ambiguity_labels)
                    top1_pred_all.append(ambiguity_labels[0])   # 添加概率最高的
                    if gold_rel_id in ambiguity_labels:
                        pred_all.append(gold_rel_id)
                    else:
                        pred_all.append(pred_rel_id)
                else:
                    continue
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        top1_acc = accuracy_score(gold_all, top1_pred_all)
        print(f"Top1 acc: {top1_acc}")
        acc = accuracy_score(gold_all, pred_all)
        print(ambiguity_prob_threshold, acc, avg_label_num/len(gold_all))
        print(label_count_dict)
        print(len(gold_all))

    def accuracy_of_accumulate_top_n_with_constraint(self, data, label2id, easy_prob_threshold, ambiguity_prob_threshold, topN=2):
        """
        top-2,假设就前两个，要求top1_prob / top2_prob <=4，不能差的太多，要不然就不是ambiguous
        还需要统计一个top1的准确率，如果这个很高，那我们的方法就很难超越直接添加的方法
        理想的场景：top-1准确率低，N个的准确率上去了。
        """
        print(f"Total unlabeled examples: {len(data)}")
        gold_all = []   # except the padding inst
        pred_all = []
        hard_gold_all = []
        hard_pred_all = []
        hard_ambig_pred_all = []    # 也按照top排序
        top1_pred_all = []
        avg_label_num = 0
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                if pred_rel_id == 0 and self.drop_NA:
                    continue
                if pred_rel_prob > easy_prob_threshold: # or pred_rel_id == 0:
                    # 这些是easy example
                    continue
                #if pred_rel_prob > 0.5:
                #    # 不是模糊数据
                #    continue
                ambiguity_labels = []
                ambiguity_probs = []
                gold_rel_id = label2id[inst['gold_relation']]
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                sum_of_prob = 0.0
                for prob, index in prob_index_tuple:
                    ambiguity_labels.append(index)
                    sum_of_prob += prob
                    ambiguity_probs.append(prob)    # sum 那个可以用这个替代，sum(list)
                    if sum_of_prob >= easy_prob_threshold:
                        break
                # 也可以多个label组成的概率和大于阈值，但每一个概率都和top1比较是小于某个比例
                rate = 5
                if len(ambiguity_labels) == topN:
                    isAmbig = True
                    for i in range(topN):
                        if ambiguity_probs[0]/ambiguity_probs[i] > rate:
                            isAmbig = False
                            break
                    if isAmbig:
                        gold_all.append(gold_rel_id)
                        for index in ambiguity_labels:
                            if index not in label_count_dict:
                                label_count_dict[index] = 1
                            else:
                                label_count_dict[index] += 1
                        avg_label_num += len(ambiguity_labels)
                        top1_pred_all.append(ambiguity_labels[0])   # 添加概率最高的
                        if gold_rel_id in ambiguity_labels:
                            pred_all.append(gold_rel_id)
                        else:
                            pred_all.append(pred_rel_id)
                    else:   # 算hard example
                        hard_gold_all.append(gold_rel_id)
                        hard_pred_all.append(pred_rel_id)
                        if gold_rel_id in ambiguity_labels[:topN]:  # topN=2
                            hard_ambig_pred_all.append(gold_rel_id)
                        else:
                            hard_ambig_pred_all.append(pred_rel_id)
                else:
                    hard_gold_all.append(gold_rel_id)
                    hard_pred_all.append(pred_rel_id)
                    if gold_rel_id in ambiguity_labels[:topN]:  # topN=2
                        hard_ambig_pred_all.append(gold_rel_id)
                    else:
                        hard_ambig_pred_all.append(pred_rel_id)
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        top1_acc = accuracy_score(gold_all, top1_pred_all)
        print(f"Top1 acc: {top1_acc}")
        acc = accuracy_score(gold_all, pred_all)
        # print(ambiguity_prob_threshold, acc, avg_label_num/len(gold_all))
        print(f"Ambi num = {len(gold_all)} and top1 acc={top1_acc} and ambig acc ={acc}")
        #print(label_count_dict)
        #print(len(gold_all))
        hard_acc = accuracy_score(hard_gold_all, hard_pred_all)
        hard_ambig_acc = accuracy_score(hard_gold_all, hard_ambig_pred_all)
        print(f"Hard num = {len(hard_gold_all)} and top1 acc={hard_acc} and ambig acc ={hard_ambig_acc}")

    def accuracy_of_sigmoid_prob(self, data, label2id, prob_threshold):
        """
        大于prob的label都作为positive。因为是二分类，尝试0.5
        easy example指的是只有一个概率大于这个阈值，其他都小于。
        """
        gold_all = []   # except the padding inst
        pred_all = []
        top1_pred_all = []
        avg_label_num = 0
        label_count_dict = {}
        for inst in data:
            if 'gold_relation' in inst:
                pred_rel_id = label2id[inst['relation']]
                pred_rel_prob = inst['prob']
                ambiguity_labels = []
                gold_rel_id = label2id[inst['gold_relation']]
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)

                for prob, index in prob_index_tuple:
                    if prob < prob_threshold:
                        break
                    ambiguity_labels.append(index)
                if len(ambiguity_labels) >= 2:  # 至少两个
                    print(inst)
                    gold_all.append(gold_rel_id)
                    for index in ambiguity_labels:
                        if index not in label_count_dict:
                            label_count_dict[index] = 1
                        else:
                            label_count_dict[index] += 1
                    avg_label_num += len(ambiguity_labels)
                    top1_pred_all.append(ambiguity_labels[0])   # 添加概率最高的
                    if gold_rel_id in ambiguity_labels:
                        pred_all.append(gold_rel_id)
                    else:
                        pred_all.append(pred_rel_id)
        # result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        top1_acc = accuracy_score(gold_all, top1_pred_all)
        print(f"Top1 acc: {top1_acc}")
        acc = accuracy_score(gold_all, pred_all)
        print(prob_threshold, acc, avg_label_num/len(gold_all))
        print(label_count_dict)
        print(len(gold_all))


def run_partial_analysis(file_name, label2id):
    label_num = len(label2id)
    drop_NA = False
    high_prob = (label_num - 1) / label_num
    low_prob = 1 / label_num
    high_prob = 0.95
    ambig_prob = 0.95
    # ambig_prob = high_prob

    analysis = PartialAnnotationAcc(file_name, drop_NA=drop_NA)
    data = analysis.load_data()
    analysis.accuracy_of_prob_threshold(data, label2id, high_prob)
    #analysis.accuracy_of_n_labels(data, label2id, n=1)

    #analysis.accuracy_of_ambiguity_prob(data, label2id, 0.1)
    print("---")
    # analysis.accuracy_of_accumulate_top_n_prob(data, label2id, high_prob, high_prob, 3)
    #analysis.accuracy_of_accumulate_some_n_larger_than_prob(data, label2id, high_prob, low_prob)
    #analysis.accuracy_of_accumulate_top_n_with_constraint(data, label2id, high_prob, low_prob, topN=2)

    print("Ambiguous Examples:")
    # analysis.accuracy_of_accumulate_ambiguity_prob(data, label2id, high_prob)
    analysis.accuracy_of_accumulate_top_n_prob(data, label2id, high_prob, ambig_prob, 5)


def run_ambiguous_analysis(file_name, label2id):
    label_num = len(label2id)
    high_prob = (label_num - 1) / label_num
    low_prob = 1 / label_num
    analysis = PartialAnnotationAcc(file_name, drop_NA=False)
    data = analysis.load_data()
    prob_threshold = 0.5
    analysis.accuracy_of_sigmoid_prob(data, label2id, prob_threshold)

    
    #analysis.accuracy_of_accumulate_ambiguity_prob(data, label2id, 0.95)



if __name__ == "__main__":
    # 先得到label2id， id2label这种
    dataset_name = "SemEval-2018-Task7"
    # dataset_name = "SemEval-2010-Task8"
    dataset = "semeval"
    dataset = "re-tacred"
    dataset_name = f"top10-{dataset}"
    label_file = f"/home/jjy/work/ST_RE/data/{dataset}_top10_label_excluding_NA/label2id.json"
    with open(label_file) as reader:
        label2id = json.load(reader)

    part_list = [100, 200, 400]
    part_list = [10]
    part = 800
    part = 10
    seeds = [0, 1, 2]
    # pseudo_file = f"/data/jjy/ST_RE_balanced/base/re-tacred/balanced_small_data_exp01/20/batch32_epoch30_fix_lr3e-05_seed0/pseudo_all.txt"
    pseudo_file = f"/data/jjy/ST_RE_micro/base/{dataset_name}/1.1_for_train_1.2_for_unlabel/batch32_epoch30_fix_lr3e-05_seed0/pseudo_all.txt"
    pseudo_file = f"/data/jjy/ST_RE_micro/base/{dataset_name}/small_data_exp01/0.1/batch32_epoch30_fix_lr3e-05_seed0/pseudo_all.txt"
    # pseudo_file = f"/data/jjy/ST_RE/base/{dataset_name}/small_data_exp01_at_least_one/200/batch32_epoch30_fix_lr3e-05_seed0/pseudo_all.txt"
    pseudo_file = f"/data/jjy/ST_RE_micro_small_data_0819_1000/base/SemEval-2010-Task8/small_data_exp01/batch32_epoch20_fix_lr5e-05_seed0/pseudo_all.txt"
    pseudo_file = f"/data/jjy/ST_RE_micro_small_data_easy_and_ambig_400/base/SemEval-2010-Task8/small_data_exp01/batch32_epoch20_fix_lr5e-05_seed0/pseudo_all.txt"
    for part in part_list:
        print(f"######################## {part} #####################")
        for seed in seeds:
            pseudo_file = f"/data/jjy/ST_RE_macro_accumulate_prob_{part}_all_new/base/{dataset_name}/balanced_small_data_exp01/batch32_epoch20_fix_lr5e-05_seed{seed}/pseudo_all.txt"
            run_partial_analysis(pseudo_file, label2id)
    # run_ambiguous_analysis(pseudo_file, label2id)
    exit()
    label_file = "/home/jjy/work/ST_RE/data/semeval/small_data_exp01_at_least_one/train_0-100.txt"
    analysis = SelfTrainingAnalysis(label_file)
    label_data = analysis.load_data()
    label_data = analysis.read_friendly_change_for_gold_data(label_data)
    with open("/home/jjy/work/ST_RE/src/exp_analysis/semeval_100_data.txt", 'w') as writer:
        for inst in label_data:
            writer.write(json.dumps(inst, indent=4) + "\n")
    pseudo_file = f"/data/jjy/ST_RE/semeval/small_data_exp01_at_least_one/100/batch32_epoch20_fix_lr5e-05_seed0/base_e1e2/pseudo_select_prob_0.0.txt"
    analysis = SelfTrainingAnalysis(pseudo_file)
    data = analysis.load_data()
    top_data, bottom_data = analysis.get_top_prob_examples(data, 1000)
    with open("/home/jjy/work/ST_RE/src/exp_analysis/semeval_top_data.txt", 'w') as writer:
        for inst in top_data:
            writer.write(json.dumps(inst, indent=4) + "\n")
    with open("/home/jjy/work/ST_RE/src/exp_analysis/semeval_bottom_data.txt", 'w') as writer:
        for inst in bottom_data:
            writer.write(json.dumps(inst, indent=4) + "\n")
    

    label_file = "/home/jjy/work/ST_RE/data/re-tacred/small_data_exp01_at_least_one/train_0-500.txt"
    analysis = SelfTrainingAnalysis(label_file)
    label_data = analysis.load_data()
    label_data = analysis.read_friendly_change_for_gold_data(label_data)
    with open("/home/jjy/work/ST_RE/src/exp_analysis/re_tacred_500_data.txt", 'w') as writer:
        for inst in label_data:
            writer.write(json.dumps(inst, indent=4) + "\n")
    exit()
    pseudo_file = f"/data/jjy/ST_RE/re-tacred/small_data_exp01_at_least_one/500/batch32_epoch20_fix_lr5e-05_seed0/base_e1e2/pseudo_select_prob_0.0.txt"
    analysis = SelfTrainingAnalysis(pseudo_file)
    data = analysis.load_data()
    top_data, bottom_data = analysis.get_top_prob_examples(data, 40000)
    with open("/home/jjy/work/ST_RE/src/exp_analysis/re_tacred_top_data.txt", 'w') as writer:
        for inst in top_data:
            writer.write(json.dumps(inst, indent=4) + "\n")
    with open("/home/jjy/work/ST_RE/src/exp_analysis/re_tacred_bottom_data.txt", 'w') as writer:
        for inst in bottom_data:
            writer.write(json.dumps(inst, indent=4) + "\n")
    exit()

    for part in part_list:
        for seed in seeds:
            print("---------")
            label_example_file = f"/home/jjy/work/ST_RE/data/semeval/small_data_exp01_at_least_one/train_0-{part}.txt"
            easy_example_file = f"/data/jjy/ST_RE/semeval/small_data_exp01_at_least_one/{part}/batch32_epoch20_fix_lr5e-05_seed{seed}/base_e1e2/pseudo_select_prob_0.9.txt"
            hard_example_file = f"/data/jjy/ST_RE/semeval/small_data_exp01_at_least_one/{part}/batch32_epoch20_fix_lr5e-05_seed{seed}/base_e1e2/pseudo_unused_prob_0.9.txt"
            analysis = SelfTrainingAnalysis(label_example_file)
            print("Labeled data")
            data = analysis.load_data()
            analysis.get_avg_entity_distance(data)
            analysis.get_avg_sen_length(data)
            analysis = SelfTrainingAnalysis(easy_example_file)
            print("Easy Example data")
            data = analysis.load_data()
            analysis.get_avg_entity_distance(data)
            analysis.get_avg_sen_length(data)
            analysis = SelfTrainingAnalysis(hard_example_file)
            print("Hard Example data")
            data = analysis.load_data()
            analysis.get_avg_entity_distance(data)
            analysis.get_avg_sen_length(data)
    exit()
    part_list = [250, 500, 1000]
    for part in part_list:
        print("---------")
        label_example_file = f"/home/jjy/work/ST_RE/data/re-tacred/small_data_exp01_at_least_one/train_0-{part}.txt"
        easy_example_file = f"/data/jjy/ST_RE/re-tacred/small_data_exp01_at_least_one/{part}/batch32_epoch20_fix_lr5e-05_seed0/base_e1e2/pseudo_select_prob_0.9.txt"
        hard_example_file = f"/data/jjy/ST_RE/re-tacred/small_data_exp01_at_least_one/{part}/batch32_epoch20_fix_lr5e-05_seed0/base_e1e2/pseudo_unused_prob_0.9.txt"
        analysis = SelfTrainingAnalysis(label_example_file)
        data = analysis.load_data()
        print("Labeled data")
        analysis.get_avg_entity_distance(data)
        analysis.get_avg_sen_length(data)
        analysis = SelfTrainingAnalysis(easy_example_file)
        print("Easy Example data")
        data = analysis.load_data()
        analysis.get_avg_entity_distance(data)
        analysis.get_avg_sen_length(data)
        analysis = SelfTrainingAnalysis(hard_example_file)
        print("Hard Example data")
        data = analysis.load_data()
        analysis.get_avg_entity_distance(data)
        analysis.get_avg_sen_length(data)