"""
将两个概率比较接近的两个关系的句子输出，最好是关系也是容易混淆的两个。


component-whole
product-producer
我觉得这两个概率高一点。
"""
import json
import os
import sys
sys.path.insert(1, "/home/jjy/work/ST_RE/src")
from data_processor.data_loader_and_dumper import JsonDataDumper, JsonDataLoader


label2id_file = "/home/jjy/work/ST_RE/data/SemEval-2010-Task8/corpus/label2id.json"
with open(label2id_file) as reader:
    label2id = json.load(reader)
id2label = {}
for label, id in label2id.items():
    id2label[id] = label


input_file = "/data/jjy/ST_RE_micro/base/SemEval-2010-Task8/small_data_exp01/0.1/batch32_epoch30_fix_lr3e-05_seed0/pseudo_hard_example_accumulate_prob_0.99_top_9.txt"

loader = JsonDataLoader(input_file)
data = loader.load_all_raw_instance()


cases = []
for inst in data:
    distri = inst['distri']
    label2prob = {}
    for i, prob in enumerate(distri):
        label2prob[i] = prob
    
    label_and_prob_list = sorted(label2prob.items(), key=lambda item:item[1], reverse=True)
    first_label, first_prob = label_and_prob_list[0]
    second_label, second_prob = label_and_prob_list[1]
    third_label, third_prob = label_and_prob_list[1]
    cond_1 = first_prob > 0.3 # and second_prob > 0.2
    cond_2 = id2label[first_label] != inst['gold_relation']
    cond_3 = first_label != 0 and second_label != 0
    if cond_1 and cond_2 and cond_3:
        print(inst)
        case = {}
        case['id'] = inst['id']
        case['token'] = " ".join(inst['token'])
        case['h'] = inst['h']
        case['t'] = inst['t']
        case['relation'] = inst['gold_relation']
        case['fisrt_pred'] = id2label[first_label]
        case['fisrt_pred_prob'] = first_prob
        case['second_pred'] = id2label[second_label]
        case['second_pred_prob'] = second_prob
        cases.append(case)
    
case_file = "/home/jjy/work/ST_RE/src/exp_analysis/cases.txt"
dumper = JsonDataDumper(case_file, overwrite=True)
dumper.dump_all_instance(cases, indent=4)

