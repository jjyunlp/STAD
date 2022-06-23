import logging
import os
import json
from re import T
import numpy as np

"""
Collect Results

全新版本。
所有关系都用上，但每个关系还是只有10/20/40/80这样子。如果空缺，则有多少用多少。比如特别少的rel，也许只有个位数，那就个位数。
另外，NA也把它当作普通关系。
同时，为了更贴近low-resource，dev也每个关系只有10个。

Approaches
Teacher:
    supervised: 只用human-annotated data
Student:
    self-training: 只用human and confident data
    hard label: human + confident + non-confident
    soft label: human + confident + non-confident
    Partial+Negative: human + confident(positive) + non-confident(negative)

Meanwhile, we need to conduct ablation experiments:
Student:
    - partial (hard + negative)
    - negative (partial + positive)
    - both (hard + positive = hard label)
"""
no_improve_num = 5     # 总共就没多少
seeds = [0, 1, 2, 3, 4]
use_micro = True
use_macro = False
lr = 5e-5
batch_size = 32
num_train_epoch = 20
# global setting
cuda_num = 1

prob_thresholds = [0.95, 0.9, 0.85, 0.8, 0.7]
prob_thresholds = [0.9]
# 0.85 is best dev for semeval on self-training baseline
prob_threshold = 0.9

dataset_name = 're-tacred_exclude_NA'
#dataset_name = 'semeval'
# the top-k and top-p, the k is set as the maximum number which is the total number of relations minus one
# This means we can train the instance as long as one relation is not labeled as positive
if dataset_name == 'semeval':
    max_label_num = 18
else:
    max_label_num = 39

exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
part = 20   # each for class
small_data_name = f"low_resource_exp{exp_id}"
train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"




def collect_results(dataset_name, exp_base_dir, method, is_baseline=False, is_two_stage=False, is_two_stage_first=False, iter=0):
    """
    100, 200
    都是相似文件目录的
    method 是完整的method形式
    is_two_stage_first: only easy + ambig2 正好对比实验的结果
    """
    def get_base_results(result_file):
        with open(result_file) as reader:
            results = json.load(reader)
            if use_micro:
                val = results['val']['micro_f1']
                test = results['test']['micro_f1']
            elif use_macro:
                val = results['val']['macro_f1']
                test = results['test']['macro_f1']
            else:
                print("Error Evaluation!!")
                exit()
        return val, test

    val_list = []
    test_list = []
    for seed in seeds:
        result_file = os.path.join(
                                exp_base_dir,
                                method,
                                dataset_name,
                                small_data_name,
                                f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}")
        if is_baseline:
            result_file = os.path.join(result_file, "results")
        elif is_two_stage:
            result_file = os.path.join(result_file, f"epoch_{iter}", 'second', "results")
        elif is_two_stage_first:
            result_file = os.path.join(result_file, f"epoch_{iter}", 'first', "results")
        else:
            result_file = os.path.join(result_file, f"epoch_{iter}", "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
        # 也可以读取pseudo data的训练情况
    result_record_file = os.path.join(f"./{dataset_name}_{part}_results.txt")

    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + "\n")
        writer.write(f"Iteration: {iter}\n")
        if is_two_stage_first:
            writer.write("first train\n")
        else:
            writer.write("second train\n")
        writer.write(f"{val_list}: avg={round(sum(val_list)/len(val_list), 5)}\n")
        writer.write(f"{test_list}: avg={round(sum(test_list)/len(test_list), 5)}\n")
        writer.write("\n")

    # calculate standard deviation    
        
    if is_two_stage_first:
        print("two-stage first train:")
    print(method)
    print(val_list, round(sum(val_list)/len(val_list), 5), np.std(val_list))
    print(test_list, round(sum(test_list)/len(test_list), 5), np.std(test_list))



def do_self_training(prob_threshold):
    # Merge
    train_merge_confident_example = True
    if train_merge_confident_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_confident_example_prob{prob_threshold}"
        collect_results(dataset_name, exp_base_dir, method)

def do_self_training_hard_label(prob_threshold):
    # Merge confident and non-confident instances in hard label
    method = f"merge_all_example_prob{prob_threshold}_in_hard_label"
    collect_results(dataset_name, exp_base_dir, method)


def do_self_training_soft_label(prob_threshold):
    # Merge confident and non-confident instances in soft label
    # 我觉得，最好是non-confident instances 使用soft label，confident不变。代码不知道是否支持
    # 特别是label的设置，搞好的话很方便修改
    # 天生支持的，换下数据生成就行，我们现在都用onehot_label，只要一个是hard一个是soft就行
    method = f"merge_all_example_prob{prob_threshold}_but_ambig_in_soft_label"
    collect_results(dataset_name, exp_base_dir, method)

def do_merge_easy_and_ambig2_training_negative(max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    # 这个就是我们的，以及两个剥离实验
    one_random_loss = True
    # follow the Kim's negative training for partial label
    if one_random_loss:
        method = f"joint_merge_easy_ambig2_prob{ambig_prob_threshold}"
        collect_results(dataset_name, exp_base_dir, method)

    one_random_loss_hard_label = True
    # follow the Kim's negative training for hard label
    # - partial label
    if one_random_loss_hard_label:
        method = f"joint_merge_easy_ambig1_prob{ambig_prob_threshold}"
        collect_results(dataset_name, exp_base_dir, method)
    
    #### POSITIVE random one for partial
    # - negative
    one_random_positive_loss_partial_label = True
    # follow the Kim's negative training for hard label
    if one_random_positive_loss_partial_label:
        method = f"positive_merge_easy_ambig2_prob{ambig_prob_threshold}"
        collect_results(dataset_name, exp_base_dir, method)


if __name__ == "__main__":
    server = 'AI'
    if server == '139':
        exp_base_dir = "/data4/jjyunlp/rc_output/self_training_mixup"   # for 139
    if server == 'AI':
        if use_micro:
            exp_base_dir = f"/data/jjy/ST_RE_micro_accumulate_prob_{part}_low_resource"
        if use_macro:
            exp_base_dir = f"/data/jjy/ST_RE_macro_accumulate_prob_{part}_low_resource"

    test_file = f"../data/{dataset_name}/test.txt"
    label_file = f"../data/{dataset_name}/label2id.json"


    train_small_base = True
    if train_small_base:
        # Train baseline: only use gold data
        method = "base"
        collect_results(dataset_name, exp_base_dir, method, is_baseline=True)

    train_self_training = True
    if train_self_training:
        """
        The baseline for self-training which only merges confident instances.
        """
        for prob_threshold in prob_thresholds:
            # to find the best prob for self-training, which is our baseline
            # it seems should be conducted on val set
            do_self_training(prob_threshold)


    train_self_training_partial_negative_and_ablation = True
    if train_self_training_partial_negative_and_ablation:
        do_merge_easy_and_ambig2_training_negative(max_label_num, prob_threshold, prob_threshold)


    train_self_training_hard_label = True
    # 将confident和non-confident的句子都加上，都以hard label的形式，即top-1的预测作为答案。
    if train_self_training_hard_label:
        # 需要先确认最好的prob，然后进行以下实验
        do_self_training_hard_label(prob_threshold)

    train_self_training_soft_label = True
    # 将confident和non-confident的句子都加上，都以hard label的形式，即top-1的预测作为答案。
    if train_self_training_soft_label:
        do_self_training_soft_label(prob_threshold)