"""
Junjie Yu, 2021-04-24, Soochow University
收集各实验的数据，输出成markdown格式，并计算各个种子的平均值，以及方差啥的。
"""
import os
import json

base_dir = '/home/jjy/work/rc/exp_outputs/re-tacred'
batch_size = 32
seeds = [x for x in range(3)]
num_train_epoch = 20
lr = 5e-5

part = 'whole'
exp_id = 'whole'   # 后续会跑多种小样本，即多次随机切分

part = 400
exp_id = '01'
small_data_name = f"small_data_exp{exp_id}"
dataset_name = "SemEval-2018-Task7"


def get_base_results(result_file):
    with open(result_file) as reader:
        results = json.load(reader)
        val = results['val']['micro_f1']
        test = results['test']['micro_f1']
        #val = results['val']['macro_f1']
        #test = results['test']['macro_f1']
    return val, test

def get_pseudo_acc(result_file):
    with open(result_file) as reader:
        results = json.load(reader)
        acc = round(results['acc'], 5)
    return acc


def collect_small_data_base_results(dataset_name, exp_base_dir, method='base'):
    """
    exp_mode: base_{}
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    val_list = []
    test_list = []
    for seed in seeds:
        result_file = os.path.join(
            exp_base_dir,
            method,
            dataset_name,
            small_data_name,
            f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
            "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))


def collect_one_self_training_base_merge_by_prob_threshold_results(dataset_name, exp_base_dir, prob_threshold, method="merge"):
    """
    100, 200
    one_self_training_base_merge
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    val_list = []
    test_list = []
    for seed in seeds:
        result_file = os.path.join(
                                exp_base_dir,
                                method,
                                dataset_name,
                                small_data_name,
                                f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                "epoch_0",
                                "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
        # 也可以读取pseudo data的训练情况
        
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))
    print("---")


def collect_one_self_training_base_merge_by_prob_threshold_with_ambiguity_results(dataset_name, exp_base_dir, prob_threshold, label_num, method="merge_partial_v2"):
    """
    100, 200
    one_self_training_base_merge
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    method = f"{method}_{label_num}_{prob_threshold}"
    val_list = []
    test_list = []
    for seed in seeds:
        result_file = os.path.join(
                                exp_base_dir,
                                method,
                                dataset_name,
                                small_data_name,
                                f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                "epoch_0",
                                "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
        # 也可以读取pseudo data的训练情况
        
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))
    print("---")


def collect_base_results(dataset_name, exp_base_dir, method):
    """
    100, 200
    都是相似文件目录的
    method 是完整的method形式
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    val_list = []
    test_list = []
    for seed in seeds:
        result_file = os.path.join(
                                exp_base_dir,
                                method,
                                dataset_name,
                                small_data_name,
                                f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                "epoch_0",
                                "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
        # 也可以读取pseudo data的训练情况
        
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))
    print("---")


def collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, lr1, lr2, method="two_stage"):
    """
    one_self_training_base_twice_fine_tune
    """
    val_list = []
    test_list = []
    for seed in seeds:
        result_file = os.path.join(
                                exp_base_dir,
                                method,
                                dataset_name,
                                small_data_name,
                                f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                "epoch_0",
                                "second",
                                "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))
    print("---")


if __name__ == "__main__":
    # 设置很多参数，输出统一的结果
    print("Start to collect results")
    dataset_name = 're-tacred'
    #dataset_name = 'SemEval-2010-Task8'
    dataset_name = 'SemEval-2018-Task7'
    exp_base_dir = f"/data/jjy/ST_RE_micro_small_data_easy_and_ambig_400"

    exp_base_dir = "/data/jjy/ST_RE_micro_accumulate_prob_400"
    # collect_small_data_self_training_results(dataset_name, exp_base_dir)
    no_cycle = True
    use_two_losses = False
    use_softmin = False

    part = 400
    exp_base_dir = f"/data/jjy/ST_RE_micro_accumulate_prob_{part}"
    prob_threshold = 0.9
    max_label_nums = [2]

    if True:
        # small data baseline
        method = "base"
        collect_small_data_base_results(dataset_name, exp_base_dir, method)

    
    if True:
        # small data baseline
        method = "base"
        collect_small_data_base_results(dataset_name, exp_base_dir, method)

    if True:
        method = f"merge_easy_example_prob{prob_threshold}"
        print(method)
        collect_base_results(dataset_name, exp_base_dir, method=method)
    
    
    if True:
        for max_label_num in max_label_nums:
            method = f"merge_easy_and_hard_example_prob{prob_threshold}_top{max_label_num}"
            print(method)
            collect_base_results(dataset_name, exp_base_dir, method)

    if True:
        for max_label_num in max_label_nums:
            method = f"merge_easy_and_ambig_hard_example_prob{prob_threshold}_top{max_label_num}"
            print(method)
            collect_base_results(dataset_name, exp_base_dir, method)
    
    lr2 = 5e-5
    lr1 = 5e-5
    lr1_list = [5e-5]
    if True:
        print("Two-Stage Easy -> Gold")
        method = f"two_stage_easy_to_gold_lr1_{lr1}_lr2_{lr2}_prob{prob_threshold}"
        print(method)
        print(lr1)
        collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, lr1, lr2, method)

    if True:
        print("Two-Stage Easy+Hard -> Gold")
        for max_label_num in max_label_nums:
            method = f"two_stage_easy_and_hard_to_gold_lr1_{lr1}_lr2_{lr2}_prob{prob_threshold}_top{max_label_num}"
            print(method)
            collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, lr1, lr2, method)
        
    if True:
        print("Two-Stage Easy and Ambig Hard -> Gold")
        for max_label_num in max_label_nums:
            method = f"two_stage_easy_and_ambig_hard_to_gold_lr1_{lr1}_lr2_{lr2}_prob{prob_threshold}_top{max_label_num}"
            print(method)
            collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, lr1, lr2, method)

    





