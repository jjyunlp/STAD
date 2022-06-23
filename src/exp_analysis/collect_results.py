"""
Junjie Yu, 2021-04-24, Soochow University
收集各实验的数据，输出成markdown格式，并计算各个种子的平均值，以及方差啥的。
"""
import os
import json

base_dir = '/home/jjy/work/rc/exp_outputs/re-tacred'
batch_size = 32
seeds = [x for x in range(3)]
num_train_epoch = 30
lr = 3e-5
exp_id = '01'
small_data_name = f"small_data_exp{exp_id}"


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


def collect_small_data_base_results(dataset_name, exp_base_dir, part_list, method='base'):
    """
    exp_mode: base_{}
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    for part in part_list:
        val_list = []
        test_list = []
        print(f"Mehtod={method}, Part={part}")
        for seed in seeds:
            result_file = os.path.join(
                exp_base_dir,
                method,
                dataset_name,
                small_data_name,
                str(part),
                f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                "results")

            val, test = get_base_results(result_file)
            val_list.append(val)
            test_list.append(test)
        print(val_list, round(sum(val_list)/len(val_list), 5))
        print(test_list, round(sum(test_list)/len(test_list), 5))


def collect_one_self_training_base_merge_by_prob_threshold_results(dataset_name, exp_base_dir, part_list, prob_threshold, method="merge"):
    """
    100, 200
    one_self_training_base_merge
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    for part in part_list:
        print(f"Part={part}")
        val_list = []
        test_list = []
        for seed in seeds:
            result_file = os.path.join(
                                    exp_base_dir,
                                    method,
                                    dataset_name,
                                    small_data_name,
                                    str(part),
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


def collect_one_self_training_base_merge_by_prob_threshold_with_ambiguity_results(dataset_name, exp_base_dir, part_list, prob_threshold, label_num, method="merge_partial_v2"):
    """
    100, 200
    one_self_training_base_merge
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    method = f"{method}_{label_num}_{prob_threshold}"
    for part in part_list:
        print(f"Part={part}")
        val_list = []
        test_list = []
        for seed in seeds:
            result_file = os.path.join(
                                    exp_base_dir,
                                    method,
                                    dataset_name,
                                    small_data_name,
                                    str(part),
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


def collect_base_results(dataset_name, exp_base_dir, part_list, method):
    """
    100, 200
    都是相似文件目录的
    method 是完整的method形式
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    for part in part_list:
        print(f"Part={part}")
        val_list = []
        test_list = []
        for seed in seeds:
            result_file = os.path.join(
                                    exp_base_dir,
                                    method,
                                    dataset_name,
                                    small_data_name,
                                    str(part),
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


def collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, part_list, prob_threshold, lr1, lr2, method="two_stage"):
    """
    one_self_training_base_twice_fine_tune
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    for part in part_list:
        print(part)
        val_list = []
        test_list = []
        for seed in seeds:
            result_file = os.path.join(
                                    exp_base_dir,
                                    method,
                                    dataset_name,
                                    small_data_name,
                                    str(part),
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

## 以下不用 了
        
def collect_one_self_training_base_twice_fine_tune_with_consistency_training_results(dataset_name, exp_base_dir, part_list, u_size):
    """
    100, 200
    one_self_training_base_twice_fine_tune with consistency training
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    num_train_epoch = 20
    for part in part_list:
        print(f"Part={part}")
        for u in u_size:
            val_list = []
            test_list = []
            for seed in seeds:
                result_file = os.path.join(exp_base_dir,
                                        dataset_name,
                                        small_data_name,
                                        str(part),
                                        f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                        # f"base_e1e2",
                                        f"one_self_training_base_twice_fine_tune_consistency_training_{u}_pseudo_dropout_0.1",
                                        "human",
                                        "epoch_0",
                                        "results")

                val, test = get_base_results(result_file)
                val_list.append(val)
                test_list.append(test)
            print(u)
            print(val_list, round(sum(val_list)/len(val_list), 5))
            print(test_list, round(sum(test_list)/len(test_list), 5))
            print("---")
        

def collect_one_self_training_base_twice_fine_tune_diff_pseudo_acc_results(dataset_name, exp_base_dir, part_list, pseudo_list, u_size):
    """
    100
    one_self_training_base_merge
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    num_train_epoch = 20
    for part in part_list:
        print(f"Part={part}")
        for pseudo in pseudo_list:
            print(f"pseudo={pseudo}")
            for u in u_size:
                val_list = []
                test_list = []
                acc_list = []
                for seed in seeds:
                    result_file = os.path.join(exp_base_dir,
                                            dataset_name,
                                            small_data_name,
                                            str(part),
                                            f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                            # f"base_e1e2",
                                            f"one_self_training_base_twice_fine_tune_{u}_pseudo_data_from_{pseudo}",
                                            "human",
                                            "epoch_0",
                                            "results")

                    val, test = get_base_results(result_file)
                    val_list.append(val)
                    test_list.append(test)
                    pseudo_acc_file = os.path.join(exp_base_dir,
                                                   dataset_name,
                                                   small_data_name,
                                                   f"{pseudo}",
                                                   f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                                   f"base_e1e2",
                                                   f"pseudo_select_{u}_result.txt"
                    )
                    pseudo_acc = get_pseudo_acc(pseudo_acc_file)
                    acc_list.append(pseudo_acc)
                print(u)
                print("val:\t", val_list, round(sum(val_list)/len(val_list), 5))
                print("test:\t", test_list, round(sum(test_list)/len(test_list), 5))
                print("acc:\t", acc_list, round(sum(acc_list)/len(acc_list), 3))
                print("---")

def collect_one_self_training_base_twice_fine_tune_insert_noise_results(dataset_name, exp_base_dir, part_list, u_size, noise_ratio):
    """
    100, 200
    one_self_training_base_twice_fine_tune
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    num_train_epoch = 20
    for part in part_list:
        print(f"Part={part}")
        for u in u_size:
            val_list = []
            test_list = []
            for seed in seeds:
                result_file = os.path.join(exp_base_dir,
                                        dataset_name,
                                        small_data_name,
                                        str(part),
                                        f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                        # f"base_e1e2",
                                        f"one_self_training_base_twice_fine_tune_{u}_insert_noise_{noise_ratio}",
                                        "human",
                                        "epoch_0",
                                        "results")

                val, test = get_base_results(result_file)
                val_list.append(val)
                test_list.append(test)
            print(u)
            print(val_list, round(sum(val_list)/len(val_list), 5))
            print(test_list, round(sum(test_list)/len(test_list), 5))
            print("---")

def collect_one_self_training_base_twice_fine_tune_two_dropout_results(dataset_name, exp_base_dir, part_list, u_size, dropout_prob):
    """
    100, 200
    use different prob dropout for pseudo data finetune.
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    num_train_epoch = 20
    for part in part_list:
        print(f"Part={part}")
        for u in u_size:
            val_list = []
            test_list = []
            for seed in seeds:
                result_file = os.path.join(exp_base_dir,
                                        dataset_name,
                                        small_data_name,
                                        str(part),
                                        f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                        # f"base_e1e2",
                                        f"one_self_training_base_twice_fine_tune_{u}_pseudo_dropout_{dropout_prob}",
                                        "human",
                                        "epoch_0",
                                        "results")

                val, test = get_base_results(result_file)
                val_list.append(val)
                test_list.append(test)
            print(u)
            print(val_list, round(sum(val_list)/len(val_list), 5))
            print(test_list, round(sum(test_list)/len(test_list), 5))
            print("---")



def collect_iterative_self_training_base_merge_results(dataset_name, exp_base_dir, part_list, self_train_epoch):
    """
    100, 200
    iterative_self_training_base_merge
    """
    batch_size = 32
    lr = 5e-5
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    num_train_epoch = 20
    for part in part_list:
        print(f"Part={part}")
        u = part * 10
        val_list = []
        test_list = []
        iter_epoch = int(u/part - 1)    # start from 0
        for seed in seeds:
            result_file = os.path.join(exp_base_dir,
                                    dataset_name,
                                    small_data_name,
                                    str(part),
                                    f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                    # f"base_e1e2",
                                    f"iterative_self_training_base_merge_all_unlabel_{self_train_epoch}_epoch",
                                    f"epoch_{self_train_epoch-1}",
                                    "results")

            val, test = get_base_results(result_file)
            val_list.append(val)
            test_list.append(test)
        print(u)
        print(val_list, round(sum(val_list)/len(val_list), 5))
        print(test_list, round(sum(test_list)/len(test_list), 5))
        print("---")


def collect_iterative_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, part_list, u_size):
    """
    100, 200
    one_self_training_base_twice_fine_tune
    """
    exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
    num_train_epoch = 20
    for part in part_list:
        print(f"Part={part}")
        for u in u_size:
            val_list = []
            test_list = []
            iter_epoch = int(u/part - 1)    # start from 0
            for seed in seeds:
                result_file = os.path.join(exp_base_dir,
                                        dataset_name,
                                        small_data_name,
                                        str(part),
                                        f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
                                        # f"base_e1e2",
                                        f"iterative_self_training_base_twice_fine_tune",
                                        "human",
                                        f"epoch_{iter_epoch}",
                                        "results")

                val, test = get_base_results(result_file)
                val_list.append(val)
                test_list.append(test)
            print(u)
            print(val_list, round(sum(val_list)/len(val_list), 5))
            print(test_list, round(sum(test_list)/len(test_list), 5))
            print("---")
    

if __name__ == "__main__":
    # 设置很多参数，输出统一的结果
    print("Start to collect results")
    dataset_name = 're-tacred'
    dataset_name = 'SemEval-2010-Task8'
    exp_base_dir = "/data4/jjyunlp/rc_output/self_training_mixup"
    exp_base_dir = "/data/jjy/ST_RE_micro"
    # collect_small_data_self_training_results(dataset_name, exp_base_dir)
    prob_thresholds = [0.9, 0.8, 0.7, 0.0]
    part_list = [1000]
    if dataset_name == "semeval":
        part_list = [200]
    part_list = [0.1]
    max_label_num = 2

    if True:
        # small data baseline
        method = "base"
        collect_small_data_base_results(dataset_name, exp_base_dir, part_list, method)

    if True:
        prob_threshold = 0.95
        print(f"merge_prob={prob_threshold}")
        method = f"merge_easy_example_prob{prob_threshold}"
        # collect_one_self_training_base_merge_by_prob_threshold_results(dataset_name, exp_base_dir, part_list, prob_threshold, method=method)
        collect_base_results(dataset_name, exp_base_dir, part_list, method=method)
    
    
    if True:
        prob_thresholds = [0.95]
        for prob_threshold in prob_thresholds:
            print(f"merge_prob={prob_threshold}, label_num={max_label_num} "),
            method = f"merge_easy_and_hard_by_accumulate_prob{prob_threshold}_top{max_label_num}"
            print(method)
            collect_base_results(dataset_name, exp_base_dir, part_list, method)

    if True:
        prob_thresholds = [0.95]
        for prob_threshold in prob_thresholds:
            print(f"merge_prob={prob_threshold}, label_num={max_label_num} "),
            method = f"merge_easy_and_ambiguity_hard_by_accumulate_prob{prob_threshold}_top{max_label_num}"
            print(method)
            collect_base_results(dataset_name, exp_base_dir, part_list, method)
                
    if True:
        print("Two-Stage Easy -> Gold")
        prob_thresholds = [0.95]
        lr1_list = [3e-5]
        lr2 = 3e-5
        prob_threshold = 0.95
        for prob_threshold in prob_thresholds:
            for lr1 in lr1_list:
                #method = f"two_stage_easy_and_hard_to_gold_prob{prob_threshold}_top{max_label_num}_lr1_{lr1}_lr2_{lr2}"
                # method = f"two_stage_hard_to_easy_and_gold_prob{prob_threshold}_top{max_label_num}_lr1_{lr1}_lr2_{lr2}"
                method = f"two_stage_easy_to_gold_prob{prob_threshold}_lr1_{lr1}_lr2_{lr2}"
                # method = f"two_stage_easy_and_ambiguity_hard_to_gold_prob{prob_threshold}_top{max_label_num}_lr1_{lr1}_lr2_{lr2}"
                print(method)
                print(lr1)
                collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, part_list, prob_threshold, lr1, lr2, method)

    if True:
        print("Two-Stage Easy+Hard -> Gold")
        prob_thresholds = [0.95]
        lr1_list = [3e-5]
        lr2 = 3e-5
        prob_threshold = 0.95
        for prob_threshold in prob_thresholds:
            for lr1 in lr1_list:
                print(method)
                print(lr1)
                method = f"two_stage_easy_and_hard_to_gold_prob{prob_threshold}_top{max_label_num}_lr1_{lr1}_lr2_{lr2}"
                collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, part_list, prob_threshold, lr1, lr2, method)
        
    if True:
        print("Two-Stage Easy and Ambig Hard -> Gold")
        prob_thresholds = [0.95]
        lr1_list = [3e-5]
        lr2 = 3e-5
        prob_threshold = 0.95
        for prob_threshold in prob_thresholds:
            for lr1 in lr1_list:
                method = f"two_stage_easy_and_ambiguity_hard_to_gold_prob{prob_threshold}_top{max_label_num}_lr1_{lr1}_lr2_{lr2}"
                print(method)
                print(lr1)
                collect_one_self_training_base_twice_fine_tune_results(dataset_name, exp_base_dir, part_list, prob_threshold, lr1, lr2, method)

    


