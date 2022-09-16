import logging
import os
import json
import numpy as np
from re import T
os.environ['MKL_THREADING_LAYER'] = 'GNU'   # For a mkl-service problem


"""
Re-tagging the test set with output prob distribution
And then, we evaluate on it.
"""
no_improve_num = 5     # 总共就没多少
use_micro = True
use_macro = False
lr = 5e-5
batch_size = 32
num_train_epoch = 20
# global setting
seeds = [1, 2, 3, 4, 5]
cuda_num = 1
only_collect_results = False     # Finally collect results from different seeds on different GPUs

# self-training confident best prob
self_training_prob_thresholds = [0.9, 0.95, 0.95, 0.8, 0.85]
# ours best prob
partial_negative_prob_thresholds = [0.9, 0.85, 0.85, 0.85, 0.85]
# 0.85 is best dev for semeval on self-training baseline

dataset_name = 're-tacred_exclude_NA'
# the top-k and top-p, the k is set as the maximum number which is the total number of relations minus one
# This means we can train the instance as long as one relation is not labeled as positive
max_label_num = 18

part = 20   # each for class


def build_data_by_seed(seed):
    # A random data split (for training set)
    # 把生成小语料的代码搬到这边，如果还没有生成，则实时生成，带上seed的文件夹名
    # 目前我不想跑了，不现实，实验太多了。
    return None


def run_base_tagging(seed, small_data_name, python_file, exp_base_dir, method="base", topN=2):
    print(f"Start to train teacher model with split={part}!")
    model_type = "bert"
    save_steps = 100        # 模型中有选择是每个epoch测试还是多少个step测试
    method_name = "baseline"
    bert_mode = 'e1e2'

    cache_feature_file_dir = os.path.join(
        exp_base_dir,
        method,
        dataset_name,
        small_data_name,
    )
    output_dir = os.path.join(
        cache_feature_file_dir,
        f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}"
        )
    run_cmd = (
                f"CUDA_VISIBLE_DEVICES={cuda_num} python -u {python_file} "
                f"--train_file={train_file} "
                f"--val_file={val_file} "
                f"--test_file={test_file} "
                f"--label_file={label_file} "
                f"--model_type={model_type} "
                f"--model_name_or_path={model_name_or_path} "
                f"--dataset_name={dataset_name} "
                f"--bert_mode={bert_mode} "
                f"--cache_feature_file_dir={cache_feature_file_dir}  "
                f"--output_dir={output_dir} "
                f"--num_train_epochs={num_train_epoch} "
                f"--topN_metric {topN} "
                f"--learning_rate={lr} --seed={seed} "
                f"--per_gpu_train_batch_size={batch_size} "
                f"--per_gpu_eval_batch_size={batch_size} "
                f"--do_train --do_eval --overwrite_output_dir "
                f"--eval_all_epoch_checkpoints "
                f"--do_lower_case  --method_name={method_name} "
                f"--save_steps={save_steps} --no_improve_num={no_improve_num} "
                # f"--use_lr_scheduler "  # 小数据不使用
    )
    if use_micro:
        run_cmd += "--micro_f1 "
    if use_macro:
        run_cmd += "--macro_f1 "
    # print(run_cmd)
    os.system(run_cmd)



def run_self_training_tagging(seed, small_data_name, python_file, exp_base_dir, method="base", topN=2):
    print(f"Start to train teacher model with split={part}!")
    model_type = "bert"
    save_steps = 100        # 模型中有选择是每个epoch测试还是多少个step测试
    method_name = "baseline"
    bert_mode = 'e1e2'

    cache_feature_file_dir = os.path.join(
        exp_base_dir,
        method,
        dataset_name,
        small_data_name,
    )
    output_dir = os.path.join(
        cache_feature_file_dir,
        f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}",
        "epoch_0",
        )
    run_cmd = (
                f"CUDA_VISIBLE_DEVICES={cuda_num} python -u {python_file} "
                f"--train_file={train_file} "
                f"--val_file={val_file} "
                f"--test_file={test_file} "
                f"--label_file={label_file} "
                f"--model_type={model_type} "
                f"--model_name_or_path={model_name_or_path} "
                f"--dataset_name={dataset_name} "
                f"--bert_mode={bert_mode} "
                f"--cache_feature_file_dir={cache_feature_file_dir}  "
                f"--output_dir={output_dir} "
                f"--num_train_epochs={num_train_epoch} "
                f"--topN_metric {topN} "
                f"--learning_rate={lr} --seed={seed} "
                f"--per_gpu_train_batch_size={batch_size} "
                f"--per_gpu_eval_batch_size={batch_size} "
                f"--do_train --do_eval --overwrite_output_dir "
                f"--eval_all_epoch_checkpoints "
                f"--do_lower_case  --method_name={method_name} "
                f"--save_steps={save_steps} --no_improve_num={no_improve_num} "
                # f"--use_lr_scheduler "  # 小数据不使用
    )
    if use_micro:
        run_cmd += "--micro_f1 "
    if use_macro:
        run_cmd += "--macro_f1 "
    # print(run_cmd)
    os.system(run_cmd)



def collect_base_topN_results(dataset_name, exp_base_dir, method, is_baseline=False, is_two_stage=False, is_two_stage_first=False, iter=0, topN=2):
    """
    collect base topN results among seeds
    """
    def get_base_results(result_file):
        with open(result_file) as reader:
            results = json.load(reader)
            if use_micro:
                test = results['micro_f1']
            elif use_macro:
                test = results['macro_f1']
            else:
                print("Error Evaluation!!")
                exit()
        return test

    val_list = []
    test_list = []
    for seed in seeds:
        small_data_name = f"low_resource_exp_{seed}"
        result_file = os.path.join(
                                exp_base_dir,
                                method,
                                dataset_name,
                                small_data_name,
                                f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}")
        if is_baseline:
            result_file = os.path.join(result_file, f"top{topN}_results.txt")
        elif is_two_stage:
            result_file = os.path.join(result_file, f"epoch_{iter}", 'second', "results")
        elif is_two_stage_first:
            result_file = os.path.join(result_file, f"epoch_{iter}", 'first', "results")
        else:
            result_file = os.path.join(result_file, f"epoch_{iter}", f"top{topN}_results.txt")

        test = get_base_results(result_file)
        test_list.append(test)
        # 也可以读取pseudo data的训练情况
    result_record_file = os.path.join(exp_base_dir, f"{dataset_name}_{part}_topN_results.txt")
    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + f"  Top-{topN}\n")
        writer.write(f"Seeds: {seeds}\tIteration: {iter}\n")
        writer.write(f"{test_list}: avg={round(sum(test_list)/len(test_list), 5)}, std={round(np.std(test_list)*100, 1)}\n")
        writer.write("\n")
        
    print(test_list, round(sum(test_list)/len(test_list), 5))


def collect_self_training_topN_results(prob_thresholds, dataset_name, exp_base_dir, method, is_baseline=False, is_two_stage=False, is_two_stage_first=False, iter=0, topN=2):
    """
    collect top N results for selected prob_thresholds and its seed.

    seeds and probs are paired

    prob_thresholds is different for self-training and ours, so, add the arg
    """
    def get_base_results(result_file):
        with open(result_file) as reader:
            results = json.load(reader)
            if use_micro:
                test = results['micro_f1']
            elif use_macro:
                test = results['macro_f1']
            else:
                print("Error Evaluation!!")
                exit()
        return test

    val_seed_prob = []  # seed: [p1_score, XX]
    test_seed_prob = []  # seed: [p1_score, XX]
    for seed, prob in zip(seeds, prob_thresholds):
        small_data_name = f"low_resource_exp_{seed}"
        val_prob = []
        test_prob = []
        method_p = f"{method}_{prob}"
        result_file = os.path.join(
                                exp_base_dir,
                                method_p,
                                dataset_name,
                                small_data_name,
                                f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}")
        if is_baseline:
            result_file = os.path.join(result_file, f"top{topN}_results.txt")
        else:
            # go here
            result_file = os.path.join(result_file, f"epoch_{iter}", f"top{topN}_results.txt")

        test = get_base_results(result_file)
        test_seed_prob.append(test)
    result_record_file = os.path.join(exp_base_dir, f"{dataset_name}_{part}_topN_results.txt")
    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + f"  Top-{topN}\n")
        # Write all results for seeds
        writer.write(f"Seeds: {seeds}\n" )
        writer.write(f"Probs: {prob_thresholds}\n")
        writer.write(f"Micro-F1(test): {test_seed_prob} Avg={round(sum(test_seed_prob)/len(test_seed_prob), 5)}, std={round(np.std(test_seed_prob)*100, 1)}\n")
        writer.write("\n")
    print(test_seed_prob)

if __name__ == "__main__":
    server = '139'
    if server == '139':
        exp_base_dir = "/data4/jjyunlp/rc_output/STAD_diff_data"   # for 139
        model_name_or_path = "/data3/jjyu/work/RelationClassification/BERT_BASE_DIR/bert-base-uncased"
    if server == 'AI':
        model_name_or_path = ""
        if use_micro:
            exp_base_dir = f"/data/jjy/ST_RE_micro_accumulate_prob_{part}_low_resource"
        if use_macro:
            exp_base_dir = f"/data/jjy/ST_RE_macro_accumulate_prob_{part}_low_resource"

    test_file = f"../data/{dataset_name}/test.txt"
    label_file = f"../data/{dataset_name}/label2id.json"
    python_file = "get_top_N_metric_analysis.py"
    topN = 5
    train_small_base = True
    if train_small_base:
        method = "base"
        print(method)
        for seed in seeds:
            print(seed)
            if only_collect_results:
                break
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            # Train baseline: only use gold data
            run_base_tagging(seed, small_data_name, python_file, exp_base_dir, method=method, topN=topN)
        # Results should be
        # Seed      1       2     3   Avg.  std.
        # Micro-F1  XX      XX    XX   YY   ZXZ
        collect_base_topN_results(dataset_name, exp_base_dir, method, is_baseline=True, topN=topN)

    train_self_training = True
    if train_self_training:
        """
        The baseline for self-training which only merges confident instances.
        """
        method = "self_training_confident"
        print(method)
        for seed, prob in zip(seeds, self_training_prob_thresholds):    # select the best prob for each seed
            if only_collect_results:
                break
            print(seed, prob)
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            method_p = f"{method}_{prob}"
            run_self_training_tagging(seed, small_data_name, python_file, exp_base_dir, method=method_p, topN=topN)
        # Search the best dev score among prob thresholds and report all results for seeds
        # Results should be
        # Seed      1       2     3   Avg.  std.
        # prob      0.95    XY    XX   YY   XYX
        # Micro-F1  XX      XX    XX   YY   YXY
        collect_self_training_topN_results(self_training_prob_thresholds, dataset_name, exp_base_dir, method, topN=topN)

    train_self_training_partial_negative_and_ablation = True
    if train_self_training_partial_negative_and_ablation:
        method = "joint_easg_partial_ambig"
        print(method)
        for seed, prob in zip(seeds, partial_negative_prob_thresholds):    # select the best prob for each seed
            if only_collect_results:
                break
            print(seed, prob)
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            method_p = f"{method}_{prob}"
            run_self_training_tagging(seed, small_data_name, python_file, exp_base_dir, method=method_p, topN=topN)
        collect_self_training_topN_results(partial_negative_prob_thresholds, dataset_name, exp_base_dir, method, topN=topN)
