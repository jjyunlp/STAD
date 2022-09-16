import logging
import os
import json
import numpy as np
from re import T
os.environ['MKL_THREADING_LAYER'] = 'GNU'   # For a mkl-service problem


"""
训练基础模型的脚本

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
We set seed to [1, 2, 3], and each seed is also applied to select training data.
For prob threshold, we set [0.95, 0.9, 0.85, 0.8], it is selected by the best dev results.
Finally, we collect results from all seeds and report the average score and the std.
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
only_collect_results = True     # Finally collect results from different seeds on different GPUs

prob_thresholds = [0.95, 0.9, 0.85, 0.8]
# 0.85 is best dev for semeval on self-training baseline

dataset_name = 'semeval'
# the top-k and top-p, the k is set as the maximum number which is the total number of relations minus one
# This means we can train the instance as long as one relation is not labeled as positive
max_label_num = 18

part = 20   # each for class


def build_data_by_seed(seed):
    # A random data split (for training set)
    # 把生成小语料的代码搬到这边，如果还没有生成，则实时生成，带上seed的文件夹名
    # 目前我不想跑了，不现实，实验太多了。
    return None


def run_teacher(seed, small_data_name, python_file, exp_base_dir, method="base"):
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
    print(run_cmd)
    os.system(run_cmd)



def run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold,
                                              ambig_prob_threshold, 
                                              method=None, 
                                              iteration=1,
                                              max_label_num=5):
    # dataset
    model_type = "bert"
    save_steps = 100
    
    use_random_subset = False
    for seed in seeds:
        # for num_train_epoch in num_train_epochs:
        base_model_dir = os.path.join(
            exp_base_dir,
            'base',
            f"{dataset_name}",
            f"{small_data_name}",
            f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}"
        )
        cache_feature_file_dir = os.path.join(
            exp_base_dir,
            method,
            f"{dataset_name}",
            f"{small_data_name}",
        )
        output_dir = os.path.join(
            cache_feature_file_dir,
            f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}"
            )
        run_cmd = (
                    f"CUDA_VISIBLE_DEVICES={cuda_num} "
                    f"python -u {python_file} "
                    f"--easy_prob_threshold={easy_prob_threshold} "
                    f"--ambig_prob_threshold={ambig_prob_threshold} "
                    f"--max_label_num={max_label_num} "
                    f"--base_model_dir={base_model_dir} "
                    f"--train_file={train_file} "
                    f"--val_file={val_file} "
                    f"--test_file={test_file} "
                    f"--label_file={label_file} "
                    f"--unlabel_file={unlabel_file} "
                    f"--model_type={model_type} "
                    f"--model_name_or_path={model_name_or_path} "
                    f"--dataset_name={dataset_name} "
                    f"--cache_feature_file_dir={cache_feature_file_dir}  "
                    f"--output_dir={output_dir} "
                    f"--delete_model "      # 想办法只保留最新的模型，但是。。这样还需要保留最好的模型
                    f"--num_train_epochs={num_train_epoch} "
                    f"--self_train_epoch={iteration} "
                    f"--learning_rate={lr_1} "
                    f"--learning_rate_2={lr_2} "
                    f"--seed={seed} "
                    f"--logging_step={100} "
                    f"--per_gpu_train_batch_size={batch_size} "
                    f"--per_gpu_eval_batch_size={batch_size} "
                    f"--eval_all_epoch_checkpoints "
                    f"--do_lower_case  "
                    f"--save_steps={save_steps} "
                    f"--no_improve_num={no_improve_num} "
                    f"--overwrite_output "
                    f"--do_train "
        )
        if dataset_name == 're-tacred':
            run_cmd += "--use_random_subset "
            #run_cmd += "--drop_NA "
            #run_cmd += "--overwrite_cache "
        if use_micro:
            run_cmd += "--micro_f1 "
        if use_macro:
            run_cmd += "--macro_f1 "
        print(run_cmd)
        os.system(run_cmd)


def run_one_self_training_base_merge_pseudo_data(seed, python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method, small_data_name, max_label_num, alpha=None, iteration=1):
    """[summary]

    Args:
        python_file ([type]): [description]
        exp_base_dir ([type]): [description]
        prob_threshold ([type]): [description]
        method ([type]): [description]
        alpha ([type], optional): loss1 + alpha * loss2. Defaults to None.
    """
    # dataset
    model_type = "bert"
    save_steps = 100
    use_random_subset = False

    # for num_train_epoch in num_train_epochs:
    # base_model_dir: the teacher model
    base_model_dir = os.path.join(
        exp_base_dir,
        'base',
        f"{dataset_name}",
        f"{small_data_name}",
        f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}"
    )
    cache_feature_file_dir = os.path.join(
        exp_base_dir,
        method,
        f"{dataset_name}",
        f"{small_data_name}",
    )
    output_dir = os.path.join(
        cache_feature_file_dir,
        f"batch32_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}"
        )
    run_cmd = (
                f"CUDA_VISIBLE_DEVICES={cuda_num} "
                f"python -u {python_file} "
                f"--easy_prob_threshold={easy_prob_threshold} "
                f"--ambig_prob_threshold={ambig_prob_threshold} "
                f"--max_label_num={max_label_num} "
                f"--base_model_dir={base_model_dir} "
                f"--train_file={train_file} "
                f"--val_file={val_file} "
                f"--test_file={test_file} "
                f"--label_file={label_file} "
                f"--unlabel_file={unlabel_file} "
                f"--model_type={model_type} "
                f"--model_name_or_path={model_name_or_path} "
                f"--dataset_name={dataset_name} "
                f"--cache_feature_file_dir={cache_feature_file_dir}  "
                f"--output_dir={output_dir} "
                f"--num_train_epochs={num_train_epoch} "
                f"--self_train_epoch={iteration} "
                f"--learning_rate={lr} "
                f"--seed={seed} "
                f"--logging_step={100} "
                f"--per_gpu_train_batch_size={batch_size} "
                f"--per_gpu_eval_batch_size={batch_size} "
                f"--eval_all_epoch_checkpoints "
                f"--do_lower_case  "
                f"--save_steps={save_steps} "
                f"--no_improve_num={no_improve_num} "
                f"--overwrite_output "
                f"--do_train "
                # f"--overwrite_cache "
    )
    if use_micro:
        run_cmd += "--micro_f1 "
    if use_macro:
        run_cmd += "--macro_f1 "
    if '_min_' in method:
        run_cmd += "--ambiguity_mode='min'"
    elif '_max_' in method:
        run_cmd += "--ambiguity_mode='max'"
    
    if 'rampup' in method:
        run_cmd += "--rampup"
    if alpha is not None:
        run_cmd += f"--alpha={alpha} "
    
    print(run_cmd)
    os.system(run_cmd)



def collect_base_results(dataset_name, exp_base_dir, method, is_baseline=False, is_two_stage=False, is_two_stage_first=False, iter=0):
    """
    collect base results
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
        small_data_name = f"low_resource_exp_{seed}"
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
    result_record_file = os.path.join(exp_base_dir, f"{dataset_name}_{part}_results.txt")
    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + "\n")
        writer.write(f"Seeds: {seeds}\tIteration: {iter}\n")
        writer.write(f"{val_list}: avg={round(sum(val_list)/len(val_list), 5)}, std={round(np.std(val_list)*100, 1)}\n")
        writer.write(f"{test_list}: avg={round(sum(test_list)/len(test_list), 5)}, std={round(np.std(test_list)*100, 1)}\n")
        writer.write("\n")
        
    if is_two_stage_first:
        print("two-stage first train:")
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))


def collect_results_among_probs(dataset_name, exp_base_dir, method, is_baseline=False, is_two_stage=False, is_two_stage_first=False, iter=0):
    """
    collect results among prob_thresholds for each seed in seeds.
    print all results on different probs for each seed first
    The format should be
    Seed: XX
    Prob:             0.95 0.9 0.85 0.8
    Micro-F1(val):      X   X    X   X
    Micro-F1(test):     X   X    X   X
    Then print/log the best val and its test score among probs for each seed
    Seed            1       2     3   Avg.  std.
    prob          0.95     XY    XX   YY   XYX
    Micro-F1(val)  XX      XX    XX   YY   YXY
    Micro-F1(test) XX      XX    XX   YY   YXY
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

    val_seed_prob = []  # seed: [p1_score, XX]
    test_seed_prob = []  # seed: [p1_score, XX]
    for seed in seeds:
        small_data_name = f"low_resource_exp_{seed}"
        val_prob = []
        test_prob = []
        for prob in prob_thresholds:
            method_p = f"{method}_{prob}"
            result_file = os.path.join(
                                    exp_base_dir,
                                    method_p,
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
                # go here
                result_file = os.path.join(result_file, f"epoch_{iter}", "results")

            val, test = get_base_results(result_file)
            val_prob.append(val)
            test_prob.append(test)
        val_seed_prob.append(val_prob)
        test_seed_prob.append(test_prob)
    result_record_file = os.path.join(exp_base_dir, f"{dataset_name}_{part}_results.txt")
    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + "\n")
        for i, seed in enumerate(seeds):
            writer.write(f"Seed: {seed}\tIteration: {iter}\n")
            writer.write(f"Prob:\t{prob_thresholds}\n")
            writer.write(f"{val_seed_prob[i]}\n")
            writer.write(f"{test_seed_prob[i]}\n")
            writer.write("\n")
        # Write selected results
        best_val_among_probs = []   # select the best result on val among probs
        best_probs = []
        test_among_probs = []
        for val_probs, test_probs in zip(val_seed_prob, test_seed_prob):
            best_val = max(val_probs)   # best score among probs
            best_index = val_probs.index(best_val)
            best_prob = prob_thresholds[best_index]
            final_test = test_probs[best_index]

            best_val_among_probs.append(best_val)
            best_probs.append(best_prob)
            test_among_probs.append(final_test)
        # Write all results for seeds
        writer.write(f"Seeds: {seeds}\n" )
        writer.write(f"Probs: {best_probs}\n")
        writer.write(f"Micro-F1(val): {best_val_among_probs} Avg={round(sum(best_val_among_probs)/len(best_val_among_probs), 5)}, std={round(np.std(best_val_among_probs)*100, 1)}\n")
        writer.write(f"Micro-F1(test): {test_among_probs} Avg={round(sum(test_among_probs)/len(test_among_probs), 5)}, std={round(np.std(test_among_probs)*100, 1)}\n")

    print(val_seed_prob)
    print(test_seed_prob)
    print(best_val_among_probs)
    print(test_among_probs)


def do_self_training(seed, prob_threshold, method, small_data_name):
    # merge single label hard examples
    # 获取一个最高的val对应的prob_threshold
    python_file = "train_one_self_training_merge_easy_example.py"
    run_one_self_training_base_merge_pseudo_data(seed, python_file, exp_base_dir, prob_threshold, prob_threshold, method, small_data_name, max_label_num)
    # collect_results(dataset_name, exp_base_dir, method)

def do_self_training_hard_label(seed, prob_threshold, method):
    # Merge confident and non-confident instances in hard label
    python_file = "train_one_self_training_merge_easy_and_hard_example.py"
    run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num)
    # collect_results(dataset_name, exp_base_dir, method)


def do_self_training_soft_label(prob_threshold):
    # Merge confident and non-confident instances in soft label
    # 我觉得，最好是non-confident instances 使用soft label，confident不变。代码不知道是否支持
    # 特别是label的设置，搞好的话很方便修改
    # 天生支持的，换下数据生成就行，我们现在都用onehot_label，只要一个是hard一个是soft就行
    method = f"merge_all_example_prob{prob_threshold}_but_ambig_in_soft_label"
    python_file = "train_one_self_training_merge_hard_confident_and_soft_ambig.py"
    run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num)
    # collect_results(dataset_name, exp_base_dir, method)

def do_merge_easy_and_ambig2_training_negative(seed, method, max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    # 这个就是我们的，以及两个剥离实验
    one_random_loss = True
    # follow the Kim's negative training for partial label
    if one_random_loss:
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_negative_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(seed, python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, small_data_name=small_data_name, max_label_num=max_label_num)
        # collect_results(dataset_name, exp_base_dir, method)

    one_random_loss_hard_label = False
    # follow the Kim's negative training for hard label
    # - partial label
    if one_random_loss_hard_label:
        method = f"joint_merge_easy_ambig1_prob{ambig_prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig1_example_by_accumulate_prob_negative_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        # collect_results(dataset_name, exp_base_dir, method)
    
    #### POSITIVE random one for partial
    # - negative
    one_random_positive_loss_partial_label = False
    # follow the Kim's negative training for hard label
    if one_random_positive_loss_partial_label:
        method = f"positive_merge_easy_ambig2_prob{ambig_prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        #collect_results(dataset_name, exp_base_dir, method)


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
    train_small_base = True
    if train_small_base:
        python_file = "train_baseline.py"
        method = "base"
        for seed in seeds:
            if only_collect_results:
                break
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            # Train baseline: only use gold data
            run_teacher(seed, small_data_name, python_file, exp_base_dir, method=method)
        # Results should be
        # Seed      1       2     3   Avg.  std.
        # Micro-F1  XX      XX    XX   YY   ZXZ
        collect_base_results(dataset_name, exp_base_dir, method, is_baseline=True)

    train_self_training = True
    if train_self_training:
        """
        The baseline for self-training which only merges confident instances.
        """
        method = "self_training_confident"
        for seed in seeds:
            if only_collect_results:
                break
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            for prob_threshold in prob_thresholds:
                # to find the best prob for self-training, which is our self-training baseline
                method_p = f"{method}_{prob_threshold}"
                do_self_training(seed, prob_threshold, method_p, small_data_name)
        # Search the best dev score among prob thresholds and report all results for seeds
        # Results should be
        # Seed      1       2     3   Avg.  std.
        # prob      0.95    XY    XX   YY   XYX
        # Micro-F1  XX      XX    XX   YY   YXY
        collect_results_among_probs(dataset_name, exp_base_dir, method)

    train_self_training_partial_negative_and_ablation = True
    if train_self_training_partial_negative_and_ablation:
        method = "joint_easg_partial_ambig"
        for seed in seeds:
            if only_collect_results:
                break
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            for prob_threshold in prob_thresholds:
                method_p = f"{method}_{prob_threshold}"
                do_merge_easy_and_ambig2_training_negative(seed, method_p, max_label_num, prob_threshold, prob_threshold)
        collect_results_among_probs(dataset_name, exp_base_dir, method)


    train_self_training_hard_label = False
    # 将confident和non-confident的句子都加上，都以hard label的形式，即top-1的预测作为答案。
    if train_self_training_hard_label:
        method = f"self_training_hard_label"
        for seed in seeds:
            if only_collect_results:
                break
            # 需要先确认最好的prob，然后进行以下实验
            small_data_name = f"low_resource_exp_{seed}"
            train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
            val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
            unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"
            for prob_threshold in prob_thresholds:
                method_p = f"{method}_{prob_threshold}"
                do_self_training_hard_label(seed, prob_threshold, method_p)

    train_self_training_soft_label = False
    # 将confident和non-confident的句子都加上，都以hard label的形式，即top-1的预测作为答案。
    if train_self_training_soft_label:
        for seed in seeds:
            if only_collect_results:
                break
            do_self_training_soft_label(prob_threshold)