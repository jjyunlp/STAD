import logging
import os
import json
import numpy as np
from re import T

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
"""
no_improve_num = 5     # 总共就没多少
seeds = [0, 1, 2, 3, 4]
use_micro = True
use_macro = False
lr = 5e-5
batch_size = 32
num_train_epoch = 20
# global setting
cuda_num = 5

prob_thresholds = [0.95, 0.9, 0.85, 0.8, 0.7]
prob_thresholds = [0.95, 0.9]
# 0.85 is best dev for semeval on self-training baseline
prob_threshold = 0.95

dataset_name = 're-tacred'
dataset_name = 'semeval'
# the top-k and top-p, the k is set as the maximum number which is the total number of relations minus one
# This means we can train the instance as long as one relation is not labeled as positive
if dataset_name == 'semeval':
    max_label_num = 19
else:
    max_label_num = 39

exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
data_split_seed = seed
part = 15   # each for class
small_data_name = f"low_resource_exp{exp_id}"
train_file = f"../data/{dataset_name}/{small_data_name}/train_{part}.txt"
val_file = f"../data/{dataset_name}/{small_data_name}/val_10.txt"   # val always 10 for each relation
unlabel_file = f"../data/{dataset_name}/{small_data_name}/unlabel_{part}.txt"


def build_data_by_seed(seed):
    # A random data split (for training set)
    # 把生成小语料的代码搬到这边，如果还没有生成，则实时生成，带上seed的文件夹名
    # 目前我不想跑了，不现实，实验太多了。
    return None


def run_teacher(python_file, exp_base_dir, method="base"):
    print(f"Start to train teacher model with split={part}!")
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    save_steps = 100        # 模型中有选择是每个epoch测试还是多少个step测试
    method_name = "baseline"
    bert_mode = 'e1e2'
    for seed in seeds:
        build_data_by_seed(seed)

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
    model_name_or_path = "../../bert-base-uncased"
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


def run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method, max_label_num, alpha=None, iteration=1):
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
    model_name_or_path = "../../bert-base-uncased"
    save_steps = 100
    use_random_subset = False

    for seed in seeds:
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
    result_record_file = os.path.join(exp_base_dir, f"{dataset_name}_{part}_results.txt")
    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + "\n")
        writer.write(f"Iteration: {iter}\n")
        if is_two_stage_first:
            writer.write("first train\n")
        else:
            writer.write("second train\n")
        writer.write(f"{val_list}: avg={round(sum(val_list)/len(val_list), 5)}, std={round(np.std(val_list)*100, 1)}\n")
        writer.write(f"{test_list}: avg={round(sum(test_list)/len(test_list), 5)}, std={round(np.std(test_list)*100, 1)}\n")
        writer.write("\n")
        
    if is_two_stage_first:
        print("two-stage first train:")
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))



def do_self_training(prob_threshold):
    # Merge
    train_merge_confident_example = True
    if train_merge_confident_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_confident_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_example.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

def do_self_training_hard_label(prob_threshold):
    # Merge confident and non-confident instances in hard label
    method = f"merge_all_example_prob{prob_threshold}_in_hard_label"
    python_file = "train_one_self_training_merge_easy_and_hard_example.py"
    run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num)
    collect_results(dataset_name, exp_base_dir, method)


def do_self_training_soft_label(prob_threshold):
    # Merge confident and non-confident instances in soft label
    # 我觉得，最好是non-confident instances 使用soft label，confident不变。代码不知道是否支持
    # 特别是label的设置，搞好的话很方便修改
    # 天生支持的，换下数据生成就行，我们现在都用onehot_label，只要一个是hard一个是soft就行
    method = f"merge_all_example_prob{prob_threshold}_but_ambig_in_soft_label"
    python_file = "train_one_self_training_merge_hard_confident_and_soft_ambig.py"
    run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num)
    collect_results(dataset_name, exp_base_dir, method)

def do_merge_easy_and_ambig2_training_negative(max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    # 这个就是我们的，以及两个剥离实验
    one_random_loss = True
    # follow the Kim's negative training for partial label
    if one_random_loss:
        method = f"joint_merge_easy_ambig2_prob{ambig_prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_negative_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    one_random_loss_hard_label = True
    # follow the Kim's negative training for hard label
    # - partial label
    if one_random_loss_hard_label:
        method = f"joint_merge_easy_ambig1_prob{ambig_prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig1_example_by_accumulate_prob_negative_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)
    
    #### POSITIVE random one for partial
    # - negative
    one_random_positive_loss_partial_label = True
    # follow the Kim's negative training for hard label
    if one_random_positive_loss_partial_label:
        method = f"positive_merge_easy_ambig2_prob{ambig_prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
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
        python_file = "train_baseline.py"
        method = "base"
        run_teacher(python_file, exp_base_dir, method=method)
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
        for prob_threshold in prob_thresholds:
            do_merge_easy_and_ambig2_training_negative(max_label_num, prob_threshold, prob_threshold)


    train_self_training_hard_label = False
    # 将confident和non-confident的句子都加上，都以hard label的形式，即top-1的预测作为答案。
    if train_self_training_hard_label:
        # 需要先确认最好的prob，然后进行以下实验
        do_self_training_hard_label(prob_threshold)

    train_self_training_soft_label = False
    # 将confident和non-confident的句子都加上，都以hard label的形式，即top-1的预测作为答案。
    if train_self_training_soft_label:
        do_self_training_soft_label(prob_threshold)