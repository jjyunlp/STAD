import logging
import os
import json

"""
训练基础模型的脚本

遍历prob_thresholds，先得到在easy上取得最佳val的值。
随后以此值作为区分easy和ambig的阈值。
ambig先从2开始尝试，不行再试3，4等label_num
"""
no_improve_num = 5
seeds = [0, 1, 2, 3, 4]
use_micro = True
use_macro = False

use_max = False
use_sum = True
if use_max and use_sum:
    print("Error max or sum")
    exit()

lr = 5e-5
lr_1 = 5e-5
lr_2 = 5e-5
lrs = [5e-5]
batch_size = 32
num_train_epoch = 20
# global setting
cuda_num = 0
max_label_num = 2
prob_thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

dataset_name = "SemEval-2010-Task8"
exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
part = 400
small_data_name = f"small_data_exp{exp_id}"
train_file = f"../data/{dataset_name}/corpus/{small_data_name}/train_{part}.txt"
val_file = f"../data/{dataset_name}/corpus/val.txt"
test_file = f"../data/{dataset_name}/corpus/test.txt"
label_file = f"../data/{dataset_name}/corpus/label2id.json"
unlabel_file = f"../data/{dataset_name}/corpus/{small_data_name}/unlabel_{part}.txt"


def run_small_data_baseline(python_file, exp_base_dir, method="base"):
    # 我建议small data的时候取消学习率规划操作，直接用一个学习率，这样方便epoch。
    # 设定一个较大的epoch，直到N轮没有提升则停止。
    # 由于数据很小，因此影响不大
    print("Start to train small data!")
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    save_steps = 100        # 模型中有选择是每个epoch测试还是多少个step测试
    method_name = "baseline"
    bert_mode_list = ['e1e2']
    for seed in seeds:
        for lr in lrs:
            # for num_train_epoch in num_train_epochs:
            for bert_mode in bert_mode_list:
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


def run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, prob_threshold, method=None):
    # dataset
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    save_steps = 100
    iteration = 1
    
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
                    f"--prob_threshold={prob_threshold} "
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
        if use_random_subset:
            run_cmd += "--use_random_subset "
        if dataset_name == 're-tacred':
            run_cmd += "--drop_NA "
            run_cmd += "--overwrite_cache "
        if use_micro:
            run_cmd += "--micro_f1 "
        if use_macro:
            run_cmd += "--macro_f1 "
        if use_max:
            run_cmd += "--ambiguity_mode='max' "
        if use_sum:
            run_cmd += "--ambiguity_mode='sum' "
        print(run_cmd)
        os.system(run_cmd)


def run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method):
    # dataset
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    save_steps = 100
    iteration = 1
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
                    f"--prob_threshold={prob_threshold} "
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
        if use_random_subset:
            run_cmd += "--use_random_subset "
        if dataset_name == 're-tacred':
            run_cmd += "--drop_NA "
            run_cmd += "--overwrite_cache "
        if use_micro:
            run_cmd += "--micro_f1 "
        if use_macro:
            run_cmd += "--macro_f1 "
        if use_max:
            run_cmd += "--ambiguity_mode='max' "
        if use_sum:
            run_cmd += "--ambiguity_mode='sum' "

        print(run_cmd)
        os.system(run_cmd)



def collect_results(dataset_name, exp_base_dir, method, is_baseline=False):
    """
    100, 200
    都是相似文件目录的
    method 是完整的method形式
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
                                f"batch{batch_size}_epoch{num_train_epoch}_fix_lr{lr}_seed{seed}")
        if is_baseline:
            result_file = os.path.join(result_file, "results")
        else:
            result_file = os.path.join(result_file, "epoch_0", "results")

        val, test = get_base_results(result_file)
        val_list.append(val)
        test_list.append(test)
        # 也可以读取pseudo data的训练情况
    result_record_file = os.path.join(exp_base_dir, f"{dataset_name}_{part}_results.txt")
    with open(result_record_file, 'a') as writer:
        # 不停的添加新的实验结果
        writer.write(method + "\n")
        writer.write(f"{val_list}: avg={round(sum(val_list)/len(val_list), 5)}\n")
        writer.write(f"{test_list}: avg={round(sum(test_list)/len(test_list), 5)}\n")
        writer.write("\n")
        
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))




def do_self_training(prob_threshold):
    # Merge
    train_merge_easy_example = True
    if train_merge_easy_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_example.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method)
        collect_results(dataset_name, exp_base_dir, method)


def do_merge_ambiguous_training(max_label_num, prob_threshold):
    train_merge_ambig1_by_accumulate_probs = True
    if train_merge_ambig1_by_accumulate_probs:
        method = f"merge_only_ambig1_example_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_merge_ambig1_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    train_merge_ambig2_by_accumulate_probs = True
    if train_merge_ambig2_by_accumulate_probs:
        method = f"merge_only_ambig2_example_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    # merge
    train_merge_easy_and_ambig1_by_accumulate_probs = False
    if train_merge_easy_and_ambig1_by_accumulate_probs:
        method = f"merge_ambig1_example_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_merge_easy_and_hard_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig2_by_accumulate_prob = False
    if train_merge_easy_and_ambig2_by_accumulate_prob:
        method = f"merge_ambig2_example_prob{prob_threshold}_top{max_label_num}_avg_label"
        if use_max:
            method += "_max"
        python_file = "train_one_self_training_merge_easy_and_ambiguity_hard_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)


def do_two_stage(prob_threshold):
    train_two_stage_easy_to_gold = True
    if train_two_stage_easy_to_gold:
        method = f"two_stage_easy_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}"
        python_file = "train_one_self_training_two_stage_easy_to_gold.py"
        run_one_self_training_two_stage_fine_tune(
            python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    # Easy + Hard -> Gold
    train_two_stage_easy_and_hard_to_gold = True
    if train_two_stage_easy_and_hard_to_gold:
        method = f"two_stage_ambig1_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_two_stage_easy_and_hard_to_gold.py"
        run_one_self_training_two_stage_fine_tune(
            python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    # Easy + Ambiguity Hard -> Gold
    train_two_stage_easy_and_ambiguity_hard_to_gold = True
    if train_two_stage_easy_and_ambiguity_hard_to_gold:
        method = f"two_stage_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_two_stage_easy_and_ambiguity_hard_to_gold.py"
        run_one_self_training_two_stage_fine_tune(
            python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)


if __name__ == "__main__":
    server = 'AI'
    if server == '139':
        exp_base_dir = "/data4/jjyunlp/rc_output/self_training_mixup"   # for 139
    if server == 'AI':
        exp_base_dir = f"/data/jjy/ST_RE_micro_accumulate_prob_{part}"
    dataset_name = "SemEval-2010-Task8"

    train_small_base = False
    if train_small_base:
        # Train baseline: only use gold data
        python_file = "train_baseline.py"
        method = "base"
        run_small_data_baseline(python_file, exp_base_dir, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_baseline=True)

    train_merge_easy = False
    if train_merge_easy:
        for prob_threshold in prob_thresholds:
            do_self_training(prob_threshold)

    train_merge_ambiguous = True
    max_label_nums = [2, 3, 4]
    if train_merge_ambiguous:
        prob_threshold = 0.85    # a prob that get best val for easy example
        for max_label_num in max_label_nums:
            do_merge_ambiguous_training(max_label_num, prob_threshold)
    #for prob_threshold in prob_thresholds:
    #    do_two_stage(prob_threshold)

    print(f"SemEval 2010 task: {part}")
