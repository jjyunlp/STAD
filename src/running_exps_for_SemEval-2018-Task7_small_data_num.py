import logging
import os
from re import T

"""
训练基础模型的脚本
用SemEval-2018 Task-7 1.1任务的数据作为train/val and test
将SemEval-2018 Task-7 1.2任务的数据中的训练集（自动标注实体，人工标注关系）作为未标注数据，即去掉人工标注关系。
当然，这个也不是完全真实的unlabel，若是完全真实，则需要一个句子中的实体完全进行两两配对。


easy, ambiguous examples的抽取使用prob+label_num两个参数
我们现在这么做：先用prob卡阈值，获取自动标注的数据
"""
no_improve_num = 5
seeds = [0, 1, 2]
use_micro = True
use_macro = False
cuda_num = 0

use_bce = False      # 使用bce loss, prob使用sigmoid
# small_data_name = "1.1_for_train_1.2_for_unlabel"

max_label_num = 0
max_label_nums = [2]
prob_threshold = 0.95
lr = 5e-5
lrs = [5e-5]    # 学习率固定
lr_1 = 5e-5
lr_2 = 5e-5
num_train_epoch = 20
part = 400
dataset_name = "SemEval-2018-Task7"
small_data_name = f"small_data_whole"

train_file = f"../data/{dataset_name}/corpus/train.txt"
val_file = f"../data/{dataset_name}/corpus/val.txt"
test_file = f"../data/{dataset_name}/corpus/test.txt"
label_file = f"../data/{dataset_name}/corpus/label2id.json"
unlabel_file = f"../data/{dataset_name}/corpus/1.2_all_as_unlabel.txt"

use_small_data = True
exp_id = '01'
if use_small_data:
    small_data_name = f"small_data_exp{exp_id}"    # 小数据就会有多种
    train_file = f"../data/{dataset_name}/corpus/{small_data_name}/train_{part}.txt"


def run_small_data_baseline(python_file, dataset_name, exp_base_dir, cuda_num, method="base"):
    # 我建议small data的时候取消学习率规划操作，直接用一个学习率，这样方便epoch。
    # 设定一个较大的epoch，直到N轮没有提升则停止。
    # 由于数据很小，因此影响不大
    print("Start to train small data!")
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    batch_size = 32
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
                if use_bce:
                    run_cmd += "--use_bce "
                print(run_cmd)
                os.system(run_cmd)


def run_one_self_training_two_stage_fine_tune(python_file, dataset_name, exp_base_dir, cuda_num, lr_1, lr_2, max_label_num=1, method=None):
    # dataset
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    batch_size = 32
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
        print(run_cmd)
        os.system(run_cmd)


def run_one_self_training_base_merge_pseudo_data(python_file, dataset_name, exp_base_dir, cuda_num, method, batch_size=None, do_softmin=False):
    # dataset
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    if batch_size is None:
        batch_size = 32
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
        if do_softmin:
            run_cmd += "--do_softmin "
        print(run_cmd)
        os.system(run_cmd)


def do_self_training():
    # Merge
    train_merge_easy_example = False
    if train_merge_easy_example:
        # merge single label hard examples
        method = f"merge_easy_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_example.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, dataset_name, exp_base_dir, cuda_num, method=method)

    # merge
    train_merge_easy_and_hard_by_accumulate_prob = False
    if train_merge_easy_and_hard_by_accumulate_prob:
        # merge single label hard examples
        # v2: easy and hard use same prob, but hard can been sum from label_num/2 lables
        method = f"merge_easy_and_hard_example_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_merge_easy_and_hard_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, dataset_name, exp_base_dir, cuda_num, method=method)

    train_merge_easy_and_ambiguity_hard_by_accumulate_prob = False
    if train_merge_easy_and_ambiguity_hard_by_accumulate_prob:
        method = f"merge_easy_and_ambig_hard_example_prob{prob_threshold}_top{max_label_num}"
        python_file = "train_one_self_training_merge_easy_and_ambiguity_hard_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, dataset_name, exp_base_dir, cuda_num, method=method)


def do_two_stage():
    train_two_stage_easy_to_gold = True
    if train_two_stage_easy_to_gold:
        method = f"two_stage_easy_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}"
        python_file = "train_one_self_training_two_stage_easy_to_gold.py"
        run_one_self_training_two_stage_fine_tune(
            python_file, dataset_name, exp_base_dir, cuda_num, lr_1=lr_1, lr_2=lr_2, method=method)

    for max_label_num in max_label_nums:
        # Easy + Hard -> Gold
        train_two_stage_easy_and_hard_to_gold = True
        if train_two_stage_easy_and_hard_to_gold:
            method = f"two_stage_easy_and_hard_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}"
            python_file = "train_one_self_training_two_stage_easy_and_hard_to_gold.py"
            run_one_self_training_two_stage_fine_tune(
                python_file, dataset_name, exp_base_dir, cuda_num, lr_1=lr_1, lr_2=lr_2,
                max_label_num=max_label_num, method=method)

        # Easy + Ambiguity Hard -> Gold
        train_two_stage_easy_and_ambiguity_hard_to_gold = True
        if train_two_stage_easy_and_ambiguity_hard_to_gold:
            method = f"two_stage_easy_and_ambig_hard_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}"
            python_file = "train_one_self_training_two_stage_easy_and_ambiguity_hard_to_gold.py"
            run_one_self_training_two_stage_fine_tune(
                python_file, dataset_name, exp_base_dir, cuda_num, lr_1=lr_1, lr_2=lr_2,
                max_label_num=max_label_num, method=method)


if __name__ == "__main__":
    server = 'AI'
    if server == '139':
        exp_base_dir = "/data4/jjyunlp/rc_output/self_training_mixup"   # for 139
    if server == 'AI':
        exp_base_dir = f"/data/jjy/ST_RE_micro_accumulate_prob_{part}"
    dataset_name = "SemEval-2018-Task7"

    if use_bce:
        exp_base_dir += "_bce"
    # run_small_data_baseline(dataset_name, exp_base_dir)
    # run_small_data_baseline('re-tacred', exp_base_dir)

    # global setting
    train_small_base = False
    if train_small_base:
        # Train baseline: only use gold data
        python_file = "train_baseline.py"
        method = "base"
        run_small_data_baseline(python_file, dataset_name, exp_base_dir, cuda_num=cuda_num, method=method)

    do_self_training()

    do_two_stage()

    print(f"SemEval 2018 task: {part}")
