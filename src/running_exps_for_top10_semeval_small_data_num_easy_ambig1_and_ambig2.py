import logging
import os
import json
from re import T

"""
训练基础模型的脚本


针对semeval只选取top-10个大比例的关系（也没有NA，有些带方向,争取做那个9个关系，都不带方向和NA的），然后unlabel也只保留这些。small data即每个类别取N个
遍历prob_thresholds，先得到在easy上取得最佳val的值。
随后以此值作为区分easy和ambig的阈值。
ambig先从2开始尝试，不行再试3，4等label_num
"""
no_improve_num = 6     # 总共就没多少
seeds = [0, 1, 2, 3, 4]
use_micro = True
use_macro = False
lr = 5e-5
lr_1 = 5e-5
lr_2 = 5e-5
lrs = [5e-5]
batch_size = 32
num_train_epoch = 20
# global setting
cuda_num = 0


prob_thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

dataset_name = 'top10-semeval'
exp_id = '01'   # 后续会跑多种小样本，即多次随机切分
part = 50   # each for class
max_label_nums = [2, 3, 4, 5, 6, 7, 8, 9]
max_label_nums = [5]
max_label_num = 5       # for merge easy 占位符一样, merge ambig1, merge hard
small_data_name = f"balanced_small_data_exp{exp_id}"
train_file = f"../data/semeval_top10_label_excluding_NA/{small_data_name}/train_{part}.txt"
val_file = f"../data/semeval_top10_label_excluding_NA/val.txt"
test_file = f"../data/semeval_top10_label_excluding_NA/test.txt"
label_file = f"../data/semeval_top10_label_excluding_NA/label2id.json"
unlabel_file = f"../data/semeval_top10_label_excluding_NA/{small_data_name}/unlabel_{part}.txt"


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

def run_whole_train_set_baseline(python_file, exp_base_dir, method):
    # 测试下方法的天花板，即直接用所有training set，虽然我们的方法会丢弃个别句子，但不管了
    # 重置训练集
    train_file = f"../data/semeval_top10_label_excluding_NA/train.txt"
    print("Start to train small data!")
    model_type = "bert"
    model_name_or_path = "../../bert-base-uncased"
    save_steps = 100        # 模型中有选择是每个epoch测试还是多少个step测试
    method_name = "baseline_whole_training_set"
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


def run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method, max_label_num=2, alpha=None, iteration=1):
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
        if dataset_name == 're-tacred':
            run_cmd += "--use_random_subset "
            #run_cmd += "--overwrite_cache "
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
        writer.write(f"{val_list}: avg={round(sum(val_list)/len(val_list), 5)}\n")
        writer.write(f"{test_list}: avg={round(sum(test_list)/len(test_list), 5)}\n")
        writer.write("\n")
        
    if is_two_stage_first:
        print("two-stage first train:")
    print(val_list, round(sum(val_list)/len(val_list), 5))
    print(test_list, round(sum(test_list)/len(test_list), 5))



def do_self_training(prob_threshold, iteration=10):
    # Merge
    train_merge_easy_example = True
    if train_merge_easy_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_example.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_example_soft_label = False
    if train_merge_easy_example_soft_label:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_example_soft_label_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_example_soft_label.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig_example_soft_label = False
    if train_merge_easy_and_ambig_example_soft_label:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_and_ambig_example_soft_label_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_and_ambig_example_soft_label.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig2_example_positive = False
    if train_merge_easy_and_ambig2_example_positive:
        # merge partial label ambiguous examples, use positive, mean the loss for partial labels
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_and_ambig2_example_positive_training{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig2_example_positive_sum_loss_then_mean = False
    if train_merge_easy_and_ambig2_example_positive_sum_loss_then_mean:
        # merge partial label ambiguous examples, use positive, mean the loss for partial labels
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_and_ambig2_example_positive_sum_loss_then_mean_training{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training_by_sum_loss.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig2_example_positive_sum_prob_then_loss = False
    if train_merge_easy_and_ambig2_example_positive_sum_prob_then_loss:
        # merge partial label ambiguous examples, use positive, mean the loss for partial labels
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_and_ambig2_example_positive_sum_prob_then_loss_training{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training_by_sum_prob.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)
    # 要跑
    train_merge_easy_and_ambig2_example_negative_sum_prob_then_loss = True
    if train_merge_easy_and_ambig2_example_negative_sum_prob_then_loss:
        # merge partial label ambiguous examples, use positive, mean the loss for partial labels
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_and_ambig2_example_negative_sum_prob_then_loss_training{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_negative_training_by_sum_prob.py"
        # 这边的max_label_num不用，只用easy
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    # run a iterative training for self-training sampling confident data
    train_iter_merge_easy_example = False
    if train_iter_merge_easy_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"iterative_merge_easy_example_prob{prob_threshold}"
        python_file = "train_iterative_self_training_merge_easy_example.py"
        # 这边的max_label_num不用，只用easy
        # 用同一个问价
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, iteration=iteration)
        for iter in iteration:
            collect_results(dataset_name, exp_base_dir, method, iter=iter)

    train_merge_ambig1_example = False
    if train_merge_ambig1_example:
        method = f"merge_ambig1_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_ambig1_example.py"
        # 这边需要max_label_num
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_hard_example = False
    if train_merge_hard_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_hard_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_hard_example.py"
        # 这边需要max_label_num
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_hard_example = False
    if train_merge_easy_and_hard_example:
        # merge single label hard examples
        # 获取一个最高的val对应的prob_threshold
        method = f"merge_easy_and_hard_example_prob{prob_threshold}"
        python_file = "train_one_self_training_merge_easy_and_hard_example.py"
        # 这边需要max_label_num
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, prob_threshold, method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

def do_merge_ambig2_training_positive(max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    # train gold + ambig2 with combined positive by sum prob
    sum_probs = True
    if sum_probs:
        method = f"merge_only_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_sum_prob"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob_positive_training_by_sum_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    sum_losses = True
    if sum_losses:
        method = f"merge_only_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_sum_loss"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob_positive_training_by_sum_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)


def do_merge_ambig2_training_negative(max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    # train gold + ambig2 with combined positive and negative by sum prob
    # clean data with positive
    # ambig2 data with negative
    sum_probs = False
    if sum_probs:
        method = f"merge_only_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_sum_negative_prob"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob_negative_training_by_sum_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    sum_losses = True
    if sum_losses:
        method = f"merge_only_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_sum_negative_loss"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob_negative_training_by_sum_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

def do_merge_easy_and_ambig1(max_label_num, easy_prob_threshold, ambig_prob_threshold):
    method = f"merge_easy_ambig1_example_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}"
    python_file = "train_one_self_training_merge_easy_ambig1_example_by_accumulate_prob_v2.py"
    run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method)
    collect_results(dataset_name, exp_base_dir, method)

def do_merge_easy_and_ambig2_training_negative(max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    # train gold + ambig2 with combined positive and negative by sum prob
    # clean data (gold + easy) with positive
    # ambig2 data with negative
    sum_losses = False
    if sum_losses:
        method = f"merge_easy_and_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_sum_negative_loss"
        python_file = "train_one_self_training_merge_easy_and_ambig2_example_by_accumulate_prob_negative_training_by_sum_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    
    one_random_loss = True
    # follow the Kim's negative training for partial label
    if one_random_loss:
        method = f"fixed_merge_easy_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_random_one_negative_loss"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_negative_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)

    one_random_loss_hard_label = False
    # follow the Kim's negative training for hard label
    if one_random_loss_hard_label:
        method = f"fixed_merge_easy_ambig1_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_random_one_negative_loss"
        python_file = "train_one_self_training_merge_easy_ambig1_example_by_accumulate_prob_negative_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)
    
    #### POSITIVE random one for partial
    one_random_positive_loss_partial_label = False
    # follow the Kim's negative training for hard label
    if one_random_positive_loss_partial_label:
        method = f"merge_easy_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_by_random_one_positive_loss"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training_by_random_one_loss.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method)


def do_merge_ambiguous_training(max_label_num, easy_prob_threshold, ambig_prob_threshold, alpha=None):
    train_merge_ambig1_by_accumulate_probs = False
    if train_merge_ambig1_by_accumulate_probs:
        method = f"merge_only_ambig1_example_easy{prob_threshold}_ambig{ambig_prob_threshold}_top{max_label_num}_batch{batch_size}"
        python_file = "train_one_self_training_merge_ambig1_example_by_accumulate_prob_v2.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    # 先把这个实验跑好
    train_merge_ambig2_by_accumulate_probs = True
    # train gold + ambig2 with combined positive and negative
    if train_merge_ambig2_by_accumulate_probs:
        method = f"merge_only_ambig2_prob{ambig_prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_ambig2_by_accumulate_probs_positive = False
    if train_merge_ambig2_by_accumulate_probs_positive:
        method = f"merge_only_ambig2_example_prob{prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_positive"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob_positive_training.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    # 将1's 的概率加起来，算loss，随后再除以1的数目
    train_merge_ambig2_by_accumulate_probs_positive_by_sum_probs_and_avg = False
    if train_merge_ambig2_by_accumulate_probs_positive_by_sum_probs_and_avg:
        method = f"merge_only_ambig2_example_prob{prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_positive_by_sum_probs_and_avg"
        python_file = "train_one_self_training_merge_ambig2_example_by_accumulate_prob_positive_training_by_sum_probs_and_avg.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    train_merge_easy_and_ambig2_by_accumulate_probs_positive_by_sum_probs_and_avg = True
    if train_merge_easy_and_ambig2_by_accumulate_probs_positive_by_sum_probs_and_avg:
        method = f"merge_easy_and_ambig2_example_prob{prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_positive_by_sum_probs_and_avg"
        python_file = "train_one_self_training_merge_easy_and_ambig2_example_by_accumulate_prob_positive_training_by_sum_probs_and_avg.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    # NEGATIVE TRAINING, 即先算所有0位置的loss，再求平均【仅针对partial example】
    train_two_loss_ambig2_by_accumulate_probs = False
    if train_two_loss_ambig2_by_accumulate_probs:
        method = f"merge_via_two_loss_only_ambig2_{ambig_prob_threshold}_top{max_label_num}_negative_training"
        #method = f"merge_via_two_loss_only_ambig2_example_prob{prob_threshold}_top{max_label_num}_negative_all_cons5_rampup"
        python_file = "train_one_self_training_two_loss_only_ambig2_example_by_accumulate_prob_v2.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    train_two_loss_ambig2_by_accumulate_probs = False
    if train_two_loss_ambig2_by_accumulate_probs:
        #method = f"merge_via_two_loss_only_ambig2_example_prob{prob_threshold}_top{max_label_num}_negative_all_cons5"
        method = f"merge_via_two_loss_only_ambig2_example_prob{prob_threshold}_top{max_label_num}_negative_all_cons5_rampup"
        python_file = "train_one_self_training_two_loss_only_ambig2_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)
    # merge
    train_merge_easy_and_ambig1_by_accumulate_probs = False
    if train_merge_easy_and_ambig1_by_accumulate_probs:
        method = f"merge_easy_ambig1_example_prob{prob_threshold}_top{max_label_num}_batch{batch_size}"
        python_file = "train_one_self_training_merge_easy_ambig1_example_by_accumulate_prob_v2.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig2_by_accumulate_prob = False
    if train_merge_easy_and_ambig2_by_accumulate_prob:
        method = f"merge_easy_ambig2_example_prob{prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method, alpha=alpha)
        collect_results(dataset_name, exp_base_dir, method)

    train_merge_easy_and_ambig2_by_accumulate_prob_positive = False
    if train_merge_easy_and_ambig2_by_accumulate_prob_positive:
        method = f"merge_easy_ambig2_example_prob{prob_threshold}_top{max_label_num}_one_loss_batch{batch_size}_positive"
        python_file = "train_one_self_training_merge_easy_ambig2_example_by_accumulate_prob_positive_training.py"
        run_one_self_training_base_merge_pseudo_data(python_file, exp_base_dir, prob_threshold, method=method, alpha=alpha)
        collect_results(dataset_name, exp_base_dir, method)


def do_two_stage_negative_sum_loss(max_label_num, easy_prob_threshold, ambig_prob_threshold, self_training_epoch=10):
    """
    max_label_num怎么没用？不过可能会继承for循环里那个变量
    跑下easy->gold
    easy + ambig1 -> gold也许不用跑了。毕竟在merge里已经验证过了。[需要跑，用于消融实验]
    """
    # easy + ambig2 to gold
    # ambig2 use negative sum loss training
    train_easy_and_ambig2_to_gold = False
    if train_easy_and_ambig2_to_gold:
        method = f"two_stage_easy_and_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}_by_sum_negative_loss"
        python_file = "train_one_self_training_two_stage_easy_and_ambig2_to_gold_negative_by_sum_loss.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)

    train_easy_and_ambig2_to_gold_random_one_partial = True
    if train_easy_and_ambig2_to_gold_random_one_partial:
        method = f"fixed_two_stage_easy_and_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}_by_random_one_negative_loss"
        python_file = "train_one_self_training_two_stage_easy_and_ambig2_to_gold_negative_by_random_one_loss.py"
        # 这个函数目前没有放max_label_num，直接使用全局变量
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold, ambig_prob_threshold, method=method, max_label_num=max_label_num)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)

    iterative_train_easy_and_ambig2_to_gold = False
    if iterative_train_easy_and_ambig2_to_gold:
        method = f"iterative_two_stage_easy_and_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}_by_sum_negative_loss"
        python_file = "train_iterative_self_training_two_stage_easy_and_ambig2_to_gold_negative_by_sum_loss.py"
        #run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold,
        #                                          ambig_prob_threshold, method=method, iteration=self_training_epoch)

        for iter in range(self_training_epoch):
            collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True, iter=iter)
            collect_results(dataset_name, exp_base_dir, method, is_two_stage=True, iter=iter)

    train_easy_and_ambig1_to_gold = False
    if train_easy_and_ambig1_to_gold:
        method = f"two_stage_easy_and_ambig1_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}_by_sum_negative_loss"
        python_file = "train_one_self_training_two_stage_easy_and_ambig1_to_gold_negative_by_sum_loss.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)


def do_two_stage_positive_sum_prob(max_label_num, easy_prob_threshold, ambig_prob_threshold, self_training_epoch=10):
    """
    max_label_num怎么没用？不过可能会继承for循环里那个变量
    跑下easy->gold
    easy + ambig1 -> gold也许不用跑了。毕竟在merge里已经验证过了。[需要跑，用于消融实验]
    """
    # easy + ambig2 to gold
    # ambig2 use negative sum loss training
    train_easy_and_ambig2_to_gold = False
    if train_easy_and_ambig2_to_gold:
        method = f"two_stage_easy_and_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}_by_sum_positive_prob"
        python_file = "train_one_self_training_two_stage_easy_and_ambig2_to_gold_positive_by_sum_prob.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)


def do_two_stage_positive_sum_loss(max_label_num, easy_prob_threshold, ambig_prob_threshold, self_training_epoch=10):
    """
    max_label_num怎么没用？不过可能会继承for循环里那个变量
    跑下easy->gold
    easy + ambig1 -> gold也许不用跑了。毕竟在merge里已经验证过了。[需要跑，用于消融实验]
    """
    # easy + ambig2 to gold
    # ambig2 use negative sum loss training
    train_easy_and_ambig2_to_gold = True
    if train_easy_and_ambig2_to_gold:
        method = f"two_stage_easy_and_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_top{max_label_num}_batch{batch_size}_by_sum_positive_loss"
        python_file = "train_one_self_training_two_stage_easy_and_ambig2_to_gold_positive_by_sum_loss.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)


def do_two_stage_easy_to_gold(easy_prob_threshold, ambig_prob_threshold):
    train_easy_to_gold = False
    if train_easy_to_gold:
        method = f"two_stage_easy_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{easy_prob_threshold}_batch{batch_size}_by_sum_negative_loss"
        python_file = "train_one_self_training_two_stage_easy_to_gold_negative_by_sum_loss.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1, lr_2, easy_prob_threshold, ambig_prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage_first=True)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)


def do_two_stage(max_label_num, prob_threshold):
    train_two_stage_easy_to_gold = False
    if train_two_stage_easy_to_gold:
        method = f"two_stage_easy_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}"
        python_file = "train_one_self_training_two_stage_easy_to_gold.py"
        run_one_self_training_two_stage_fine_tune(
            python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    # Easy + Hard -> Gold
    train_two_stage_easy_and_hard_to_gold = False
    if train_two_stage_easy_and_hard_to_gold:
        method = f"two_stage_ambig1_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}_negative_training"
        python_file = "train_one_self_training_two_stage_easy_and_hard_to_gold.py"
        run_one_self_training_two_stage_fine_tune(
            python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method)

    # Ambig1 -> Easy + Gold
    train_two_stage_ambig1_to_easy_and_gold = False
    if train_two_stage_ambig1_to_easy_and_gold:
        method = f"two_stage_ambig1_to_easy_and_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}_batch{batch_size}_positive"
        python_file = "train_one_self_training_two_stage_ambig1_to_easy_and_gold_positive_training.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)
    # Ambig2 (Positive Training) -> Easy + Gold
    train_two_stage_ambig2_to_easy_and_gold = True
    if train_two_stage_ambig2_to_easy_and_gold:
        method = f"two_stage_ambig2_to_easy_and_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}_batch{batch_size}_positive"
        python_file = "train_one_self_training_two_stage_ambig2_to_easy_and_gold_positive_training.py"
        run_one_self_training_two_stage_fine_tune( python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)

    # Ambig2 (Negative Training) -> Easy + Gold
    train_two_stage_ambig2_to_easy_and_gold = True
    if train_two_stage_ambig2_to_easy_and_gold:
        method = f"two_stage_ambig2_to_easy_and_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}_batch{batch_size}_negative_to_positive"
        python_file = "train_one_self_training_two_stage_ambig2_to_easy_and_gold_negative_training.py"
        run_one_self_training_two_stage_fine_tune( python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)

    # Easy(Positive) + Ambig2 (Negative) -> Gold (Positive)
    train_two_stage_easy_and_ambig2_to_gold = False
    if train_two_stage_easy_and_ambig2_to_gold:
        method = f"two_stage_easy_and_ambig2_to_gold_lr1_{lr_1}_lr2_{lr_2}_prob{prob_threshold}_top{max_label_num}_batch{batch_size}_negative_to_positive"
        python_file = "train_one_self_training_two_stage_easy_and_ambig2_to_gold_negative_training.py"
        run_one_self_training_two_stage_fine_tune(python_file, exp_base_dir, lr_1=lr_1, lr_2=lr_2, prob_threshold=prob_threshold, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_two_stage=True)


if __name__ == "__main__":
    server = 'AI'
    if server == '139':
        exp_base_dir = "/data4/jjyunlp/rc_output/self_training_mixup"   # for 139
    if server == 'AI':
        if use_micro:
            exp_base_dir = f"/data/jjy/ST_RE_micro_accumulate_prob_{part}_all_new"
        if use_macro:
            exp_base_dir = f"/data/jjy/ST_RE_macro_accumulate_prob_{part}_all_new"

    train_small_base = False
    if train_small_base:
        # Train baseline: only use gold data
        python_file = "train_baseline.py"
        method = "base"
        run_small_data_baseline(python_file, exp_base_dir, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_baseline=True)

    train_merge_easy = True
    if train_merge_easy:
        # also merge ambig
        # for prob_threshold in prob_thresholds:
        prob_threshold = 0.95
        
        do_self_training(prob_threshold)

    train_merge_ambiguous = False
    if train_merge_ambiguous:
        easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
        ambig_prob_threshold = 0.95
        for max_label_num in max_label_nums:
            do_merge_ambig2_training_positive(max_label_num, easy_prob_threshold, ambig_prob_threshold)

    train_merge_ambiguous_negative = False
    if train_merge_ambiguous_negative:
        easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
        ambig_prob_threshold = 0.95
        for max_label_num in max_label_nums:
            do_merge_ambig2_training_negative(max_label_num, easy_prob_threshold, ambig_prob_threshold)

    train_merge_easy_and_ambig1 = False
    if train_merge_easy_and_ambig1:
        easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
        ambig_prob_threshold = 0.95
        for max_label_num in max_label_nums:
            do_merge_easy_and_ambig1(max_label_num, easy_prob_threshold, ambig_prob_threshold)
    
    train_merge_easy_and_ambiguous_negative = True
    if train_merge_easy_and_ambiguous_negative:
        easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
        ambig_prob_threshold = 0.95
        for max_label_num in max_label_nums:
            do_merge_easy_and_ambig2_training_negative(max_label_num, easy_prob_threshold, ambig_prob_threshold)

    train_two_stage = False
    if train_two_stage:
        for max_label_num in max_label_nums:
            prob_threshold = 0.95    # a prob that get best val for easy example
            do_two_stage(max_label_num, prob_threshold)

    train_two_stage_sum_negative_loss = False
    # easy+ambig1/2 to gold
    if train_two_stage_sum_negative_loss:
        for max_label_num in max_label_nums:
            easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
            ambig_prob_threshold = 0.95
            do_two_stage_negative_sum_loss(max_label_num, easy_prob_threshold, ambig_prob_threshold)

    train_two_stage_sum_positive_prob = False
    # easy+ambig1/2 to gold
    if train_two_stage_sum_positive_prob:
        for max_label_num in max_label_nums:
            easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
            ambig_prob_threshold = 0.95
            do_two_stage_positive_sum_prob(max_label_num, easy_prob_threshold, ambig_prob_threshold)

    train_two_stage_sum_positive_loss = False
    # easy+ambig1/2 to gold
    if train_two_stage_sum_positive_loss:
        for max_label_num in max_label_nums:
            easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
            ambig_prob_threshold = 0.95
            do_two_stage_positive_sum_loss(max_label_num, easy_prob_threshold, ambig_prob_threshold)
    
    train_two_stage_easy_to_gold_sum_negative_loss = False
    if train_two_stage_easy_to_gold_sum_negative_loss:
        # 因为easy -> gold不需要max_label_num，因此，单独出来
        easy_prob_threshold = 0.95    # a prob that get best val for easy example,稍微大一点
        ambig_prob_threshold = 0.95
        do_two_stage_easy_to_gold(easy_prob_threshold, ambig_prob_threshold)

    train_whole_train_set = False
    if train_whole_train_set:
        # 测试数据的天花板
        python_file = "train_baseline.py"
        method = "base_whole_train_set"
        run_whole_train_set_baseline(python_file, exp_base_dir, method=method)
        collect_results(dataset_name, exp_base_dir, method, is_baseline=True)

    print(f"Top-10 SemEval: {part}")
