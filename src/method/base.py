import os
import json
import logging
from tqdm import tqdm, trange
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from model.model_utils import TensorOperation
from itertools import cycle
from transformers import (
    # PreTrainedModel,
    # WEIGHTS_NAME,
    AdamW,
    BertConfig,
    # BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
try:
    from torch.utils.tensorboard import SummaryWriter   # 这个是新版本的
except ImportError:
    from tensorboardX import SummaryWriter  # 低版本仍旧用这个

from utils.compute_metrics import EvaluationAndAnalysis


class BaseTrainAndTest():
    """
    train and test the file.
    The model and data are arguments for train/test
    the tokenizer is used to been saved with model.
    """
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tensor_operator = TensorOperation()

    def get_saving_steps(self, only_epoch_check, one_epoch_steps, num_train_epochs,
                         val_step=100, max_times=10):
        """
        不想浪费太多时间，也不想错过好结果，
        所以，val_step最小间隔默认是100，最多测max_times次。跟轮数没关系轮。
        从最后一步往前推算，每隔val_step测一次，最多测max_times次
        """
        steps = []
        if only_epoch_check:
            steps = [(x+1) * one_epoch_steps for x in range(num_train_epochs)]
        else:
            for i in range(max_times):
                step = one_epoch_steps * num_train_epochs - i * val_step
                if step < 0:
                    break
                steps.append(step)
        logging.info(f"Check Steps: {steps}")
        return steps

    def train(self, args, model, train_dataset, val_dataset=None, test_dataset=None,
              output_dir=None, use_random_subset=False, pseudo_training=False,
              use_soft_label=False, learning_rate=None):
        if learning_rate is None:
            print("Set learning rate Or use default")
            learning_rate = args.learning_rate
        print(learning_rate)

        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size
                                      )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        args.save_steps = len(epoch_iterator) * args.num_train_epochs
        # 计算训练的步数
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        one_epoch_steps = len(train_dataloader) // args.gradient_accumulation_steps
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate, eps=args.adam_epsilon)
        # 目前warmup_steps是0
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logging.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        best_dev = 0.0
        no_improve_num = 0
        end_training = False
        logging.info("To get checking steps (based one epochs)")
        check_steps = self.get_saving_steps(True, one_epoch_steps, args.num_train_epochs)
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            # args.save_steps = len(epoch_iterator) * args.num_train_epochs
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                if use_soft_label:
                    labels_onehot = batch[5].to(args.device)
                else:
                    labels_onehot = self.tensor_operator.tensor2onehot(batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    if args.use_lr_scheduler:
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    cond_1 = args.local_rank in [-1, 0]
                    cond_2 = args.logging_steps > 0
                    cond_3 = global_step % args.logging_steps == 0
                    # record loss and lr
                    if cond_1 and cond_2 and cond_3:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset,
                                                      use_random_subset=use_random_subset)
                        if args.dataset_name == "tacred":
                            current_dev = val_results['micro_f1']
                        if args.dataset_name == "trec" or args.dataset_name == "yahoo":
                            # 除了修改这里，还需要修改test的部分。。。
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "re-tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "re-tacred_exclude_NA":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "semeval":
                            # 目前都用micro就行了，衡量工具而已。
                            current_dev = val_results['micro_f1']
                            # current_dev = val_results['macro_f1']
                        elif args.dataset_name == "SemEval-2010-Task8":
                            # 目前都用micro就行了，衡量工具而已。
                            if args.micro_f1:
                                current_dev = val_results['micro_f1']
                            elif args.macro_f1:
                                current_dev = val_results['macro_f1']
                            else:
                                print("Error F1 Evaluation. micro or macro?")
                                exit()
                        elif args.dataset_name == "SemEval-2018-Task7" or args.dataset_name == "top10-re-tacred":
                            if args.micro_f1:
                                current_dev = val_results['micro_f1']
                            elif args.macro_f1:
                                current_dev = val_results['macro_f1']
                            else:
                                print("Error F1 Evaluation. micro or macro?")
                                exit()
                        elif args.dataset_name == "top10-semeval":
                            if args.micro_f1:
                                current_dev = val_results['micro_f1']
                            elif args.macro_f1:
                                current_dev = val_results['macro_f1']
                            else:
                                print("Error F1 Evaluation. micro or macro?")
                                exit()
                        else:
                            print("Error task")
                            logging.info(f"Error task name {args.dataset_name}")
                            exit()
                        logging.info(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        print(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        if current_dev > best_dev:
                            no_improve_num = 0
                            best_dev = current_dev
                            test_results, _, _ = self.test(args, model, test_dataset,
                                                           use_random_subset=use_random_subset)
                            logging.info(f"Test={test_results}\n")
                            print(f"Test={test_results}\n")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            self.tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logging.info(f"Saving model checkpoint to {output_dir}")
                            # optimizer和scheduler是否可以不保存
                            # 目前用不到，就不保存了。主要是为了断点续跑
                            if False:
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logging.info("Saving optimizer and scheduler states to %s", output_dir)
                        else:
                            no_improve_num += 1
                        if no_improve_num >= args.no_improve_num:
                            logging.info(f"No improvement in {args.no_improve_num}.\n" \
                                         f"Current Epoch={epoch}\nCurrent Global Step={global_step}"
                                         )
                            end_training = True
                            break
                    
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break
        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step

    def test(self, args, model, eval_dataset, prefix="", record=False, use_random_subset=False):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        if use_random_subset:
            print("Use Random Subset of Test")
            sample_size = int(0.2 * len(eval_dataset))
            eval_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                np.random.choice(range(len(eval_dataset)), sample_size))
        else:
            print("Use Full Set of Test")
            eval_sampler = SequentialSampler(eval_dataset)

        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)
        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logging.info("***** Running evaluation {} *****".format(prefix))
        logging.info("  Num examples = %d", len(eval_dataset))
        logging.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          }
                outputs = model(args, **inputs)
                logits = outputs[-1]    # 不管什么模型，最后一位放logits
                # eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(),
                                          axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        analysis = EvaluationAndAnalysis()
        if args.micro_f1 and args.macro_f1:
            print("Error Evaluation: both micro and macro F1")
            exit()
        if args.dataset_name == "semeval":  # semeval-2010 task8
            if args.micro_f1:
                result = analysis.micro_f1_exclude_NA(out_label_ids, preds)
            if args.macro_f1:
                result = analysis.macro_f1_for_semeval2010(out_label_ids, preds)
        elif args.dataset_name == "trec" or args.dataset_name == "yahoo":
            # 目前只支持micro_f1
            if args.micro_f1:
                result = analysis.micro_f1(out_label_ids, preds)    # each relation is calculatedd
        elif args.dataset_name == "SemEval-2018-Task7":
            if args.micro_f1:
                result = analysis.micro_f1(out_label_ids, preds)
            if args.macro_f1:
                result = analysis.macro_f1_for_semeval2018(out_label_ids, preds)
        elif args.dataset_name == "re-tacred_exclude_NA":
            if args.micro_f1:
                result = analysis.micro_f1(out_label_ids, preds)    # each relation is calculated
        elif args.dataset_name == "top10-re-tacred":
            if args.micro_f1:
                result = analysis.micro_f1(out_label_ids, preds)
            if args.macro_f1:
                result = analysis.macro_f1_for_top10_re_tacred(out_label_ids, preds)
        elif args.dataset_name == "top10-semeval":
            if args.micro_f1:
                result = analysis.micro_f1(out_label_ids, preds)
            if args.macro_f1:
                # 直接借用top10_re_tacred，反正都是10个非NA
                result = analysis.macro_f1_for_top10_re_tacred(out_label_ids, preds)
        elif args.dataset_name == "re-tacred":
            if args.micro_f1:
                result = analysis.micro_f1_exclude_NA(out_label_ids, preds)
            if args.macro_f1:
                print("Error, No macro f1 for tacred Now")
                exit()
        else:
            print(f"Error dataset_name: {args.dataset_name}")
            exit()
        """
        results.update(result)
        if record:
            output_eval_file = os.path.join(eval_output_dir, f"{split}_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info(" %s %s = %s", split, key, str(result[key]))
                    writer.write("%s %s = %s\n" % (split, key, str(result[key])))
                writer.write("gold\tpred\n")
                for i in range(len(preds)):
                    writer.write(f"{out_label_ids[i]}\t{preds[i]}\n")

        return results
        """
        # 希望把评测标准/结果分析保存等，与测试分割开
        return (result, out_label_ids, preds)

    def label(self, args, model, unlabel_dataset):
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(unlabel_dataset)
        eval_dataloader = DataLoader(unlabel_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logging.info("***** Running Labeling *****")
        logging.info("  Num examples = %d", len(unlabel_dataset))
        logging.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        all_max_probs = []
        all_distributions = []
        all_max_index = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4]
                          }
                outputs = model(args, **inputs)
                logits = outputs[-1]

            nb_eval_steps += 1
            prob_func = torch.nn.Softmax(dim=-1)
            if args.use_bce:
                prob_func = nn.Sigmoid()
            probs = prob_func(logits)
            # hard label
            all_distributions += list(probs.detach().cpu().numpy())
            max_probs_tensor, max_index_tensor = torch.max(probs, dim=-1)
            max_probs = list(max_probs_tensor.detach().cpu().numpy())
            max_index = list(max_index_tensor.detach().cpu().numpy())
            all_max_probs += max_probs
            all_max_index += max_index
        return (all_max_index, all_max_probs, all_distributions)


