"""
Core function for self-training, like sampling
"""
import os
import json
import logging
from .base import BaseTrainAndTest
from sklearn.metrics import accuracy_score
from utils.compute_metrics import EvaluationAndAnalysis
from data_processor.data_loader_and_dumper import JsonDataDumper
import random
from tqdm import tqdm, trange
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

#from model.model_utils import TensorOperation
#from model.model_utils.TensorOperation import tensor2onehot
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


# 工具函数，暂时就这样复制在这边用吧
def tensor2onehot(index, size):
    """
    index: [batch_size]
    size: the size of one hot (num of classification types)
    """
    batch_size = index.size()[0]
    one_hot = torch.FloatTensor(batch_size, size).zero_()
    #scatter_(dim, index, src)
    # dim (python:int) – the axis along which to index, to put src in a tensor
    # src: the value to write, hence, src=1 will output onehot
    # Hence, the index and one_hot(all initialized 0) should have the same size in dimesion dim
    index = index.unsqueeze(-1)
    one_hot.scatter_(1, index, 1)
    return one_hot

class SelfTrainingTrainAndTest(BaseTrainAndTest):
    def tagging(self, args, model, unlabel_tensor, unlabel_inst, id2rel, using_sharpen=False):
        """用当前model给unlabel_data进行标注

        Args:
            model ([type]): [description]
            unlabel_data ([type]): [description]
            output_file (str): 输出的pesudo file
        Return:
            pesudo_inst
        """
        # 用当前模型对这一轮的unlabel data生成伪标签，同时获得一个同人工标注数据大小、分布的数据。
        logging.info("Start label unlabeled data!")
        all_label, all_prob, all_distri = self.label(args, model, unlabel_tensor)
        logging.info("End Labeling.")
        for label, prob, distri in zip(all_label[:20], all_prob[:20], all_distri[:20]):
            logging.info(f"label={label} prob={prob}, distri={distri}")
        # 接着将标注的信息加到句子上(dict形式)，包括预测的类别，预测类别的概率，以及概率分布
        for label, prob, distri, inst in zip(all_label, all_prob, all_distri, unlabel_inst):
            # 我猜这个是地址传递的。为了方便验证预测的准确率，我们将人工的标注放在gold_relation中
            if 'gold_relation' not in inst:     # 这个也防止，再次使用unused data时，会把前一轮的预测当作gold
                # 第一轮的时候，将这个relation变成gold_relation，而relation是一个变动的
                inst['gold_relation'] = inst['relation']    # 为了实验分析时，验证方法选择的可靠性
            inst['relation'] = id2rel[int(label)]   # 这么做是为了让pesudo的inst与gold的有一样的格式
            inst['prob'] = prob.tolist()   # list可以存json，但narray不行
            # inst['distri'] = str(distri)
            if using_sharpen:
                # 将概率分布进行sharpen处理，需要是np array，list不行
                p = distri
                pt = p**(1/args.T)  # 0.5前后调整，若T=1，则相当于没有
                targets = pt / sum(pt)
                inst['distri'] = targets.tolist()
            else:
                inst['distri'] = distri.tolist()
        # 地址传递的，这个unlabel_inst已经被标注好了
        return unlabel_inst

    def tagging_sampling_dump_or_load(current_data_dir,
                                      model,
                                      data_processor,
                                      sampler,
                                      unlabel_inst, 
                                      unlabel_tensor,
                                      label2id):
        pseudo_all_file = os.path.join(
            current_data_dir,
            'pseudo_all.txt'
        )
        pseudo_select_file = os.path.join(
            current_data_dir,
            f'pseudo_select_prob_{args.prob_threshold}.txt'
        )
        pseudo_unused_file = os.path.join(
            current_data_dir,
            f'pseudo_unused_prob_{args.prob_threshold}.txt'
        )
        overwrite_result = True     # 重写result

        # 数据输出到文件中
        logging.info("To tagging and sampling and write to file "
                     "OR load tagged and samplied data from file")
        if os.path.isfile(pseudo_all_file):
            logging.info(f"Read exist all pseudo file from {pseudo_all_file}")
            pseudo_all_inst, _ = data_processor.get_examples_from_file(pseudo_all_file)
        else:
            pseudo_all_inst = self.tagging(args, model, unlabel_tensor, unlabel_inst, id2label)
            data_processor.dump_data(pseudo_all_file, pseudo_all_inst)
        pseudo_all_result_file = os.path.join(current_data_dir, 'pseudo_all_result.txt')
        if not os.path.exists(pseudo_all_result_file) or overwrite_result:
            evaluate_pseudo(pseudo_all_inst, pseudo_all_result_file, label2id)

        pseudo_select_bert = None
        pseudo_unused_bert = None   # 如果是sampler，则不会生成bert，bert这个可用可不用，可用后面自己convert
        if os.path.isfile(pseudo_select_file) and os.path.isfile(pseudo_unused_file):
            logging.info(f"Read exist select pseudo file from {pseudo_select_file}")
            pseudo_select_inst, pseudo_select_bert = data_processor.get_examples_from_file(
                pseudo_select_file)
            pseudo_unused_inst, pseudo_unused_bert = data_processor.get_examples_from_file(
                pseudo_unused_file)
        else:
            pseudo_select_inst, pseudo_unused_inst = sampler.sort_and_prob_return(
                pseudo_all_inst, args.prob_threshold)
            data_processor.dump_data(pseudo_select_file, pseudo_select_inst)
            data_processor.dump_data(pseudo_unused_file, pseudo_unused_inst)
            pseudo_select_bert = data_processor._create_examples_from_json(pseudo_select_inst)
            pseudo_unused_bert = data_processor._create_examples_from_json(pseudo_unused_inst)
        pseudo_select_result_file = os.path.join(current_data_dir, f'pseudo_select_prob_{args.prob_threshold}_result.txt')
        if not os.path.exists(pseudo_select_result_file):  # or overwrite_result:
            evaluate_pseudo(pseudo_select_inst, pseudo_select_result_file, label2id)
        # 上述pseudo data中，都含有gold_label, self_pred, self_prob, self_distri
        print(f"num of pseudo data is: {len(pseudo_select_inst)}")
        return (pseudo_select_inst, pseudo_select_bert, pseudo_unused_inst, pseudo_unused_bert)

    def train_onehot(self, args, model, train_dataset, val_dataset=None, test_dataset=None,
              output_dir=None, use_random_subset=False, pseudo_training=False,
              use_soft_label=False, learning_rate=None, batch_size=None, negative_training=False,
              use_random_one_negative_training=False,
              use_random_one_negative_training_for_hard_label=False,
              use_random_one_positive_training=False):

        if learning_rate is None:
            learning_rate = args.learning_rate
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        if batch_size is None:
            batch_size = args.per_gpu_train_batch_size
        args.train_batch_size = batch_size * max(1, args.n_gpu)
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
                temp_hard_labels = batch[5]
                batch = tuple(t.to(args.device) for t in batch)
                #print("Start to put tensor to inputs")
                onehot_batch = batch[6]
                if use_random_one_negative_training:
                    # 我们的flag是反过来的，如果是fully的，那就是1，否则是0
                    # 将partial label的multi-hot改成只剩一个位置是0
                    # 比如原来是[1, 1, 0, 0, 0]，则随机改成[1,1,1,0,1]
                    new_onehot_batch = []
                    for (onehot, flag) in zip(batch[6], batch[7]):
                        if flag == 1:
                            #print(onehot)
                            #print(onehot.tolist())
                            #print(onehot.detach().tolist())    不需要detach
                            #print("---")
                            new_onehot_batch.append(onehot.tolist())
                            continue
                        negative_index = random.randint(0, args.rel_num - 1)    #random:[] semeval is 19 and re-tacred is 40

                        while onehot[negative_index] == 1:  # If the random label is in C+, re-random it
                            negative_index = random.randint(0, args.rel_num - 1)
                        new_onehot = [1] * args.rel_num

                        new_onehot[negative_index] = 0
                        new_onehot_batch.append(new_onehot)
                    onehot_batch = torch.tensor(new_onehot_batch, dtype=torch.float).to(args.device)
                if use_random_one_negative_training_for_hard_label:
                    new_onehot_batch = []
                    # 将partial的onehot转换成hard的onehot
                    hard_label_batch = tensor2onehot(temp_hard_labels, args.rel_num)
                    for (onehot, flag) in zip(hard_label_batch, batch[7]):
                        if flag == 1:
                            #print(onehot)
                            #print(onehot.tolist())
                            #print(onehot.detach().tolist())    不需要detach
                            #print("---")
                            new_onehot_batch.append(onehot.tolist())
                            continue
                        negative_index = random.randint(0, args.rel_num - 1)

                        while onehot[negative_index] == 1:
                            negative_index = random.randint(0, args.rel_num - 1)
                        new_onehot = [1] * args.rel_num
                        new_onehot[negative_index] = 0
                        new_onehot_batch.append(new_onehot)
                    onehot_batch = torch.tensor(new_onehot_batch, dtype=torch.float).to(args.device)
                if use_random_one_positive_training:
                    # select one positive label from candidates
                    new_onehot_batch = []
                    for (onehot, flag) in zip(batch[6], batch[7]):
                        if flag == 1:
                            # keep confident data unchange 
                            new_onehot_batch.append(onehot.tolist())
                            continue
                        positive_index = random.randint(0, args.rel_num - 1)

                        while onehot[positive_index] == 0:      # 直到这个随机的是partial labels中的
                            positive_index = random.randint(0, args.rel_num - 1)
                        # new_onehot = [0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] # 为什么我当时写成这样？我怀疑会出错
                        new_onehot = [0] * args.rel_num

                        new_onehot[positive_index] = 1
                        new_onehot_batch.append(new_onehot)
                    onehot_batch = torch.tensor(new_onehot_batch, dtype=torch.float).to(args.device)

                #for a, b, c, flag in zip(onehot_batch, batch[5], batch[6], batch[7]):
                #    print(a,c,flag)
                #print(onehot_batch)
                #print(batch[6])
                #exit()
                    # copy from SENT to get random negative label
                    # neg_label = (onehot + torch.LongTensor(onehot.size()).cuda().random_(1, 10)) % 10
                    # print(neg_label)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": onehot_batch,    # 针对randomly select one negative做了一些改变
                          "flags": batch[7],
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
                        elif args.dataset_name == "re-tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "re-tacred_exclude_NA":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "semeval":
                            # 目前都用micro就行了，衡量工具而已。
                            current_dev = val_results['micro_f1']
                            # current_dev = val_results['macro_f1']
                        elif args.dataset_name == "SemEval-2010-Task8":
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

    def train_onehot_two_losses(self, args, model, train_dataset, ambig_dataset, val_dataset=None, test_dataset=None,
              output_dir=None, use_random_subset=False, pseudo_training=False,
              use_soft_label=False, learning_rate=None, batch_size=None):

        if learning_rate is None:
            learning_rate = args.learning_rate
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        if batch_size is None:
            batch_size = args.per_gpu_train_batch_size
        args.train_batch_size = batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size
                                      )
        ambig_sampler = RandomSampler(ambig_dataset) if args.local_rank == -1 else DistributedSampler(ambig_dataset)
        ambig_dataloader = DataLoader(ambig_dataset,
                                      sampler=ambig_sampler,
                                      batch_size=args.train_batch_size
                                      )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        """
        if len(train_dataloader) < len(ambig_dataloader):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        else:
            epoch_iterator = tqdm(ambig_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        """
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
        tr_loss_x, logging_loss_x = 0.0, 0.0
        tr_loss_u, logging_loss_u = 0.0, 0.0
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
            # 谁小用谁，不要循环，太过拟合了。
            for step, (batch, ambig_batch) in enumerate(zip(train_dataloader, cycle(ambig_dataloader))):
            # for step, (batch, ambig_batch) in enumerate(zip(epoch_iterator, ambig_dataloader)):
                #if step >= len(epoch_iterator):
                    # 按照小的数据集训练
                #    break
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": batch[6],
                          }
                outputs = model(args, **inputs)
                loss_x = outputs[0]  # model outputs are always tuple in transformers (see doc)
                # for ambiguous data
                ambig_batch = tuple(t.to(args.device) for t in ambig_batch)
                inputs = {"input_ids": ambig_batch[0],
                          "head_start_id": ambig_batch[1],
                          "tail_start_id": ambig_batch[2],
                          "attention_mask": ambig_batch[3],
                          "token_type_ids": ambig_batch[4],
                          "labels": ambig_batch[5],
                          "labels_onehot": ambig_batch[6],
                          }
                inputs["negative_training"] = True
                outputs = model(args, **inputs)
                loss_u = outputs[0]  # model outputs are always tuple in transformers (see doc)
                # loss = loss_x + loss_u
                loss = loss_x + args.alpha * loss_u
                if args.rampup:
                    loss = loss_x + ((epoch+1)/args.num_train_epochs) * args.alpha * loss_u
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_x = loss_x / args.gradient_accumulation_steps
                    loss_u = loss_u / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                tr_loss_x += loss_x.item()
                tr_loss_u += loss_u.item()
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

                        loss_x_scalar = (tr_loss_x - logging_loss_x) / args.logging_steps
                        logs["loss_x"] = loss_x_scalar
                        logging_loss_x = tr_loss_x
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset,
                                                      use_random_subset=use_random_subset)
                        if args.dataset_name == "tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "re-tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "semeval":
                            # 目前都用micro就行了，衡量工具而已。
                            current_dev = val_results['micro_f1']
                            # current_dev = val_results['macro_f1']
                        elif args.dataset_name == "SemEval-2010-Task8":
                            if args.micro_f1:
                                current_dev = val_results['micro_f1']
                            elif args.macro_f1:
                                current_dev = val_results['macro_f1']
                            else:
                                print("Error F1 Evaluation. micro or macro?")
                                exit()
                        elif args.dataset_name == "SemEval-2018-Task7":
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

    def train_positive_for_easy_negative_for_hard(self, args, model, train_dataset, val_dataset, test_dataset, output_dir, use_random_subset=None, pseudo_training=None, using_soft_label=False):
        """
        original data for supervised training
        augmented data for consistent training
        two losses
        used for pseudo data training, we only select one bt example for each original example
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                        sampler=train_sampler,
                                        batch_size=args.train_batch_size,
                                        drop_last=True
                                        )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(train_dataloader)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // \
                (training_batch_num // args.gradient_accumulation_steps) + 1
        else:
            t_total = training_batch_num // \
                args.gradient_accumulation_steps * args.num_train_epochs
        one_epoch_steps = training_batch_num // args.gradient_accumulation_steps
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
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
                raise ImportError("Please install apex from" \
                                  "https://www.github.com/nvidia/apex to use fp16 training.")
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

        tr_loss, logging_loss = 0.0, 0.0
        tr_loss_x, logging_loss_x = 0.0, 0.0
        tr_loss_c, logging_loss_c = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch",
            disable=args.local_rank not in [-1, 0],
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
            args.save_steps = len(train_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, batch in enumerate(train_dataloader):
                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = batch[11].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(batch[11], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": batch[6],
                          "head_start_id_u": batch[7],
                          "tail_start_id_u": batch[8],
                          "attention_mask_u": batch[9],
                          "token_type_ids_u": batch[10],
                          "labels_u": batch[11],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
                          "current_epoch": epoch+1,     # 用于更新loss weight
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_c = outputs[1:3]   # loss_c: consistency loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_c = loss_c.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_x = loss_x / args.gradient_accumulation_steps
                    loss_c = loss_c / args.gradient_accumulation_steps

                # 是不是应该放在这里啊，要不然accumulation就没用了。
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                tr_loss_x += loss_x.item()
                tr_loss_c += loss_c.item()

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
                    # # 对于小数据，每一轮输出一下就够了，要不然小的那些基本见不到loss的情况
                    if args.logging_steps > check_steps[0]:
                        cond_3 = global_step in check_steps
                    # record loss and lr
                    if cond_1 and cond_2 and cond_3:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss
                        logging.info(f"loss: {logging_loss}")

                        loss_x_scalar = (tr_loss_x - logging_loss_x) / args.logging_steps
                        logs["loss_x"] = loss_x_scalar
                        logging_loss_x = tr_loss_x
                        loss_c_scalar = (tr_loss_c - logging_loss_c) / args.logging_steps
                        logs["loss_c"] = loss_c_scalar
                        logging_loss_c = tr_loss_c

                        print(f"loss={logging_loss}, loss_x={logging_loss_x}, loss_c={logging_loss_c}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset, use_random_subset=use_random_subset)
                        if args.dataset_name == "re-tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "semeval":
                            current_dev = val_results['micro_f1']
                        else:
                            print("Error task")
                            exit()
                        logging.info(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        print(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        if current_dev > best_dev:
                            no_improve_num = 0
                            best_dev = current_dev
                            test_results, _, _ = self.test(args, model, test_dataset, use_random_subset=use_random_subset)
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

                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
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
                train_iterator.close()
                break
            """
            if epoch == args.num_train_epochs - 1:  # Only save model of last epoch for saving disk
                # Save model checkpoint for each epoch, start from 1
                output_dir = os.path.join(args.output_dir, "checkpoint-epoch-{}".format(epoch + 1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
            """
        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step


class Sampling():
    def __init__(self, id2rel, clean_data=None, drop_NA=False):
        """
        clean_data is all gold training insts, convert it to rel2insts
        """
        print("Sampling")
        self.id2rel = id2rel
        if clean_data is not None:
            self.train_rel2inst = self.convert_inst_to_dict(clean_data)
        self.drop_NA = drop_NA

    def evaluate_pseudo_data(self, pseudo_inst, result_file, label2id):
        """evaluate the accuracy and f1 for pseudo data(as we have gold label)"""
        analysis = EvaluationAndAnalysis()
        gold_all = []   # except the padding inst
        pred_all = []
        for inst in pseudo_inst:
            if 'gold_relation' in inst:
                gold_all.append(label2id[inst['gold_relation']])
                pred_all.append(label2id[inst['relation']])     # the pred relation
        result = analysis.micro_f1_for_tacred(gold_all, pred_all, verbose=True)
        acc = accuracy_score(gold_all, pred_all)
        result['acc'] = acc
        with open(result_file, 'w') as writer:
            json.dump(result, writer, indent=2)
    
    def convert_inst_to_dict(self, insts):
        rel2inst = {}
        for inst in insts:
            rel = inst['relation']
            if rel not in rel2inst:
                rel2inst[rel] = [inst]
            else:
                rel2inst[rel].append(inst)
        return rel2inst
    
    def add_pesudo_label(self, inst_data, all_labels, all_probs, all_distributions):
        # all_distri: softmax后，但为argmax，即一个概率分布
        for label, distribution, prob, inst in zip(all_labels, all_distributions, all_probs, inst_data):
            # 直接替换原来的DSlabel，反正目前用不到，这样就能和原始training set一致了
            # inst.add_element('gold_relation', inst['relation'])   # 保留原始的label，便于评测
            # 2021/3/22 需要修改数据结构，使之便于修改和增补
            inst['relation'] = self.id2rel[int(label)]   # 我猜这个是地址传递的`
            inst['distribution'] = str(distribution)
            inst['prob'] = str(prob)
        return inst_data

    def sort_and_return_with_prob_threshold(self, data, prob_threshold):
        """将输入的inst类型的数据按照预测的概率排序，返回高于某个预测概率阈值的句子

        Returns:
            high_prob_data and low_prob_data
        """
        high_prob_data = []
        low_prob_data = []
        for inst in data:
            if float(inst['prob']) < prob_threshold:
                low_prob_data.append(inst)
            else:
                high_prob_data.append(inst)

        return (high_prob_data, low_prob_data)

    def sample_with_prob_threshold(self, data, prob_threshold):
        """返回高于某个预测概率阈值的句子，没有排序

        Returns:
            high_prob_data and low_prob_data
        """
        high_prob_data = []
        low_prob_data = []
        for inst in data:
            if float(inst['prob']) < prob_threshold:
                low_prob_data.append(inst)
            else:
                high_prob_data.append(inst)

        return (high_prob_data, low_prob_data)

    def sample_with_prob_threshold_with_accumulate(self, data, easy_prob_threshold, ambig_prob_threshold, topN):
        """返回高于某个预测概率阈值的句子，没有排序
        并且返回topN个label的总概率大于等于阈值的

        Returns:
            easy, hard, noisy examples
        """
        easy_examples = []
        na_easy_examples = []
        hard_examples = []
        na_hard_examples = []

        noisy_examples = []
        for inst in data:
            if float(inst['prob']) >= easy_prob_threshold:
                if inst['relation'] == 'no_relation' and self.drop_NA:      # NA放1:1
                    na_easy_examples.append(inst)
                else:
                    easy_examples.append(inst)
            else:
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                # all_prob, all_index = zip(*prob_index_tuple)
                sum_of_prob = 0.0
                ambiguity_labels = []
                for prob, index in prob_index_tuple[:topN]:
                    sum_of_prob += prob
                    ambiguity_labels.append(index)
                    if sum_of_prob >= ambig_prob_threshold:
                        break
                
                if sum_of_prob >= ambig_prob_threshold:
                    if self.drop_NA and ambiguity_labels[0] == 0:
                        # 模糊标注中第一个是NA，则加入到na_hard
                        na_hard_examples.append(inst)
                    else:
                        hard_examples.append(inst)
                else:
                    noisy_examples.append(inst)
        
        # 按照1:1加入NA和非NA，若NA少，则比例会小，但tacred基本不会
        easy_examples += na_easy_examples[:len(easy_examples)]
        hard_examples += na_hard_examples[:len(hard_examples)]
        # 剩余的na easy and hard就直接没了
        return (easy_examples, hard_examples, noisy_examples)

    def sample_with_prob_threshold_by_accumulate_with_constraint(self, data, prob_threshold, topN, prob_constraint):
        """返回高于某个预测概率阈值的句子，没有排序
        并且返回topN个label的总概率大于等于阈值的, 且top1/top2 <= 5 ，即两个概率不能相差太大

        Returns:
            easy, hard, noisy examples
        """
        easy_examples = []
        na_easy_examples = []
        hard_examples = []
        na_hard_examples = []

        noisy_examples = []
        for inst in data:
            if float(inst['prob']) >= prob_threshold:
                if inst['relation'] == 'no_relation' and self.drop_NA:      # NA放1:1
                    na_easy_examples.append(inst)
                else:
                    easy_examples.append(inst)
            else:
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                # all_prob, all_index = zip(*prob_index_tuple)
                sum_of_prob = 0.0
                ambiguity_labels = []
                ambiguity_probs = []
                for prob, index in prob_index_tuple[:topN]:
                    sum_of_prob += prob
                    ambiguity_labels.append(index)
                    ambiguity_probs.append(prob)
                    if sum_of_prob >= prob_threshold:
                        break
                
                if sum_of_prob >= prob_threshold and ambiguity_probs[0]/ambiguity_probs[1] <= prob_constraint:
                    if self.drop_NA and ambiguity_labels[0] == 0:
                        # 模糊标注中第一个是NA，则加入到na_hard
                        na_hard_examples.append(inst)
                    else:
                        hard_examples.append(inst)
                else:
                    noisy_examples.append(inst)
        
        # 按照1:1加入NA和非NA，若NA少，则比例会小，但tacred基本不会
        easy_examples += na_easy_examples[:len(easy_examples)]
        hard_examples += na_hard_examples[:len(hard_examples)]
        # 剩余的na easy and hard就直接没了
        return (easy_examples, hard_examples, noisy_examples)

    def sample_with_n_labels_larger_than_ambig_prob_threshold(self, data, easy_prob, ambig_prob):
        """
        返回模糊数据，即那些多个（至少两个）label上的概率大于1/label_num的句子，
        比如10分类，那easy指的是最高概率大于9/10的句子，ambig指的是有多个label的概率>=1/10。
        noisy则是剩余的

        Returns:
            easy, hard, noisy examples
        """
        easy_examples = []
        hard_examples = []  # ambiguous example
        noisy_examples = []

        for inst in data:
            if float(inst['prob']) >= easy_prob:
                easy_examples.append(inst)
            else:
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                prob_index_tuple = sorted(index_with_prob, reverse=True)
                # all_prob, all_index = zip(*prob_index_tuple)
                sum_of_prob = 0.0
                ambiguity_labels = []
                for prob, index in prob_index_tuple:
                    if prob >= ambig_prob:
                        sum_of_prob += prob
                        ambiguity_labels.append(index)
                if len(ambiguity_labels) >= 2:
                    # at least 2 labels so it could be a ambiguous example
                    hard_examples.append(inst)
                else:
                    noisy_examples.append(inst)
        
        return (easy_examples, hard_examples, noisy_examples)
    
    # 以下不知道要不要用
    def dump_data(self, data, data_file):
        dumper = JsonDataDumper(data_file, overwrite=True)
        dumper.dump_all_instance(data)
        logging.info(f"dump {len(data)} samples to {data_file}")

    def output_data(self, label_data, unlabel_data):
        # We output the topN as new labeled data and left as unused unlabeled data
        label_file = os.path.join(self.output_dir, f"epoch_{self.epoch}", f"label_data")
        unlabel_file = os.path.join(self.output_dir, f"epoch_{self.epoch}", f"unlabel_data")
        self.dump_data(label_data, label_file)
        self.dump_data(unlabel_data, unlabel_file)


class DistributionSampling(Sampling):
    """
    分布符合training set的分布，根据当前training set的分布，实时调整新获取数据。
	目前不用了.
    """
    def padding_examples(self, candidate_insts, num):
        """padding examples for relation under correct number"""
        padding_insts = []
        for i in range(num):
            index = random.randint(0, len(candidate_insts)-1)
            padding_insts.append(candidate_insts[index])
        return padding_insts

    def sort_and_prob_return(self, data, prob_threshold, type_limit=False):
        """将输入的inst类型的数据按照预测的概率排序，返回高于某个预测概率阈值的句子

        Args:
            data ([type]): [description]
            distribution ([type]): [description]
            type_limit (bool, optional): [description]. Defaults to False.
            pseudo size: 对于只跑一次self-training的方案，需要一次性加入所有量。

        Returns:
            label_inst and unlabel_inst
        """
        # distribution是一个dict，记录每一个reltion应该获取的实例数目
        # data.sort(key=lambda k: (float(k['prob'])), reverse=True)
        # 不用排序了。。
        high_prob_data = []
        low_prob_data = []
        for inst in data:
            # 因为这个概率条件，可能会导致生成的数据少于训练集，那在merge时没关系，但mixup时需要循环迭代。
            # 改进：为了保证数量和分布，如果unlabel data中没有某些类别的句子，那就随机复制train data中的
            # 如果已经取得的有一些了，那就把该类别下的pseudo data和clean data中的合并在一起，再随机padding
            if float(inst['prob']) < prob_threshold:
                low_prob_data.append(inst)
            else:
                high_prob_data.append(inst)

        return (high_prob_data, low_prob_data)

    def sort_and_distribution_return(self, data, distribution, type_limit=False, padding=False):
        """将输入的inst类型的数据按照预测的概率排序，返回符合分布的一个数据

        Args:
            data ([type]): [description]
            distribution ([type]): [description]
            type_limit (bool, optional): [description]. Defaults to False.
            pseudo size: 对于只跑一次self-training的方案，需要一次性加入所有量。

        Returns:
            label_inst and unlabel_inst
        """
        # distribution是一个dict，记录每一个reltion应该获取的实例数目
        data.sort(key=lambda k: (float(k['prob'])), reverse=True)
        unlabel_data = []
        rel2inst = {}
        for inst in data:
            # 因为这个概率条件，可能会导致生成的数据少于训练集，那在merge时没关系，但mixup时需要循环迭代。
            # 改进：为了保证数量和分布，如果unlabel data中没有某些类别的句子，那就随机复制train data中的
            # 如果已经取得的有一些了，那就把该类别下的pseudo data和clean data中的合并在一起，再随机padding
            if float(inst['prob']) < 0.2:
                unlabel_data.append(inst)
            else:
                if type_limit and not self.check_head_type(inst):
                    # Only NYT dataset has type of entity
                    unlabel_data.append(inst)
                elif inst['relation'] in distribution:
                    if inst['relation'] not in rel2inst:
                        rel2inst[inst['relation']] = [inst]
                    elif len(rel2inst[inst['relation']]) < distribution[inst['relation']]:
                        rel2inst[inst['relation']].append(inst)
                    else:
                        unlabel_data.append(inst)
                else:
                    unlabel_data.append(inst)
        label_data = []
        # 不应该用pseudo这个dict，因为有的关系一个句子都没有的话，就没了
        if padding:
            for rel, train_insts in self.train_rel2inst.items():
                insts = []  # maybe empty
                if rel in rel2inst:
                    insts = rel2inst[rel]
                if len(insts) < distribution[rel]:
                    # get candidate insts for this rel from pseudo data and clean data
                    print(f"padding for {rel} with {distribution[rel]-len(insts)}")
                    candidate_insts = insts + train_insts
                    padding_insts = self.padding_examples(candidate_insts, distribution[rel] - len(insts))
                    label_data += padding_insts
                label_data += insts
        else:
            for _, insts in rel2inst.items():
                label_data += insts

        print(len(label_data))
        return (label_data, unlabel_data)
    
    def random_distribution_return(self, data, distribution, type_limit=False):
        """
        按照distribution，随机的选择新数据.
        data是一个list,随机化一下
        现在inst是普通的dict了.
        """
        random.shuffle(data)
        unlabel_data = []

        rel2inst = {}
        for inst in data:
            if type_limit and not self.check_head_type(inst):
                # Only NYT dataset has type of entity
                unlabel_data.append(inst)
            elif inst.label in distribution:
                if inst.label not in rel2inst:
                    rel2inst[inst.label] = [inst]
                elif len(rel2inst[inst.label]) < distribution[inst.label]:
                    rel2inst[inst.label].append(inst)
                else:
                    unlabel_data.append(inst)
            else:
                unlabel_data.append(inst)
        label_data = []
        for insts in rel2inst.values():
            for inst in insts:
                label_data.append(inst)
            
        return (label_data, unlabel_data)

    def get_distribution(self, data, label_list=None):
        """data是bert格式的结构，REInputExample.
        另外，还需要更新下，对于关系库中的关系，每个关系至少有一个

        Args:
            data ([type]): [description]
            label_list: 用于初始化分布，每个关系都用上，确保至少有一个

        Returns:
            [type]: [description]
        """
        # 根据数据，获取分布
        # 如何保证至少有一个，由于NA肯定有很多，只要谁没有，就+1，然后NA-1 
        distribution_num, distribution_prob = {}, {}
        for inst in data:
            if inst.label not in distribution_num:
                distribution_num[inst.label] = 1
            else:
                distribution_num[inst.label] += 1
        # 更新下，使得每个关系至少有一个.
        # 但这个，如果训练集中某个数据一个都没有的话，其实也没必要。原本是为了防止有的数据比较少，一比例后少于一个。
        # 但我们不会出现这种情况，我们直接copy原有训练集。
        if label_list:
            for label in label_list:
                if label not in distribution_num:
                    logging.info(f"Relation {label} can not found in this data, we init it as 1.")
                    distribution_num[label] = 1
                    distribution_num['no_relation'] -= 1    # 这个反正多的很
        total_size = len(data)
        for rel, num in distribution_num.items():
            distribution_prob[rel] = num/total_size
        return (distribution_num, distribution_prob)

    def get_distribution_from_bag(self, data, label_list=None):
        """data是bert格式的结构，REInputExample.
        另外，还需要更新下，对于关系库中的关系，每个关系至少有一个
        这边的train data是一个bag，即一个list

        Args:
            data ([type]): [description]
            label_list: 用于初始化分布，每个关系都用上，确保至少有一个

        Returns:
            [type]: [description]
        """
        # 根据数据，获取分布
        # 如何保证至少有一个，由于NA肯定有很多，只要谁没有，就+1，然后NA-1 
        distribution_num, distribution_prob = {}, {}
        for inst in data:
            inst = inst[0]  # 是个list
            if inst.label not in distribution_num:
                distribution_num[inst.label] = 1
            else:
                distribution_num[inst.label] += 1
        # 更新下，使得每个关系至少有一个.
        # 但这个，如果训练集中某个数据一个都没有的话，其实也没必要。原本是为了防止有的数据比较少，一比例后少于一个。
        # 但我们不会出现这种情况，我们直接copy原有训练集。
        if label_list:
            for label in label_list:
                if label not in distribution_num:
                    logging.info(f"Relation {label} can not found in this data, we init it as 1.")
                    distribution_num[label] = 1
                    distribution_num['no_relation'] -= 1    # 这个反正多的很
        total_size = len(data)
        for rel, num in distribution_num.items():
            distribution_prob[rel] = num/total_size
        return (distribution_num, distribution_prob)


class GetDistribution():
    """
    得到training set的分布，test的分布，计算两者之间的差距，从而确定本次self-training应该获取的量。
    """
    def __init__(self):
        print("get distribution")
    
    def get_distribution(self, data):
        # 根据数据，获取分布
        distribution_num, distribution_prob = {}, {}
        for inst in data:
            if inst.label not in distribution_num:
                distribution_num[inst.label] = 1
            else:
                distribution_num[inst.label] += 1
        total_size = len(data)
        for rel, num in distribution_num.items():
            distribution_prob[rel] = num/total_size
        return distribution_num, distribution_prob
    
