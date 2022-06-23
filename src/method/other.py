from baseline import Baseline

a = Baseline('aa', 'b')


class O():
    def train_cls_e1_e2_alone(self, args, model, train_dataset, val_dataset=None, test_dataset=None, output_dir=None):
        """ Train the model
            Save model after each epoch, not step
            To save the time, we only do evaluation on dev set (to select hyper-ps.)
            and test set after final epoch.
            mode:区分各个不同实验的checkpoint目录
        想方设法把这个函数好好改改，调用方便.
        除了修改test，tester也许可以当作参数赋值，而不是在train里初始化。
        另外，整个训练的逻辑也修改下。
        如果需要测试，则会用到val/test_dataset和tester
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size
                                      )
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
        # Check if continuing training from a checkpoint
        if False and os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

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
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(epoch_iterator) * args.num_train_epochs
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "em_input_ids": batch[6],
                          "em_attention_mask": batch[7],
                          "em_token_type_ids": batch[8],
                          "training": True,     # 当mixup和base混用model时需要这个。反正不用的时候有这个也不无所谓。
                          # "training_baseline": True,    # 临时，为了在self-training中先训练baseline
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
                        val_results, _, _ = self.test_cls_e1_e2_alone(args, model, val_dataset)
                        if args.dataset_name == "tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "re-tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "semeval":
                            # 目前都用micro就行了，衡量工具而已。
                            current_dev = val_results['micro_f1']
                            # current_dev = val_results['macro_f1']
                        else:
                            print("Error task")
                            logging.info(f"Error task name {args.dataset_name}")
                            exit()
                        logging.info(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        print(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        if current_dev > best_dev:
                            no_improve_num = 0
                            best_dev = current_dev
                            test_results, _, _ = self.test_cls_e1_e2_alone(args, model, test_dataset)
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

    def test_cls_e1_e2_alone(self, args, model, eval_dataset, prefix="", record=False):
        """为了用于entity mask
        """
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
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
                          "em_input_ids": batch[6],
                          "em_attention_mask": batch[7],
                          "em_token_type_ids": batch[8],
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
        result = analysis.micro_f1_for_tacred(out_label_ids, preds, verbose=True)
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

    def train_multi_task(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, use_random_subset=False, using_soft_label=False):
        """ human data + merged pesudo data by iteration
        two losses
        multi-task framework
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            # 没必要，上面使用了RandomSampler后，每个epoch都是重新打乱的
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_u = outputs[1:3]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
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
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_u={logging_loss_u}")

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
                    pesudo_epoch_iterator.close()
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

    def train_data_with_augment_training(self, args, model, train_dataset, val_dataset, test_dataset,
        output_dir, use_random_subset=None, pseudo_training=None, using_soft_label=False,
        test_augment=False):
        """
        original data for supervised training
        augmented data for also for supervised training
        one averaged loss
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
                        if test_augment:
                            val_results, _, _ = self.test_data_with_augment_testing(args, model, val_dataset, use_random_subset=use_random_subset)
                        else:
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
                            if test_augment:
                                test_results, _, _ = self.test_data_with_augment_testing(args, model, test_dataset, use_random_subset=use_random_subset)
                            else:
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

    def test_data_with_augment_testing(self, args, model, eval_dataset, prefix="", record=False, use_random_subset=False):
        """
        two inputs for each sentence, one original sentence and one entity-mask sentence
        """
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
                          "input_ids_u": batch[6],      # These are entity mask
                          "head_start_id_u": batch[7],
                          "tail_start_id_u": batch[8],
                          "attention_mask_u": batch[9],
                          "token_type_ids_u": batch[10],
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
        result = analysis.micro_f1_for_tacred(out_label_ids, preds, verbose=True)
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

    def train_data_with_augment_consistency_training(self, args, model, train_dataset, val_dataset, test_dataset, output_dir, use_random_subset=None, pseudo_training=None, using_soft_label=False):
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

    def train_clean_ce_noisy_kl(self, args, model, clean_dataset, noisy_dataset, val_dataset, test_dataset, output_dir, use_random_subset=None, pseudo_training=None, using_soft_label=False):
        """
        clean data for supervised training(CE)
        noisy data for consistent training(KL)
        two losses
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        clean_sampler = RandomSampler(clean_dataset) \
                            if args.local_rank == -1 else DistributedSampler(clean_dataset)
        clean_dataloader = DataLoader(clean_dataset,
                                      sampler=clean_sampler,
                                      batch_size=args.train_batch_size,
                                      drop_last=True
                                      )
        noisy_sampler = RandomSampler(noisy_dataset) \
                            if args.local_rank == -1 else DistributedSampler(noisy_dataset)
        noisy_dataloader = DataLoader(noisy_dataset,
                                        sampler=noisy_sampler,
                                        batch_size=int(args.train_batch_size/2),     # 每个句子有两个
                                        drop_last=True
                                        )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(clean_dataloader)
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
        logging.info("  Num examples = %d", len(clean_dataset))
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
            # 以clean data为主
            for step, (clean_batch, noisy_batch) in enumerate(zip(clean_dataloader, cycle(noisy_dataloader))):
                model.train()

                clean_labels_onehot = tensor2onehot(clean_batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    noisy_labels_onehot = noisy_batch[11].to(args.device)
                else:
                    noisy_labels_onehot = tensor2onehot(noisy_batch[11], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                clean_batch = tuple(t.to(args.device) for t in clean_batch)
                noisy_batch = tuple(t.to(args.device) for t in noisy_batch)
                inputs = {"input_ids": clean_batch[0],
                          "head_start_id": clean_batch[1],
                          "tail_start_id": clean_batch[2],
                          "attention_mask": clean_batch[3],
                          "token_type_ids": clean_batch[4],
                          "labels": clean_batch[5],
                          "labels_onehot": clean_labels_onehot,
                          "noisy_input_ids_a": noisy_batch[0],
                          "noisy_head_start_id_a": noisy_batch[1],
                          "noisy_tail_start_id_a": noisy_batch[2],
                          "noisy_attention_mask_a": noisy_batch[3],
                          "noisy_token_type_ids_a": noisy_batch[4],
                          "noisy_input_ids_b": noisy_batch[6],
                          "noisy_head_start_id_b": noisy_batch[7],
                          "noisy_tail_start_id_b": noisy_batch[8],
                          "noisy_attention_mask_b": noisy_batch[9],
                          "noisy_token_type_ids_b": noisy_batch[10],
                          "noisy_labels": noisy_batch[11],
                          "noisy_labels_onehot": noisy_labels_onehot,
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

    def train_data_with_rdrop(self, args, model, train_dataset, val_dataset, test_dataset, output_dir, use_random_subset=None, pseudo_training=None, using_soft_label=False):
        """
        original data for supervised training
        twice original data for r-drop training
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
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
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

    def train_mixmatch(self, args, model, label_train_dataset, unlabel_train_dataset,
                       val_dataset=None, test_dataset=None, mode="base"):
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        # label_train_sampler = RandomSampler(label_train_dataset) if args.local_rank == -1 else DistributedSampler(label_train_dataset)
        # label_train_dataloader = DataLoader(label_train_dataset,
        #                                     sampler=label_train_sampler,
        #                                     batch_size=args.train_batch_size,
        #                                     drop_last=True      # 要不然会出现与unlabel data对不齐的情况。
        #                                     )
        # unlabel_train_sampler = RandomSampler(unlabel_train_dataset) if args.local_rank == -1 else DistributedSampler(unlabel_train_dataset)
        # unlabel_train_dataloader = DataLoader(unlabel_train_dataset,
        #                                       sampler=unlabel_train_sampler,
        #                                       batch_size=args.train_batch_size,
        #                                       drop_last=True
        #                                       )
        # 计算训练的步数
        
        # if args.max_steps > 0:
        #     t_total = args.max_steps
        #     args.num_train_epochs = args.max_steps // (len(label_train_dataloader) // args.gradient_accumulation_steps) + 1
        # else:
        #     t_total = len(label_train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # one_epoch_steps = len(label_train_dataloader) // args.gradient_accumulation_steps
        train_iteration = 200     # 每一个epoch需要跑的次数

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
                          lr=args.learning_rate, eps=args.adam_epsilon)
        # 目前warmup_steps是0
        # 学习率是否可以尝试在每一个epoch中减少至0，而不是整个训练过程。
        # 毕竟，目前可以理解成一个epoch是原来的一次训练。
        # 若如此，则需要把scheduler放到for循环中初始化,num_training_steps=每一轮的steps
        # 看起来optimizer也应该放下去。。。这边就暂时理解不了。
        # 也可以考虑下干脆就不要学习率衰减了
        t_total = train_iteration * args.num_train_epochs   # 反正目前也不更新lr
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
        logging.info("  Label Num examples = %d", len(label_train_dataset))
        logging.info("  Unlabel Num examples = %d", len(unlabel_train_dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logging.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        # logging.info("  Total optimization steps = %d", t_total)
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        tr_loss, logging_loss = 0.0, 0.0
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
        model.zero_grad()
        training_epochs = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        best_dev = 0.0
        no_improve_num = 0
        end_training = False
        logging.info("To get checking steps (based one epochs)")
        # check_steps = self.get_saving_steps(True, one_epoch_steps, args.num_train_epochs)
        for epoch, _ in enumerate(training_epochs):
            # 放epoch下，从而每一轮都选取随机的unlabel data
            logging.info(f"train_batch_size={args.train_batch_size}")
            label_train_sampler = RandomSampler(label_train_dataset) if args.local_rank == -1 else DistributedSampler(label_train_dataset)
            label_train_dataloader = DataLoader(label_train_dataset,
                                                sampler=label_train_sampler,
                                                batch_size=args.train_batch_size,
                                                drop_last=True      # 要不然会出现与unlabel data对不齐的情况。
                                                )
            unlabel_train_sampler = RandomSampler(unlabel_train_dataset) if args.local_rank == -1 else DistributedSampler(unlabel_train_dataset)
            unlabel_train_dataloader = DataLoader(unlabel_train_dataset,
                                                  sampler=unlabel_train_sampler,
                                                  batch_size=args.train_batch_size,
                                                  drop_last=True
                                                  )
            if end_training:
                # 提前结束训练
                break
            # label_epoch_iterator = tqdm(label_train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            # args.save_steps = len(label_epoch_iterator) * args.num_train_epochs
            # Unlabel data在这边每次重新打乱下，从而能够在多轮迭代的前提下，基本随机的遍历所有数据
            # for step, (label_batch, unlabel_batch) in enumerate(zip(cycle(label_epoch_iterator), cycle(unlabel_train_dataloader))):
            # 考虑提前设定步数

            # for step, (label_batch, unlabel_batch) in enumerate(zip(label_epoch_iterator, cycle(unlabel_train_dataloader))):
            # step_tqdm = tqdm(iterator(rain_iteration), desc="Iteration")
            training_iterator = trange(
                0, train_iteration, desc="Iteration", disable=args.local_rank not in [-1, 0],
            )
            # for step in tqdm(training_iterator):
            label_train_iter = iter(label_train_dataloader)
            unlabel_train_iter = iter(unlabel_train_dataloader)
            for step, _ in enumerate(training_iterator):
                # 基本要判断下是否还是个iterator（因为迭代完了），不是的话再转化下，
                try:
                    label_batch = next(label_train_iter)
                except:
                    label_train_iter = iter(label_train_dataloader)
                    label_batch = next(label_train_iter)
                try:
                    unlabel_batch = next(unlabel_train_iter)
                except:
                    unlabel_train_iter = iter(unlabel_train_dataloader)
                    unlabel_batch = next(unlabel_train_iter)

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                labels_onehot = tensor2onehot(label_batch[5], 42).to(args.device)
                unlabels_onehot = tensor2onehot(unlabel_batch[5], 42).to(args.device)
                label_batch = tuple(t.to(args.device) for t in label_batch)
                unlabel_batch = tuple(t.to(args.device) for t in unlabel_batch)     # 这个只有batch个
                # label_batch[0] = [8, 1, 128]  unlabel_batch[0] = [8, 3, 128] -> [8, 4, 128]
                # 现在unlabel_batch[0] = [8, 4, 128], 其中4中的第一个是原句
                input_ids = torch.cat([label_batch[0], unlabel_batch[0]], dim=1)
                # 合并label + unlabel: [8, 1] -> [8, K+1] 每一个4维的向量中，第一个位置是human
                head_start_id = torch.cat([label_batch[1], unlabel_batch[1]], dim=1)
                tail_start_id = torch.cat([label_batch[2], unlabel_batch[2]], dim=1)
                attention_mask = torch.cat([label_batch[3], unlabel_batch[3]], dim=1)
                token_type_ids = torch.cat([label_batch[4], unlabel_batch[4]], dim=1)

                inputs = {"input_ids": input_ids,
                          "head_start_id": head_start_id,
                          "tail_start_id": tail_start_id,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "labels": labels_onehot,  # 还需要实时生成unlabel 部分的targets
                          "unlabels_onehot": unlabels_onehot,
                          "epoch": epoch,
                          "total_epoch": args.num_train_epochs
                          }
                loss, loss_x, loss_u = model(args, **inputs, training=True)
                # print(loss)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_x = loss_x / args.gradient_accumulation_steps
                    loss_u = loss_u / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward(retain_graph=True)

                tr_loss += loss.item()
                tr_loss_x += loss_x.item()
                tr_loss_u += loss_u.item()

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    # !!!!!!!
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
                    """
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
                        if args.dataset_name == "tacred":
                            current_dev = val_results['micro_f1']
                        elif args.dataset_name == "semeval":
                            current_dev = val_results['macro_f1']
                        else:
                            print("Error task")
                            exit()
                        logging.info(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        print(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
                        if current_dev > best_dev:
                            no_improve_num = 0
                            best_dev = current_dev
                            test_results, _, _ = self.test(args, model, test_dataset)
                            logging.info(f"Test={test_results}\n")
                            print(f"Test={test_results}\n")
                            output_dir = os.path.join(args.output_dir, mode, "best_dev_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            self.tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logging.info(f"Saving model checkpoint to {output_dir}")

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
                    """
                # if args.max_steps > 0 and global_step > args.max_steps:
                #     label_epoch_iterator.close()
                #     break
            # if args.max_steps > 0 and global_step > args.max_steps:
            #     train_iterator.close()
            #     break
            # 直接在每一轮结束的时候进行测试
            val_results, _, _ = self.test(args, model, val_dataset)
            if args.dataset_name == "re-tacred":
                current_dev = val_results['micro_f1']
            elif args.dataset_name == "semeval":
                current_dev = val_results['micro_f1']
            else:
                print("Error task")
                exit()
            logging.info(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
            logging.info(f"Current Detailed: {val_results}")
            print(f"{global_step}\nDev={current_dev}\tBest Dev={best_dev}\n")
            if current_dev > best_dev:
                no_improve_num = 0
                best_dev = current_dev
                test_results, _, _ = self.test(args, model, test_dataset)
                logging.info(f"Test={test_results}\n")
                print(f"Test={test_results}\n")
                output_dir = os.path.join(args.output_dir, mode, "best_dev_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info(f"Saving model checkpoint to {output_dir}")

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

    def train_mixup(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir):
        """ Train the model
        encoder with mixup
        这个基本不用，即默认输入的pseudo data等于human data，顶多少几个。当时用于模型每轮更新的mixip
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
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // \
                (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // \
                args.gradient_accumulation_steps * args.num_train_epochs
        one_epoch_steps = len(train_dataloader) // args.gradient_accumulation_steps
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
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(epoch_iterator, cycle(pesudo_dataloader))
            ):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_u = outputs[1:3]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
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
                        logging.info(f"loss: {logging_loss}")
                        
                        loss_x_scalar = (tr_loss_x - logging_loss_x) / args.logging_steps
                        logs["loss_x"] = loss_x_scalar
                        logging_loss_x = tr_loss_x
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_u={logging_loss_u}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    epoch_iterator.close()
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

    def train_merge_mixup(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir):
        """ Train the model
        encoder with mixup
        这个应该叫base_merge_mixup
        pseudo 数据越来越多，因此，cycle human data
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
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
                        logging.info(f"loss: {logging_loss}")
                        
                        print(f"loss={loss}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_mixmatch_merge(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """ human data + merged pesudo data by iteration
        two losses
        multi-task framework
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_u = outputs[1:3]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
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
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_u={logging_loss_u}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_mixmatch_merge_with_consistency(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """ human data + merged pesudo data by iteration
        augment data for consistency loss
        two losses
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "task_a": True,
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_u = outputs[1:3]   # loss_c: consistency loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_x = loss_x / args.gradient_accumulation_steps
                    loss_u = loss_u / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # 添加这个
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    if args.use_lr_scheduler:
                        scheduler.step()  # Update learning rate schedule
                    model.zero_grad()

                tr_loss += loss.item()
                tr_loss_x += loss_x.item()
                tr_loss_u += loss_u.item()
                # 可以考虑上一轮得到的u_logits直接拿来用，但是loss更新过一次了
                inputs = {"input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "bag_input_ids_u": [self_batch[6], self_batch[12], self_batch[18]],
                          "bag_head_start_id_u": [self_batch[7], self_batch[13], self_batch[19]],
                          "bag_tail_start_id_u": [self_batch[8], self_batch[14], self_batch[20]],
                          "bag_attention_mask_u": [self_batch[9], self_batch[15], self_batch[21]],
                          "bag_token_type_ids_u": [self_batch[10], self_batch[16], self_batch[22]],      # label其实共享就行
                          "labels_onehot_u": labels_onehot_u,
                          "task_b": True,
                          }
                """
                inputs = {"input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "bag_input_ids_u": [self_batch[6]],
                          "bag_head_start_id_u": [self_batch[7]],
                          "bag_tail_start_id_u": [self_batch[8]],
                          "bag_attention_mask_u": [self_batch[9]],
                          "bag_token_type_ids_u": [self_batch[10]],
                          "labels_onehot_u": labels_onehot_u,
                          "task_b": True,
                          }
                """
                loss_c = model(args, **inputs)

                if args.n_gpu > 1:
                    loss_c = loss_c.mean()
                if args.gradient_accumulation_steps > 1:
                    loss_c = loss_c / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_c.backward()
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
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u
                        loss_c_scalar = (tr_loss_c - logging_loss_c) / args.logging_steps
                        logs["loss_c"] = loss_c_scalar
                        logging_loss_c = tr_loss_c

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_u={logging_loss_u}, loss_c={logging_loss_c}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_uda_with_merge_consistency(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """
        human data for supervised training
        merged pesudo data by iteration for consistent training
        two losses
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "bag_input_ids_u": [self_batch[6], self_batch[12], self_batch[18]],
                          "bag_head_start_id_u": [self_batch[7], self_batch[13], self_batch[19]],
                          "bag_tail_start_id_u": [self_batch[8], self_batch[14], self_batch[20]],
                          "bag_attention_mask_u": [self_batch[9], self_batch[15], self_batch[21]],
                          "bag_token_type_ids_u": [self_batch[10], self_batch[16], self_batch[22]],      # label其实共享就行
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

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_c={logging_loss_c}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_uda_with_merge_consistency_pipeline(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """
        human data for supervised training
        merged pesudo data by iteration for consistent training
        two losses
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "bag_input_ids_u": [self_batch[6], self_batch[12], self_batch[18]],
                          "bag_head_start_id_u": [self_batch[7], self_batch[13], self_batch[19]],
                          "bag_tail_start_id_u": [self_batch[8], self_batch[14], self_batch[20]],
                          "bag_attention_mask_u": [self_batch[9], self_batch[15], self_batch[21]],
                          "bag_token_type_ids_u": [self_batch[10], self_batch[16], self_batch[22]],      # label其实共享就行
                          "task_a": True,
                          "current_epoch": epoch+1,     # 用于更新loss weight
                          }
                outputs = model(args, **inputs)
                loss_x = outputs  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss_x = loss_x.mean()
                if args.gradient_accumulation_steps > 1:
                    loss_x = loss_x / args.gradient_accumulation_steps

                # 是不是应该放在这里啊，要不然accumulation就没用了。
                if args.fp16:
                    with amp.scale_loss(loss_x, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_x.backward()
                inputs = {
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "bag_input_ids_u": [self_batch[6], self_batch[12], self_batch[18]],
                          "bag_head_start_id_u": [self_batch[7], self_batch[13], self_batch[19]],
                          "bag_tail_start_id_u": [self_batch[8], self_batch[14], self_batch[20]],
                          "bag_attention_mask_u": [self_batch[9], self_batch[15], self_batch[21]],
                          "bag_token_type_ids_u": [self_batch[10], self_batch[16], self_batch[22]],      # label其实共享就行
                          "task_b": True,
                          "current_epoch": epoch+1,     # 用于更新loss weight
                          }
                outputs = model(args, **inputs)
                loss_c = outputs  # model outputs are always tuple in transformers (see doc)
                if args.n_gpu > 1:
                    loss_c = loss_c.mean()
                if args.gradient_accumulation_steps > 1:
                    loss_c = loss_c / args.gradient_accumulation_steps

                # 是不是应该放在这里啊，要不然accumulation就没用了。
                if args.fp16:
                    with amp.scale_loss(loss_c, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_c.backward()


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

                        loss_x_scalar = (tr_loss_x - logging_loss_x) / args.logging_steps
                        logs["loss_x"] = loss_x_scalar
                        logging_loss_x = tr_loss_x
                        loss_c_scalar = (tr_loss_c - logging_loss_c) / args.logging_steps
                        logs["loss_c"] = loss_c_scalar
                        logging_loss_c = tr_loss_c


                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_mixmatch_merge_with_mixup_consistency(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """ human data + merged pesudo data by iteration
        human and pseudo data both use augment data
        augment data for consistency loss
        two losses
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "bag_input_ids": [batch[6], batch[12], batch[18]],
                          "bag_head_start_id": [batch[7], batch[13], batch[19]],
                          "bag_tail_start_id": [batch[8], batch[14], batch[20]],
                          "bag_attention_mask": [batch[9], batch[15], batch[21]],
                          "bag_token_type_ids": [batch[10], batch[16], batch[22]],      # label其实共享就行
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "bag_input_ids_u": [self_batch[6], self_batch[12], self_batch[18]],
                          "bag_head_start_id_u": [self_batch[7], self_batch[13], self_batch[19]],
                          "bag_tail_start_id_u": [self_batch[8], self_batch[14], self_batch[20]],
                          "bag_attention_mask_u": [self_batch[9], self_batch[15], self_batch[21]],
                          "bag_token_type_ids_u": [self_batch[10], self_batch[16], self_batch[22]],      # label其实共享就行
                          "training": True,
                          "current_epoch": epoch+1,     # 用于更新loss weight
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_u, loss_c = outputs[1:4]   # loss_c: consistency loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
                    loss_c = loss_c.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    loss_x = loss_x / args.gradient_accumulation_steps
                    loss_u = loss_u / args.gradient_accumulation_steps
                    loss_c = loss_c / args.gradient_accumulation_steps

                # 是不是应该放在这里啊，要不然accumulation就没用了。
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                tr_loss_x += loss_x.item()
                tr_loss_u += loss_u.item()
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
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u
                        loss_c_scalar = (tr_loss_c - logging_loss_c) / args.logging_steps
                        logs["loss_c"] = loss_c_scalar
                        logging_loss_c = tr_loss_c

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_u={logging_loss_u}, loss_c={logging_loss_c}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_mixmatch_merge_with_aux(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """ human data + merged pesudo data by iteration
        two losses
        add aux inputs
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "aux_mask_a_u": self_batch[6],
                          "aux_mask_b_u": self_batch[7],
                          "aux_mask_c_u": self_batch[8],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_x, loss_u = outputs[1:3]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    loss_x = loss_x.mean()
                    loss_u = loss_u.mean()
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
                        loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                        logs["loss_u"] = loss_u_scalar
                        logging_loss_u = tr_loss_u

                        print(f"loss={loss}, loss_x={logging_loss_x}, loss_u={logging_loss_u}")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_self_training_base(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """
        a universal training framework for self-training
        base: no augmentation, we can run it on gpu05(1080ti)
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
        tr_loss_xc, logging_loss_xc = 0.0, 0.0
        tr_loss_uc, logging_loss_uc = 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "training": True,
                          "current_epoch": epoch+1,     # 用于更新loss weight
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                loss_x, loss_u, loss_xc, loss_uc = outputs[1:5]   # loss_c: consistency loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if loss_x is not None:
                        loss_x = loss_x.mean()
                    if loss_u is not None:
                        loss_u = loss_u.mean()
                    if loss_xc is not None:
                        loss_xc = loss_xc.mean()
                    if loss_uc is not None:
                        loss_uc = loss_uc.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    if loss_x is not None:
                        loss_x = loss_x / args.gradient_accumulation_steps
                    if loss_u is not None:
                        loss_u = loss_u / args.gradient_accumulation_steps
                    if loss_xc is not None:
                        loss_xc = loss_xc / args.gradient_accumulation_steps
                    if loss_uc is not None:
                        loss_uc = loss_uc / args.gradient_accumulation_steps

                # 是不是应该放在这里啊，要不然accumulation就没用了。
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if loss_x is not None:
                    tr_loss_x += loss_x.item()
                if loss_u is not None:
                    tr_loss_u += loss_u.item()
                if loss_xc is not None:
                    tr_loss_xc += loss_xc.item()
                if loss_uc is not None:
                    tr_loss_uc += loss_uc.item()

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

                        if loss_x is not None:
                            loss_x_scalar = (tr_loss_x - logging_loss_x) / args.logging_steps
                            logs["loss_x"] = loss_x_scalar
                            logging_loss_x = tr_loss_x

                        if loss_u is not None:
                            loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                            logs["loss_u"] = loss_u_scalar
                            logging_loss_u = tr_loss_u

                        if loss_xc is not None:
                            loss_xc_scalar = (tr_loss_xc - logging_loss_xc) / args.logging_steps
                            logs["loss_xc"] = loss_xc_scalar
                            logging_loss_xc = tr_loss_xc

                        if loss_uc is not None:
                            loss_uc_scalar = (tr_loss_uc - logging_loss_uc) / args.logging_steps
                            logs["loss_uc"] = loss_uc_scalar
                            logging_loss_uc = tr_loss_uc

                        print(f"loss={loss}", 
                              f"loss_x={logging_loss_x} ",
                              f"loss_u={logging_loss_u} ",
                              f"loss_xc={logging_loss_xc} ",
                              f"loss_uc={logging_loss_uc} ")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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

    def train_self_training(self, args, model, train_dataset, pesudo_dataset, val_dataset, test_dataset, output_dir, using_soft_label=False):
        """
        a universal training framework for self-training
        """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        logging.info(f"train_batch_size={args.train_batch_size}")
        train_sampler = RandomSampler(train_dataset) \
                            if args.local_rank == -1 else DistributedSampler(train_dataset)
        # 这个由于数据量很小，我怕丢失几个句子会对结果有所影响。有一个解决方案就是迭代选择的时候，不是pesudo data大
        # 很多，那就每次随机sample train data，然后生成，这样就算是弥补了最后一个batch的丢失问题。
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.train_batch_size,
        #                               drop_last=True
        #                               )
        pesudo_sampler = RandomSampler(pesudo_dataset) \
                            if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
        pesudo_dataloader = DataLoader(pesudo_dataset,
                                       sampler=pesudo_sampler,
                                       batch_size=args.train_batch_size,
                                       drop_last=True
                                       )
        # 计算训练的步数
        # 在这边，理想情况下是pesudo data >= train data
        # pesudo 后面在每个epoch还会生成一次...为了training_batch_num及相应的参数，再说吧
        training_batch_num = len(pesudo_dataloader)
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
        tr_loss_x, tr_loss_u, logging_loss_x, logging_loss_u = 0.0, 0.0, 0.0, 0.0
        tr_loss_xc, logging_loss_xc = 0.0, 0.0
        tr_loss_uc, logging_loss_uc = 0.0, 0.0
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
        # 某一轮的时候，对我而言训练集已经固定了，目标是训练得到最好的val对应的模型。
        # 虽然后面的mixup会打乱，lamda的不同导致每一个epoch都有不一样的东西，但没办法。。。我们不是在语料层面进行
        # mixup，而是hidden states层面
        for epoch, _ in enumerate(train_iterator):
            if end_training:
                # 提前结束训练
                break
            # 将数据打乱放在这里看起来也行，不知道会不会增加时间
            train_sampler = RandomSampler(train_dataset) \
                                if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=args.train_batch_size,
                                          drop_last=True
                                          )
            pesudo_sampler = RandomSampler(pesudo_dataset) \
                                if args.local_rank == -1 else DistributedSampler(pesudo_dataset)
            pesudo_dataloader = DataLoader(pesudo_dataset,
                                           sampler=pesudo_sampler,
                                           batch_size=args.train_batch_size,
                                           drop_last=True
                                           )
            
            pesudo_epoch_iterator = tqdm(pesudo_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            args.save_steps = len(pesudo_epoch_iterator) * args.num_train_epochs
            # for step, batch in enumerate(epoch_iterator):
            for step, (batch, self_batch) in enumerate(
                zip(cycle(train_dataloader), pesudo_epoch_iterator)):
                # Skip past any already trained steps if resuming training

                model.train()

                labels_onehot = tensor2onehot(batch[5], args.rel_num).to(args.device)
                if using_soft_label:
                    labels_onehot_u = self_batch[5].to(args.device)
                else:
                    labels_onehot_u = tensor2onehot(self_batch[5], args.rel_num).to(args.device)
                # 这些对显存毫无影响。
                batch = tuple(t.to(args.device) for t in batch)
                self_batch = tuple(t.to(args.device) for t in self_batch)
                # if args.training_mode != 4:   如果因为这边输入会导致显存不够，那我们bag=None
                inputs = {"input_ids": batch[0],
                          "head_start_id": batch[1],
                          "tail_start_id": batch[2],
                          "attention_mask": batch[3],
                          "token_type_ids": batch[4],
                          "labels": batch[5],
                          "labels_onehot": labels_onehot,
                          "bag_input_ids": [batch[6], batch[12], batch[18]],
                          "bag_head_start_id": [batch[7], batch[13], batch[19]],
                          "bag_tail_start_id": [batch[8], batch[14], batch[20]],
                          "bag_attention_mask": [batch[9], batch[15], batch[21]],
                          "bag_token_type_ids": [batch[10], batch[16], batch[22]],
                          "input_ids_u": self_batch[0],
                          "head_start_id_u": self_batch[1],
                          "tail_start_id_u": self_batch[2],
                          "attention_mask_u": self_batch[3],
                          "token_type_ids_u": self_batch[4],
                          "labels_u": self_batch[5],
                          "labels_onehot_u": labels_onehot_u,
                          "bag_input_ids_u": [self_batch[6], self_batch[12], self_batch[18]],
                          "bag_head_start_id_u": [self_batch[7], self_batch[13], self_batch[19]],
                          "bag_tail_start_id_u": [self_batch[8], self_batch[14], self_batch[20]],
                          "bag_attention_mask_u": [self_batch[9], self_batch[15], self_batch[21]],
                          "bag_token_type_ids_u": [self_batch[10], self_batch[16], self_batch[22]],      # label其实共享就行
                          "training": True,
                          "current_epoch": epoch+1,     # 用于更新loss weight
                          }
                outputs = model(args, **inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                loss_x, loss_u, loss_xc, loss_uc = outputs[1:5]   # loss_c: consistency loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if loss_x is not None:
                        loss_x = loss_x.mean()
                    if loss_u is not None:
                        loss_u = loss_u.mean()
                    if loss_xc is not None:
                        loss_xc = loss_xc.mean()
                    if loss_uc is not None:
                        loss_uc = loss_uc.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    if loss_x is not None:
                        loss_x = loss_x / args.gradient_accumulation_steps
                    if loss_u is not None:
                        loss_u = loss_u / args.gradient_accumulation_steps
                    if loss_xc is not None:
                        loss_xc = loss_xc / args.gradient_accumulation_steps
                    if loss_uc is not None:
                        loss_uc = loss_uc / args.gradient_accumulation_steps

                # 是不是应该放在这里啊，要不然accumulation就没用了。
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if loss_x is not None:
                    tr_loss_x += loss_x.item()
                if loss_u is not None:
                    tr_loss_u += loss_u.item()
                if loss_xc is not None:
                    tr_loss_xc += loss_xc.item()
                if loss_uc is not None:
                    tr_loss_uc += loss_uc.item()

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

                        if loss_x is not None:
                            loss_x_scalar = (tr_loss_x - logging_loss_x) / args.logging_steps
                            logs["loss_x"] = loss_x_scalar
                            logging_loss_x = tr_loss_x

                        if loss_u is not None:
                            loss_u_scalar = (tr_loss_u - logging_loss_u) / args.logging_steps
                            logs["loss_u"] = loss_u_scalar
                            logging_loss_u = tr_loss_u

                        if loss_xc is not None:
                            loss_xc_scalar = (tr_loss_xc - logging_loss_xc) / args.logging_steps
                            logs["loss_xc"] = loss_xc_scalar
                            logging_loss_xc = tr_loss_xc

                        if loss_uc is not None:
                            loss_uc_scalar = (tr_loss_uc - logging_loss_uc) / args.logging_steps
                            logs["loss_uc"] = loss_uc_scalar
                            logging_loss_uc = tr_loss_uc

                        print(f"loss={loss}", 
                              f"loss_x={logging_loss_x} ",
                              f"loss_u={logging_loss_u} ",
                              f"loss_xc={logging_loss_xc} ",
                              f"loss_uc={logging_loss_uc} ")

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    # check_steps: select which steps to check model on val
                    if args.local_rank in [-1, 0] and global_step in check_steps:
                        val_results, _, _ = self.test(args, model, val_dataset)
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
                            test_results, _, _ = self.test(args, model, test_dataset)
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
                    pesudo_epoch_iterator.close()
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