"""
2021/8/4, Junjie Yu, Soochow University
Self-training with Merge.
Train a model to label unlabel data, and then merge high-probability data.
use low-probability data as first stage fine-tune.
then use above model to fine-tuning on (gold+high-probability data)
"""
import random
import os
import numpy as np
import torch
import logging
import json

from transformers import BertConfig, BertTokenizer 
# from transformers.modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model
from data_processor.inst_to_example_converter import SelfTrainingREDataProcessor
from data_processor.example_to_feature_converter import REFeatureConverter
from data_processor.feature_to_tensor_converter import RETensorConverter
from data_processor import never_split
from model.model_utils import ModelInit, set_seed
from model.bert_model import BertModelOutCLS, BertModelOutE1E2
from method.base import BaseTrainAndTest

try:
    from torch.utils.tensorboard import SummaryWriter   # 这个是新版本的
except ImportError:
    from tensorboardX import SummaryWriter  # 低版本仍旧用这个
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from utils.arguments import SelfTrainingREArguments
from method.self_training import Sampling, SelfTrainingTrainAndTest


# First, get all arguments
parser = SelfTrainingREArguments()
parser.add_arguments()
args = parser.parse_args()

# To ensure the output dir
if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({args.output_dir}) already exists and is not empty. "
        "Use --overwrite_output_dir to overcome."
        )
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    # Do this, and we use CUDA_VISIBLE_DEVICES to set cuda devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device
# Setup logging
logging.basicConfig(
    filename=f"{args.output_dir}/training.log",
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # 把info及以上等级的信息输出 NOSET<DEBUG<INFO<WARNING<ERROR<CRITICAL
)
logging.info(f"set seed: {args.seed}")
set_seed(args.seed, args.n_gpu)

# 也可以尝试给些参数
head_start_vocab_id, tail_start_vocab_id = 104, 106

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()
args.output_mode = 'classification'
logging.info("Training/evaluation parameters %s", args)

config_class = BertConfig
tokenizer_class = BertTokenizer
model_class = BertModelOutE1E2
model_init = ModelInit(args)

# 标注和未标注数据随着迭代会一直变化
train_dataset = None
train_insts = None
unlabel_dataset = None
unlabel_insts = None
train_distri_num = None
data_processor = SelfTrainingREDataProcessor(
    args.dataset_name, args.train_file, args.val_file, args.test_file,
    args.unlabel_file, args.label_file)
logging.info("Data processor", data_processor)

train_inst, train_bert = data_processor.get_train_examples()
val_inst, val_bert = data_processor.get_val_examples()
test_inst, test_bert = data_processor.get_test_examples()
unlabel_inst, unlabel_bert = data_processor.get_unlabel_examples()
label_list, label2id, id2label = data_processor.get_labels()

num_labels = len(label_list)
args.rel_num = num_labels
tokenizer = model_init.init_tokenizer(tokenizer_class, never_split=never_split.never_split)

logging.info("Convert BERT examples to BERT tensors")
feature_converter = REFeatureConverter(
    args, head_start_vocab_id, tail_start_vocab_id, label2id, tokenizer)

train_feature = feature_converter.convert_examples_to_features(train_bert, max_length=args.max_seq_length)
val_feature = feature_converter.convert_examples_to_features(val_bert, max_length=args.max_seq_length)
test_feature = feature_converter.convert_examples_to_features(test_bert, max_length=args.max_seq_length)
unlabel_feature = feature_converter.convert_examples_to_features(unlabel_bert, max_length=args.max_seq_length)

tensor_converter = RETensorConverter()
train_tensor = tensor_converter.feature_to_tensor(train_feature)
val_tensor = tensor_converter.feature_to_tensor(val_feature)
test_tensor = tensor_converter.feature_to_tensor(test_feature)
unlabel_tensor = tensor_converter.feature_to_tensor(unlabel_feature)

config = model_init.init_config(config_class, num_labels)
model = model_init.init_model(model_class, config)
trainer = SelfTrainingTrainAndTest(tokenizer)

checkpoint_name = 'best_val_checkpoint'
base_model_checkpoint_dir = os.path.join(args.base_model_dir, checkpoint_name)

if os.path.exists(base_model_checkpoint_dir):
    logging.info(f"Start loading model from checkpoint from {base_model_checkpoint_dir}")
    model = model_class.from_pretrained(base_model_checkpoint_dir)
    logging.info("Load baseline model.")
else:
    logging.info("Start training baseline")
    print("Start")
    print(base_model_checkpoint_dir)
    exit()  # Should be trained before
    trainer.train(args, model, train_tensor, val_tensor, test_tensor,
                  output_dir=base_model_checkpoint_dir)
    logging.info("End Training.")
    logging.info(f"Load the best checkpoint in {base_model_checkpoint_dir}")
    model = model_class.from_pretrained(base_model_checkpoint_dir)
    model.to(args.device)
    if args.local_rank in [-1, 0]:
        json_results = {}
        result, val_golds, val_preds = trainer.test(args, model, val_tensor)
        json_results['val'] = result

        result, test_golds, test_preds = trainer.test(args, model, test_tensor)
        json_results['test'] = result
        with open(os.path.join(args.base_model_output_dir, "results"), 'w') as writer:
            json.dump(json_results, writer, indent=2)

sampler = Sampling(id2label, train_inst)
model.to(args.device)
current_data_dir = args.base_model_dir
# 目前没有循环，若循环，则需考虑输出一个每一轮的train_inst file。
for self_train_epoch in range(args.self_train_epoch):
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
    # To get above data
    # Whether exist all labeled file
    if os.path.isfile(pseudo_all_file):
        logging.info(f"Read exist all pseudo file from {pseudo_all_file}")
        pseudo_all_inst, _ = data_processor.get_examples_from_file(pseudo_all_file)
    else:
        pseudo_all_inst = trainer.tagging(args, model, unlabel_tensor, unlabel_inst, id2label)
        data_processor.dump_data(pseudo_all_file, pseudo_all_inst)

    # Whether exist sampled file and unused file
    if os.path.isfile(pseudo_select_file) and os.path.isfile(pseudo_unused_file):
        logging.info(f"Read exist select pseudo file from {pseudo_select_file}")
        pseudo_select_inst, pseudo_select_bert = data_processor.get_examples_from_file(
            pseudo_select_file)
        logging.info(f"Read exist unused pseudo file from {pseudo_unused_file}")
        pseudo_unused_inst, pseudo_unused_bert = data_processor.get_examples_from_file(
            pseudo_unused_file)
    else:
        pseudo_select_inst, pseudo_unused_inst = sampler.sample_with_prob_threshold(
            pseudo_all_inst, args.prob_threshold)
        pseudo_select_bert = data_processor._create_examples_from_json_insert_entity_tag(pseudo_select_inst)
        pseudo_unused_bert = data_processor._create_examples_from_json_insert_entity_tag(pseudo_unused_inst)
        data_processor.dump_data(pseudo_select_file, pseudo_select_inst)
        data_processor.dump_data(pseudo_unused_file, pseudo_unused_inst)

    output_dir = os.path.join(args.output_dir, f"epoch_{str(self_train_epoch)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 调整下，要两个(first and second)都在才continue
    #if os.path.exists(os.path.join(output_dir, 'results')):  # and overwrite
        # 方便续跑
    #    continue
    # 每一轮的model都是从头开始的
    logging.info("Init model from BERT")
    model = model_init.init_model(model_class, config)
    logging.info(f"Start self-training epoch {self_train_epoch}")
    print(f"Start self-training epoch {self_train_epoch}")
    logging.info(f"output in {output_dir}")
    # First, use low-probability to fine-tune
    pseudo_unused_feature = feature_converter.convert_examples_to_features(pseudo_unused_bert, max_length=args.max_seq_length)
    pseudo_unused_tensor = tensor_converter.feature_to_tensor(pseudo_unused_feature)
    logging.info(f"Epoch {self_train_epoch}'s first training size={len(pseudo_unused_inst)}")
    checkpoint = os.path.join(output_dir, 'first', checkpoint_name)
    logging.info(f"Load the best checkpoint in {checkpoint}")
    if not os.path.exists(checkpoint):
        logging.info("Start training model")
        trainer.train(args, model, pseudo_unused_tensor, val_tensor, test_tensor,
                      output_dir=checkpoint, use_random_subset=args.use_random_subset,
                      learning_rate=args.learning_rate)
        logging.info("End Training.")

    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    if os.path.isdir(checkpoint) and args.delete_model:
        delete_checkpoint = f"rm -r {checkpoint}"
        logging.info("delete checkpoint to save disk")
        os.system(delete_checkpoint)

    result_file = os.path.join(output_dir, 'first', "results")
    if not os.path.exists(result_file):
        logging.info("Test the best model for first fine-tuned model")
        json_results = {}
        result, _, _ = trainer.test(args, model, val_tensor)
        json_results['val'] = result
        result, _, _ = trainer.test(args, model, test_tensor)
        json_results['test'] = result
        print(json_results)
        with open(result_file, 'w') as writer:
            json.dump(json_results, writer, indent=2)
    # Second, use merged gold and high-probability data to fine-tune
    train_inst += pseudo_select_inst
    train_bert += pseudo_select_bert
    logging.info(f"Epoch {self_train_epoch}'s second training size={len(train_inst)}")
    # Merge
    # 开始训练，直接调用训练baseline的方法
    train_feature = feature_converter.convert_examples_to_features(train_bert, max_length=args.max_seq_length)
    train_tensor = tensor_converter.feature_to_tensor(train_feature)
    checkpoint = os.path.join(output_dir, 'second', checkpoint_name)
    if not os.path.exists(checkpoint):
        logging.info("Start training model")
        trainer.train(args, model, train_tensor, val_tensor, test_tensor,
                      output_dir=checkpoint, use_random_subset=args.use_random_subset,
                      learning_rate=args.learning_rate_2)   #
        logging.info("End Training.")

    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    # 最终的测试，把结果存到文件里；当然，这个也可以直接放到train中，输出best_dev and corresponding test
    result_file = os.path.join(output_dir, 'second', "results")
    if not os.path.exists(result_file):
        logging.info("Test the best model for merge mode")
        json_results = {}
        result, _, _ = trainer.test(args, model, val_tensor)
        json_results['val'] = result
        result, _, _ = trainer.test(args, model, test_tensor)
        json_results['test'] = result
        print(json_results)
        with open(result_file, 'w') as writer:
            json.dump(json_results, writer, indent=2)
    # if iteration, we should to keep model is the last iteration, then use it to tagging.ß
    if os.path.isdir(checkpoint) and args.delete_model:
        delete_checkpoint = f"rm -r {checkpoint}"
        logging.info("delete checkpoint to save disk")
        os.system(delete_checkpoint)
    # update data dir
    current_data_dir = output_dir

