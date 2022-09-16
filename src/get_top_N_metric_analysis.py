"""
2022/08/23, Junjie Yu, Soochow University
Test and Tagging the test set with prob distribution, 
hence, we can evaluate the top-N.
We compare the base model, self-training model, and our PartialNegative Model
"""

import os
import torch
import logging
import json
os.environ['MKL_THREADING_LAYER'] = 'GNU'   # For a mkl-service problem

from torch.utils.data import TensorDataset

from transformers import BertConfig, BertTokenizer
from utils.arguments import REArguments
from data_processor.inst_to_example_converter import REDataProcessor
from data_processor.example_to_feature_converter import REFeatureConverter
from data_processor.feature_to_tensor_converter import RETensorConverter
from data_processor.data_loader_and_dumper import JsonDataDumper
from data_processor import never_split
from model.model_utils import ModelInit, set_seed
from model.bert_model import BertModelOutCLS, BertModelOutE1E2, BertModelOutCLSE1E2, BertModelOutE1E2Positive
from method.base import BaseTrainAndTest
from method.self_training import Sampling, SelfTrainingTrainAndTest
from model.bert_model import BertModelOutCLS, BertModelOutE1E2PositiveAndNegative
try:
    from torch.utils.tensorboard import SummaryWriter   # 这个是新版本的
except ImportError:
    from tensorboardX import SummaryWriter  # 低版本仍旧用这个
# First, get all arguments
from utils.compute_metrics import EvaluationAndAnalysis

parser = REArguments()
parser.parser.add_argument(
    "--bert_mode",
    default=None,
    type=str,
    required=True,
    help="cls, e1e2, or clse1e2.",
)

parser.parser.add_argument(
    "--topN_metric",
    default=2,
    type=int,
    required=True,
    help="performance of top N preds",
)
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

config_class = BertConfig
tokenizer_class = BertTokenizer

print(args.bert_mode)
if args.bert_mode == "cls":
    model_class = BertModelOutCLS
elif args.bert_mode == "e1e2":
    model_class = BertModelOutE1E2
elif args.bert_mode == "clse1e2":
    model_class = BertModelOutCLSE1E2
else:
    print(f"Error output mode: {args.bert_mode}")
    exit()

model_class = BertModelOutE1E2Positive

data_processor = REDataProcessor(args.dataset_name,
                                args.train_file,
                                args.val_file,
                                args.test_file,
                                args.label_file)
# 读取原始数据，都不考虑cache什么的，每次直接搞
val_inst, val_bert = data_processor.get_val_examples()
test_inst, test_bert = data_processor.get_test_examples()
label_list, label2id, id2label = data_processor.get_labels()
num_labels = len(label_list)
args.rel_num = num_labels

test_prediction_file = os.path.join(args.output_dir, "test_tagging_result.txt")
test_prediction_inst = []
if not os.path.exists(test_prediction_file):
    model_init = ModelInit(args)
    tokenizer = model_init.init_tokenizer(tokenizer_class, never_split=never_split.never_split)

    logging.info("Convert BERT examples to BERT tensors")
    feature_converter = REFeatureConverter(
                                        args,
                                        head_start_vocab_id, tail_start_vocab_id,
                                        label2id, tokenizer)

    val_feature = feature_converter.convert_examples_to_features(val_bert, max_length=args.max_seq_length)
    test_feature = feature_converter.convert_examples_to_features(test_bert, max_length=args.max_seq_length)

    # 将feature转换成tensor

    tensor_converter = RETensorConverter()

    # 如果是跑baseline的话，确实可以考虑将这个convert存起来
    val_tensor = tensor_converter.feature_to_tensor(val_feature)
    test_tensor = tensor_converter.feature_to_tensor(test_feature)

    config = model_init.init_config(config_class, num_labels, None)
    model = model_init.init_model(model_class, config)
    trainer = SelfTrainingTrainAndTest(tokenizer)

    checkpoint_name = 'best_val_checkpoint'
    result_file = os.path.join(args.output_dir, "results")
    # Load the baseline model
    ckpt_dir = os.path.join(args.output_dir, checkpoint_name)
    logging.info(f"Load the best checkpoint in {ckpt_dir}")
    model = model_class.from_pretrained(ckpt_dir)
    model.to(args.device)
    test_prediction_inst = trainer.tagging(args, model, test_tensor, test_inst, id2label)
    data_processor.dump_data(test_prediction_file, test_prediction_inst)
else:
    print(test_prediction_file)
    with open(test_prediction_file) as reader:
        for line in reader.readlines():
            test_prediction_inst.append(json.loads(line))
print(len(test_prediction_inst))

# Start evaluate top-N
# Get preds and labels
# As top-N, we search top-N preds, if one of it matches answer, we select the answer,
# Otherwise we use the top-1 wrong prediction
N = args.topN_metric
labels = []
preds = []
for inst in test_prediction_inst:
    label = label2id[inst['gold_relation']]
    distribution = inst['distri']
    topN_preds = sorted(range(len(distribution)), key=lambda i: distribution[i], reverse=True)[:N]
    labels.append(label)
    if label in topN_preds:
        preds.append(label)
    else:
        preds.append(topN_preds[0]) # if all False, add the top1 false

analysis = EvaluationAndAnalysis()
if args.dataset_name == "semeval":
    result = analysis.micro_f1_exclude_NA(labels, preds)
elif args.dataset_name == "re-tacred_exclude_NA":
    if args.micro_f1:
        result = analysis.micro_f1(labels, preds)    # each relation is calculated

print(result['micro_f1'])

# write the result
with open(os.path.join(args.output_dir, f"top{N}_results.txt"), 'w') as writer:
    json.dump(result, writer, indent=2)