"""
2021/4/21, Junjie Yu, Soochow University
The baseline of labeled data
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
try:
    from torch.utils.tensorboard import SummaryWriter   # 这个是新版本的
except ImportError:
    from tensorboardX import SummaryWriter  # 低版本仍旧用这个

# First, get all arguments


parser = REArguments()
parser.parser.add_argument(
    "--bert_mode",
    default=None,
    type=str,
    required=True,
    help="cls, e1e2, or clse1e2.",
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

# 标注和未标注数据随着迭代会一直变化
train_dataset = None
train_insts = None
unlabel_dataset = None
unlabel_insts = None
train_distri_num = None
logging.info("Training/evaluation parameters %s", args)
logging.info("Load and convert original datasets")

data_processor = REDataProcessor(args.dataset_name,
                                 args.train_file,
                                 args.val_file,
                                 args.test_file,
                                 args.label_file)
# 读取原始数据，都不考虑cache什么的，每次直接搞
train_inst, train_bert = data_processor.get_train_examples()
val_inst, val_bert = data_processor.get_val_examples()
test_inst, test_bert = data_processor.get_test_examples()
label_list, label2id, id2label = data_processor.get_labels()
num_labels = len(label_list)
args.rel_num = num_labels

model_init = ModelInit(args)
tokenizer = model_init.init_tokenizer(tokenizer_class, never_split=never_split.never_split)

logging.info("Convert BERT examples to BERT tensors")
feature_converter = REFeatureConverter(
                                    args,
                                    head_start_vocab_id, tail_start_vocab_id,
                                    label2id, tokenizer)

train_feature = feature_converter.convert_examples_to_features(train_bert, max_length=args.max_seq_length)
val_feature = feature_converter.convert_examples_to_features(val_bert, max_length=args.max_seq_length)
test_feature = feature_converter.convert_examples_to_features(test_bert, max_length=args.max_seq_length)

# 将feature转换成tensor

tensor_converter = RETensorConverter()

# 如果是跑baseline的话，确实可以考虑将这个convert存起来
train_tensor = tensor_converter.feature_to_tensor(train_feature)
val_tensor = tensor_converter.feature_to_tensor(val_feature)
test_tensor = tensor_converter.feature_to_tensor(test_feature)

config = model_init.init_config(config_class, num_labels, None)
model = model_init.init_model(model_class, config)
trainer = BaseTrainAndTest(tokenizer)

checkpoint_name = 'best_val_checkpoint'
result_file = os.path.join(args.output_dir, "results")
# 这边应该是判断是否训练了baseline，训练好了直接用，没有就训练一个
checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
if os.path.exists(checkpoint_dir) and os.path.isfile(result_file):
    print("Exist checkpoint and result file, SKIP.")
    exit()
logging.info("Start training baseline")
trainer.train(args, model, train_tensor, val_tensor, test_tensor, output_dir=checkpoint_dir,
              use_random_subset=args.use_random_subset)
logging.info("End Training.")
logging.info(f"Load the best checkpoint in {checkpoint_dir}")
model = model_class.from_pretrained(checkpoint_dir)
model.to(args.device)
if args.local_rank in [-1, 0]:
    json_results = {}
    result, val_golds, val_preds = trainer.test(args, model, val_tensor)
    json_results['val'] = result
    result, test_golds, test_preds = trainer.test(args, model, test_tensor)
    json_results['test'] = result
    with open(os.path.join(args.output_dir, "results"), 'w') as writer:
        json.dump(json_results, writer, indent=2)
