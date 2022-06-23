import argparse


class REArguments():
    """太复杂了，作为base arguments，应该想办法精简，而其他参数都放到对应的方法的class中。
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        return self.parser.parse_args()

    def add_arguments(self):
        self.parser.add_argument(
            "--use_lr_scheduler", action="store_true",
            help="是否使用学习率下降，小数据实验不使用，方便epoch设定.",
        )
        self.parser.add_argument("--use_random_subset",
                                 action="store_true",
                                 help="Whether to test with a random subset of original dataset, save time")
        self.parser.add_argument(
            "--train_file",
            default=None,
            type=str,
            required=True,
            help="train file path and name.",
        )
        self.parser.add_argument(
            "--val_file",
            default=None,
            type=str,
            required=True,
            help="val file path and name.",
        )
        self.parser.add_argument(
            "--test_file",
            default=None,
            type=str,
            required=True,
            help="test file path and name.",
        )

        self.parser.add_argument(
            "--label_file",
            default=None,
            type=str,
            required=True,
            help="relation label file path and name.",
        )
        self.parser.add_argument(
            "--model_type",
            default="bert",
            type=str,
            required=True,
            help="Model type selected, always bert now",
        )
        self.parser.add_argument(
            "--data_type",
            default="sdp",
            type=str,
            required=False,
            help="sdp middle or others",
        )
        self.parser.add_argument(
            "--data_types",
            default="sdp|sdp_k1|middle",
            type=str,
            required=False,
            help="sdp middle or others",
        )
        self.parser.add_argument(
            "--insert_tag",
            default="yes",
            type=str,
            required=False,
            help="yes or no",
        )
        self.parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
        )
        self.parser.add_argument(
            "--dataset_name",
            default=None,
            type=str,
            required=True,
            help="The name of the task to train selected in the list: [tacred, semeval]",
        )
        self.parser.add_argument(
            "--method_name",
            default=None,
            type=str,
            required=False,
            help="baseline, insert_type ... ",
        )
        self.parser.add_argument(
            "--cache_feature_file_dir",
            default=None,
            type=str,
            required=True,
            help="The cached directory where the features of train/dev/test will be written.",
        )
        self.parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )

        # Other parameters
        self.parser.add_argument(
            "--model_dir",
            default=None,
            type=str,
            help="Used for evaluation. The directory where the model predictions and checkpoints will be written.",
        )
        self.parser.add_argument(
            "--few_shot",
            default=None,
            type=str,
            help="5, 10, 20"
        )
        self.parser.add_argument(
            "--piecewise",
            default=None,
            type=str,
            help="2e: 2 entity; 2e1r: 2 entity 1 relation; 2e3r: 2 entity 3 relation; 3s: 3 segment"
        )
        self.parser.add_argument(
            "--pooling_mode",
            default=None,
            type=str,
            help="self-att, max, avg"
        )

        self.parser.add_argument(
            "--type_source",
            default=None,
            type=str,
            help="gold or auto",
        )
        self.parser.add_argument(
            "--ent_type_num",
            default=17,
            type=int,
            help="The number of all entity types (for NER not 17 for TACRED)",            
        )
        self.parser.add_argument(
            "--ent_type_emb_size",
            default=128,
            type=int,
            help="The hidden size of entity type embedding. \
                Random initialized or load from a word2vec trained embedding. \
                The size should be matched.",            
        )

        self.parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
        )
        self.parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        self.parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        self.parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        self.parser.add_argument("--without_NA", action="store_true", help="Whether NA is in relations")
        self.parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        self.parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        self.parser.add_argument(
            "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
        )
        self.parser.add_argument(
            "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
        )

        self.parser.add_argument(
            "--delete_model", action="store_true", help="Delete final model to save disk",
        )

        # For small data
        self.parser.add_argument(
            "--use_small_data", action="store_true", help="Whether to use small data as training data",
        )
        self.parser.add_argument(
            "--small_data_type",
            default="rate_0.1",
            type=str,
        )
        self.parser.add_argument(
            "--micro_f1", action="store_true", help="Use micro f1",
        )
        self.parser.add_argument(
            "--macro_f1", action="store_true", help="Use macro f1",
        )

        self.parser.add_argument(
            "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
        )
        self.parser.add_argument(
            "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
        )
        self.parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        self.parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
        self.parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument(
            "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.",
        )
        self.parser.add_argument(
            "--max_steps",
            default=-1,
            type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )
        self.parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

        self.parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
        self.parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
        self.parser.add_argument("--max_save_times", type=int, default=10, help="number to do checkpoint.")

        self.parser.add_argument("--no_improve_num", type=int, default=5, help="!!")
        self.parser.add_argument(
            "--eval_all_checkpoints",
            action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
        )
        self.parser.add_argument(
            "--eval_all_epoch_checkpoints",
            action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with epoch number",
        )
        self.parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        self.parser.add_argument(
            "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
        )
        self.parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
        )
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

        self.parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        self.parser.add_argument(
            "--fp16_opt_level",
            type=str,
            default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",
        )
        self.parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        self.parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
        self.parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

        self.parser.add_argument(
            "--use_cache", action="store_true", help="Use the cached pseudo example files",
        )
        self.parser.add_argument(
            "--use_bce", action="store_true", help="Use bce loss",
        )


class SelfTrainingREArguments(REArguments):
    def __init__(self):
        super().__init__()

    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument(
            "--unlabel_file",
            default=None,
            type=str,
            help="unlabeled file path and name.",
            required=True
        )
        self.parser.add_argument(
            "--base_model_dir",
            default=None,
            type=str,
            required=True,
            help="where the checkpoint of base model. self-training method need to load/train base model",
        )
        self.parser.add_argument('--easy_prob_threshold', default=0.90, type=float, help="threshold for high-probability pseudo data")
        self.parser.add_argument('--ambig_prob_threshold', default=0.95, type=float, help="threshold for ambig-probability pseudo data")
        self.parser.add_argument('--ambiguity_prob', default=0.1, type=float, help="threshold for low-probability pseudo data get its ambiguity label")
        self.parser.add_argument('--self_train_epoch', default=1, type=int, help="for iteration self-training")
        self.parser.add_argument('--max_label_num', default=1, type=int, help="select top_N probs as labels")
        self.parser.add_argument("--learning_rate_2", default=5e-5, type=float, help="The initial learning rate for Adam.")
        self.parser.add_argument("--drop_NA", action="store_true", help="Whether to drop data with no_relation label in top or topN during ambiguity labeling.")
        self.parser.add_argument("--rampup", action="store_true", help="Whether to use rampup on loss when training on ambiguity labeling.")
        self.parser.add_argument("--do_softmin", action="store_true", help="Whether to do softmin based loss update for ambiguous data.")
        self.parser.add_argument(
            "--ambiguity_mode",
            default='sum',
            type=str,
            help="sum or max or min",
        )
        self.parser.add_argument('--alpha', default=1.0, type=float, help="weight for loss")


class MultiTaskLearningREArguments(REArguments):
    def __init__(self):
        super().__init__()

    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument(
            "--extra_train_file",
            default=None,
            type=str,
            required=True,
            help="extra train file path and name.",
        )
        self.parser.add_argument(
            "--extra_val_file",
            default=None,
            type=str,
            required=True,
            help="extra val file path and name.",
        )
        self.parser.add_argument(
            "--extra_test_file",
            default=None,
            type=str,
            required=True,
            help="extra test file path and name.",
        )

        self.parser.add_argument(
            "--extra_label_file",
            default=None,
            type=str,
            required=True,
            help="extra relation label file path and name.",
        )
