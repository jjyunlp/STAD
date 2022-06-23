import logging
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import TensorDataset
import torch

logger = logging.getLogger(__name__)


class REInputFeatures(InputFeatures):
    """See base class."""
    def __init__(self, input_ids, head_start_id, tail_start_id,
                 attention_mask=None, token_type_ids=None, label=None,
                 aux_mask_a=None, aux_mask_b=None, aux_mask_c=None,
                 em_input_ids=None, em_attention_mask=None, em_token_type_ids=None,
                 onehot_label=None):
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.head_start_id = head_start_id
        self.tail_start_id = tail_start_id
        self.aux_mask_a = aux_mask_a
        self.aux_mask_b = aux_mask_b
        self.aux_mask_c = aux_mask_c
        # 下面三个用于cls 输出的entity mask的句子
        self.em_input_ids = em_input_ids
        self.em_attention_mask = em_attention_mask
        self.em_token_type_ids = em_token_type_ids

        self.onehot_label = onehot_label
        # 表明当前实例是否为partial labeled data
        # 因为soft label的sum=1，因此不必修改
        if self.onehot_label is not None:
            if sum(self.onehot_label) == 1:
                self.flag = 1
            else:
                self.flag = 0


class REFeatureConverter(object):
    """
    base example to feature class
    Convert examples to features for model.
    (1) All examples from different data (like TACRED or SemEval-2010) will be 
    processed in the same way.
    (2) Used for baseline model and any other models (e.g, add entity type).
    """
    def __init__(self, args,
                 head_start_vocab_id,
                 tail_start_vocab_id,
                 label2id,
                 tokenizer):
        """
        head/tail_start_vocab_id: the id of head token in vocab.txt of BERT
            like [E1] is 104. As we may insert different tags as boundary of 
            entities, we will provide this information at data process.
        """
        print("Invoke convert example to features")
        self.args = args
        self.head_start_vocab_id = head_start_vocab_id
        self.tail_start_vocab_id = tail_start_vocab_id
        self.label2id = label2id
        self.tokenizer = tokenizer
        

    def get_tag_id(
            self,
            input_ids,
            input_vocab_id,
            subj_first=None,
            ):
        """得到插入在实体前的标志的位置，上面的写的就不够base，为啥要两个一起。。"""
        location = 0  # if not exsit, e.g., oversize
        # (1) ... [E1] .. [/E1] ... [E2] .. [/E2] ...
        if input_vocab_id in input_ids:
            location = input_ids.index(input_vocab_id)
        return location
        
    def convert_examples_to_features(
        self,
        examples,
        max_length=512,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        using_soft_label=False,
    ):
        """Only for labeled examples

        Args:
            examples ([type]): [description]
            max_length (int, optional): [description]. Defaults to 512.
            pad_on_left (bool, optional): [description]. Defaults to False.
            pad_token (int, optional): [description]. Defaults to 0.
            pad_token_segment_id (int, optional): [description]. Defaults to 0.
            mask_padding_with_zero (bool, optional): [description]. Defaults to True.
            using_soft_label (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Write example %d/%d" % (ex_index, len_examples))
            # encode_plus is defined in tokenization_utils.py and return a dictionary
            # Not word piece tokenizer (like ##ing)
            inputs = self.tokenizer.encode_plus(example.text_a,
                                                example.text_b,
                                                add_special_tokens=True,
                                                max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # Find entity start tag (index=1([e1]) and 3([e2]))
            head_start_id = self.get_tag_id(input_ids, self.head_start_vocab_id)
            tail_start_id = self.get_tag_id(input_ids, self.tail_start_vocab_id)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) \
                    + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # 确认下，上述各个信息生成没有问题（从长度角度）
            assert len(input_ids) == max_length, \
                f"Error with input length {len(input_ids)} vs {max_length}"
            assert len(attention_mask) == max_length, \
                "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, \
                "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            if not using_soft_label:
                label = self.label2id[example.label]
            else:
                label = example.label
            if ex_index < 5:
                # 我还是希望这边能输出原本的文本的，这样对应着看也容易发现bug
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                if using_soft_label:
                    logger.info("soft label: %s" % " ".join([str(x) for x in label]))
                else:
                    logger.info("label: %s" % label)

            features.append(
                REInputFeatures(
                    input_ids=input_ids,
                    head_start_id=head_start_id,
                    tail_start_id=tail_start_id,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label
                    )
            )

        return features


class SelfTrainingREFeatureConverter(object):
    """
    base example to feature class
    Convert examples to features for model.
    (1) All examples from different data (like TACRED or SemEval-2010) will be 
    processed in the same way.
    (2) Used for baseline model and any other models (e.g, add entity type).
    """
    def __init__(self, args,
                 head_start_vocab_id,
                 tail_start_vocab_id,
                 label2id,
                 tokenizer):
        """
        head/tail_start_vocab_id: the id of head token in vocab.txt of BERT
            like [E1] is 104. As we may insert different tags as boundary of 
            entities, we will provide this information at data process.
        """
        print("Invoke convert example to features")
        self.args = args
        self.head_start_vocab_id = head_start_vocab_id
        self.tail_start_vocab_id = tail_start_vocab_id
        self.label2id = label2id
        self.tokenizer = tokenizer
        

    def get_tag_id(
            self,
            input_ids,
            input_vocab_id,
            subj_first=None,
            ):
        """得到插入在实体前的标志的位置，上面的写的就不够base，为啥要两个一起。。"""
        location = 0  # if not exsit, e.g., oversize
        # (1) ... [E1] .. [/E1] ... [E2] .. [/E2] ...
        if input_vocab_id in input_ids:
            location = input_ids.index(input_vocab_id)
        return location
        
    def convert_examples_to_features(
        self,
        examples,
        max_length=512,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        using_soft_label=False,
        avg_partial_labels=False,   # [1, 1, 0, 0] -> [1/2, 1/2, 0, 0]
    ):
        """Only for labeled examples

        Args:
            examples ([type]): [description]
            max_length (int, optional): [description]. Defaults to 512.
            pad_on_left (bool, optional): [description]. Defaults to False.
            pad_token (int, optional): [description]. Defaults to 0.
            pad_token_segment_id (int, optional): [description]. Defaults to 0.
            mask_padding_with_zero (bool, optional): [description]. Defaults to True.
            using_soft_label (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Write example %d/%d" % (ex_index, len_examples))
            # encode_plus is defined in tokenization_utils.py and return a dictionary
            # Not word piece tokenizer (like ##ing)
            inputs = self.tokenizer.encode_plus(example.text_a,
                                                example.text_b,
                                                add_special_tokens=True,
                                                max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # Find entity start tag (index=1([e1]) and 3([e2]))
            head_start_id = self.get_tag_id(input_ids, self.head_start_vocab_id)
            tail_start_id = self.get_tag_id(input_ids, self.tail_start_vocab_id)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) \
                    + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # 确认下，上述各个信息生成没有问题（从长度角度）
            assert len(input_ids) == max_length, \
                f"Error with input length {len(input_ids)} vs {max_length}"
            assert len(attention_mask) == max_length, \
                "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, \
                "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            if not using_soft_label:
                label = self.label2id[example.label]
            else:
                label = example.label
            onehot_label = example.onehot_label
            if avg_partial_labels:
                total = sum(onehot_label)
                onehot_label = [x/total for x in onehot_label]
            if ex_index < 5:
                # 我还是希望这边能输出原本的文本的，这样对应着看也容易发现bug
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                if using_soft_label:
                    logger.info("soft label: %s" % " ".join([str(x) for x in label]))
                else:
                    logger.info("label: %s" % label)
                if example.onehot_label is not None:    # unlabel data is none
                    logger.info("onehot label: %s" % " ".join([str(x) for x in onehot_label]))

            features.append(
                REInputFeatures(
                    input_ids=input_ids,
                    head_start_id=head_start_id,
                    tail_start_id=tail_start_id,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                    onehot_label=onehot_label
                    )
            )

        return features

class REBagFeatureConverter():
    """
    TODO
    """
    def __init__(self, args,
                 head_start_vocab_id,
                 tail_start_vocab_id,
                 label2id,
                 tokenizer):
        """
        head/tail_start_vocab_id: the id of head token in vocab.txt of BERT
            like [E1] is 104. As we may insert different tags as boundary of 
            entities, we will provide this information at data process.
        """
        print("Invoke convert example to features")
        self.args = args
        self.head_start_vocab_id = head_start_vocab_id
        self.tail_start_vocab_id = tail_start_vocab_id
        self.label2id = label2id
        self.tokenizer = tokenizer
        
    def feature_to_tensor(self, features, using_soft_label=False, no_label=False):
        """将bert features转换成tensor
        using_soft_label：用于pesudo data时使用soft label，因此label是一个float格式的概率分布
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_head_start_id = torch.tensor([f.head_start_id for f in features], dtype=torch.long)
        # all_head_start_id = torch.tensor([f.head_start_id for f in bags for bags in features], dtype=torch.long)
        all_tail_start_id = torch.tensor([f.tail_start_id for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if not no_label:
            if using_soft_label:
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
            else:
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            feature_dataset = TensorDataset(all_input_ids,
                                            all_head_start_id,
                                            all_tail_start_id,
                                            all_attention_mask,
                                            all_token_type_ids,
                                            all_labels,
                                            )
        else:
            feature_dataset = TensorDataset(all_input_ids,
                                            all_head_start_id,
                                            all_tail_start_id,
                                            all_attention_mask,
                                            all_token_type_ids,
                                            )
        return feature_dataset

    def bag_feature_to_bag_tensor(self, bag_features, using_soft_label=False, bag_num=4):
        # 这种都应该放到子类中
        """将bert features转换成tensor
        using_soft_label：用于pesudo data时使用soft label，因此label是一个float格式的概率分布
        """
        bag_tensors = []
        for i in range(bag_num):  # original + google + baidu + xiaoniu  
            # 这边没问题，可以去看convert_bag_examples_to_features,label 并没有搞成bag
            all_input_ids = torch.tensor([f.input_ids[i] for f in bag_features], dtype=torch.long)
            all_head_start_id = torch.tensor([f.head_start_id[i] for f in bag_features], dtype=torch.long)
            # all_head_start_id = torch.tensor([f.head_start_id for f in bags for bags in features], dtype=torch.long)
            all_tail_start_id = torch.tensor([f.tail_start_id[i] for f in bag_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask[i] for f in bag_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids[i] for f in bag_features], dtype=torch.long)
            # label只有一个，不是bag
            if using_soft_label:
                all_labels = torch.tensor([f.label for f in bag_features], dtype=torch.float)
            else:
                all_labels = torch.tensor([f.label for f in bag_features], dtype=torch.long)

            bag_tensors += [all_input_ids, all_head_start_id, all_tail_start_id, all_attention_mask, all_token_type_ids, all_labels]
        feature_dataset = TensorDataset(*bag_tensors)
        return feature_dataset

    def feature_with_aux_to_tensor(self, features, using_soft_label=False):
        """将bert features转换成tensor
        using_soft_label：用于pesudo data时使用soft label，因此label是一个float格式的概率分布
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_head_start_id = torch.tensor([f.head_start_id for f in features], dtype=torch.long)
        # all_head_start_id = torch.tensor([f.head_start_id for f in bags for bags in features], dtype=torch.long)
        all_tail_start_id = torch.tensor([f.tail_start_id for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_aux_mask_a = torch.tensor([f.aux_mask_a for f in features], dtype=torch.long)
        all_aux_mask_b = torch.tensor([f.aux_mask_b for f in features], dtype=torch.long)
        all_aux_mask_c = torch.tensor([f.aux_mask_c for f in features], dtype=torch.long)
        if using_soft_label:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        else:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        feature_dataset = TensorDataset(all_input_ids,
                                        all_head_start_id,
                                        all_tail_start_id,
                                        all_attention_mask,
                                        all_token_type_ids,
                                        all_labels,
                                        all_aux_mask_a,
                                        all_aux_mask_b,
                                        all_aux_mask_c,
                                        )
        return feature_dataset

    def get_start_tag_id(
            self,
            method_name, 
            input_ids, 
            subj_first=None, 
            entity_start_tag_vocab_ids=None
            ):
        """得到插入在实体前的标志的位置"""
        head_start_id, tail_start_id = 0, 0  # if not exsit, e.g., oversize
        #(1) ... [E1] .. [/E1] ... [E2] .. [/E2] ...
        if self.head_start_vocab_id in input_ids:
            head_start_id = input_ids.index(self.head_start_vocab_id)
        if self.tail_start_vocab_id in input_ids:
            tail_start_id = input_ids.index(self.tail_start_vocab_id)            
        return (head_start_id, tail_start_id)

    def get_tag_id(
            self,
            input_ids,
            input_vocab_id,
            subj_first=None,
            ):
        """得到插入在实体前的标志的位置，上面的写的就不够base，为啥要两个一起。。"""
        location = 0  # if not exsit, e.g., oversize
        # (1) ... [E1] .. [/E1] ... [E2] .. [/E2] ...
        if input_vocab_id in input_ids:
            location = input_ids.index(input_vocab_id)
        return location
        
    def convert_examples_to_features(
        self,
        examples,
        max_length=512,
        task=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        using_soft_label=False,
    ):
        """
        Loads a data file into a list of ``InputFeatures``

        Args:
            using_soft_label: 用了，那输入的label直接是一个float的list，不需要再做转换。否则是个string
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing 
                the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using
                the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression``
                or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the 
                left rather than on the right (default)
            pad_token: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is 
                usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will 
                be filled by ``1`` for actual values and by ``0`` for padded values. 
                If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        max_length = self.args.max_seq_length
        # 分类标签体系保持了前后一致，根据预先设定的json文件中关系与id的对应，得到一个
        # 按照序号排列的列表。
        output_mode = "classification"
        logger.info("Using output mode %s for task %s" % (output_mode, task))

        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Write example %d/%d" % (ex_index, len_examples))
            # encode_plus is defined in tokenization_utils.py and return a dictionary
            # Not word piece tokenizer (like ##ing)
            inputs = self.tokenizer.encode_plus(example.text_a,
                                                example.text_b,
                                                add_special_tokens=True,
                                                max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # Find entity start tag (index=1([e1]) and 3([e2]))
            head_start_id, tail_start_id = self.get_start_tag_id(
                self.args.method_name, input_ids, self.head_start_vocab_id, self.tail_start_vocab_id
            )
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) \
                    + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # 确认下，上述各个信息生成没有问题（从长度角度）
            assert len(input_ids) == max_length, \
                f"Error with input length {len(input_ids)} vs {max_length}"
            assert len(attention_mask) == max_length, \
                "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, \
                "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            if not using_soft_label:
                label = self.label2id[example.label]
            else:
                label = example.label
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                if using_soft_label:
                    logger.info("soft label: %s" % " ".join([str(x) for x in label]))
                else:
                    logger.info("label: %s" % label)

            features.append(
                REInputFeatures(
                    input_ids=input_ids,
                    head_start_id=head_start_id,
                    tail_start_id=tail_start_id,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label
                    )
            )

        return features

    def convert_examples_to_features_for_cls_e1_e2_alone(
        self,
        examples,
        max_length=512,
        task=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        using_soft_label=False,
    ):
        """text_a是原句，text_b是entity mask 的句子

        """
        max_length = self.args.max_seq_length
        # 分类标签体系保持了前后一致，根据预先设定的json文件中关系与id的对应，得到一个
        # 按照序号排列的列表。
        output_mode = "classification"
        logger.info("Using output mode %s for task %s" % (output_mode, task))

        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Write example %d/%d" % (ex_index, len_examples))
            # encode_plus is defined in tokenization_utils.py and return a dictionary
            # Not word piece tokenizer (like ##ing)
            inputs = self.tokenizer.encode_plus(example.text_a,
                                                None,
                                                add_special_tokens=True,
                                                max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # Find entity start tag (index=1([e1]) and 3([e2]))
            head_start_id, tail_start_id = self.get_start_tag_id(
                self.args.method_name, input_ids, self.head_start_vocab_id, self.tail_start_vocab_id
            )
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) \
                    + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            
            # 使用entity mask的text_b生成特征
            em_inputs = self.tokenizer.encode_plus(example.text_b,
                                                   None,
                                                   add_special_tokens=True,
                                                   max_length=max_length)
            em_input_ids, em_token_type_ids = em_inputs["input_ids"], em_inputs["token_type_ids"]
            """
            需要修改，生成head tail的id.
            现在使用[CLS]，因此不需要
            # Find entity start tag (index=1([e1]) and 3([e2]))
            em_head_start_id, em_tail_start_id = self.get_start_tag_id(
                self.args.method_name, em_input_ids, self.head_start_vocab_id, self.tail_start_vocab_id
            )
            """
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            em_attention_mask = [1 if mask_padding_with_zero else 0] * len(em_input_ids)
            # Zero-pad up to the sequence length.
            em_padding_length = max_length - len(em_input_ids)
            if pad_on_left:
                em_input_ids = ([pad_token] * em_padding_length) + em_input_ids
                em_attention_mask = ([0 if mask_padding_with_zero else 1] * em_padding_length) \
                    + em_attention_mask
                em_token_type_ids = ([pad_token_segment_id] * em_padding_length) + em_token_type_ids
            else:
                em_input_ids = em_input_ids + ([pad_token] * em_padding_length)
                em_attention_mask = em_attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * em_padding_length)
                em_token_type_ids = em_token_type_ids + ([pad_token_segment_id] * em_padding_length)

            # 确认下，上述各个信息生成没有问题（从长度角度）
            assert len(input_ids) == max_length, \
                f"Error with input length {len(input_ids)} vs {max_length}"
            assert len(attention_mask) == max_length, \
                "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, \
                "Error with input length {} vs {}".format(len(token_type_ids), max_length)
            assert len(em_input_ids) == max_length, \
                f"Error with input length {len(em_input_ids)} vs {max_length}"
            assert len(em_attention_mask) == max_length, \
                "Error with input length {} vs {}".format(len(em_attention_mask), max_length)
            assert len(em_token_type_ids) == max_length, \
                "Error with input length {} vs {}".format(len(em_token_type_ids), max_length)

            if not using_soft_label:
                label = self.label2id[example.label]
            else:
                label = example.label
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("em_input_ids: %s" % " ".join([str(x) for x in em_input_ids]))
                logger.info("em_attention_mask: %s" % " ".join([str(x) for x in em_attention_mask]))
                logger.info("em_token_type_ids: %s" % " ".join([str(x) for x in em_token_type_ids]))
                if using_soft_label:
                    logger.info("soft label: %s" % " ".join([str(x) for x in label]))
                else:
                    logger.info("label: %s" % label)

            features.append(
                REInputFeatures(
                    input_ids=input_ids,
                    head_start_id=head_start_id,
                    tail_start_id=tail_start_id,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    em_input_ids=em_input_ids,
                    em_attention_mask=em_attention_mask,
                    em_token_type_ids=em_token_type_ids,
                    label=label
                    )
            )

        return features

    def convert_examples_with_aux_to_features(
        self,
        examples,
        max_length=512,
        task=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        using_soft_label=False,
    ):
        """
        根据句子中实体的位置，输入三个mask，方便将需要遮盖的地方padding，从而方向实现auxiliary input
        """
        max_length = self.args.max_seq_length
        # 分类标签体系保持了前后一致，根据预先设定的json文件中关系与id的对应，得到一个
        # 按照序号排列的列表。
        output_mode = "classification"
        logger.info("Using output mode %s for task %s" % (output_mode, task))

        features = []
        tail_end_vocab_id = 107     # 先固定着用了。。。对于tacred，还需要判断两实体的前后
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Write example %d/%d" % (ex_index, len_examples))
            # encode_plus is defined in tokenization_utils.py and return a dictionary
            # Not word piece tokenizer (like ##ing)
            inputs = self.tokenizer.encode_plus(example.text_a,
                                                example.text_b,
                                                add_special_tokens=True,
                                                max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # Find entity start tag (index=1([e1]) and 3([e2]))
            head_start_id, tail_start_id = self.get_start_tag_id(
                self.args.method_name, input_ids, self.head_start_vocab_id, self.tail_start_vocab_id
            )
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            tail_end_id = self.get_tag_id(input_ids, tail_end_vocab_id)
            # 生成aux mask
            x_len = head_start_id + 1   # start from 0, so length should add 1
            y_len = len(input_ids) - tail_end_id - 1
            aux_mask_a = [0] * x_len + [1] * (len(input_ids) - x_len) + [0] * padding_length
            aux_mask_b = [0] * x_len + [1] * (len(input_ids) - x_len - y_len) + [0] * y_len + [0] * padding_length
            aux_mask_c = [1] * (len(input_ids) - y_len) + [0] * y_len + [0] * padding_length

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) \
                    + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)



            # 确认下，上述各个信息生成没有问题（从长度角度）
            assert len(input_ids) == max_length, \
                f"Error with input length {len(input_ids)} vs {max_length}"
            assert len(attention_mask) == max_length, \
                "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, \
                "Error with input length {} vs {}".format(len(token_type_ids), max_length)
            assert len(aux_mask_a) == max_length, \
                "Error with input length {} vs {} for aux_mask_a".format(len(aux_mask_b), max_length)
            assert len(aux_mask_b) == max_length, \
                "Error with input length {} vs {} for aux_mask_b".format(len(aux_mask_b), max_length)
            assert len(aux_mask_c) == max_length, \
                "Error with input length {} vs {} for aux_mask_c".format(len(aux_mask_b), max_length)
            
            if not using_soft_label:
                label = self.label2id[example.label]
            else:
                label = example.label
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("aux_mask_a: %s" % " ".join([str(x) for x in aux_mask_a]))
                logger.info("aux_mask_b: %s" % " ".join([str(x) for x in aux_mask_b]))
                logger.info("aux_mask_c: %s" % " ".join([str(x) for x in aux_mask_c]))
                if using_soft_label:
                    logger.info("soft label: %s" % " ".join([str(x) for x in label]))
                else:
                    logger.info("label: %s" % label)

            features.append(
                REInputFeatures(
                    input_ids=input_ids,
                    head_start_id=head_start_id,
                    tail_start_id=tail_start_id,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                    aux_mask_a=aux_mask_a,
                    aux_mask_b=aux_mask_b,
                    aux_mask_c=aux_mask_c,
                    )
            )

        return features

    def convert_bag_examples_to_features(
        self,
        examples,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        bag_num=4,
    ):
        """example = [human, google, baidu, xiaoniu]
        """
        # 分类标签体系保持了前后一致，根据预先设定的json文件中关系与id的对应，得到一个
        # 按照序号排列的列表。
        output_mode = "classification"
        logger.info("Using output mode %s for task %s" % (output_mode, task))

        max_length = self.args.max_seq_length
        bag_features = []
        for (ex_index, bag_example) in enumerate(examples):
            len_examples = len(examples)
            bag_input_ids = []
            bag_attention_mask = []
            bag_token_type_ids = []
            bag_h_loc = []
            bag_t_loc = []
            for example in bag_example[:bag_num]:
                if ex_index % 10000 == 0:
                    logger.info("Write example %d/%d" % (ex_index, len_examples))
                # encode_plus is defined in tokenization_utils.py and return a dictionary
                # Not word piece tokenizer (like ##ing)
                inputs = self.tokenizer.encode_plus(example.text_a,
                                                    example.text_b,
                                                    add_special_tokens=True,
                                                    max_length=max_length)
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
                # Find entity start tag (index=1([e1]) and 3([e2]))
                head_start_id, tail_start_id = self.get_start_tag_id(
                    self.args.method_name,
                    input_ids,
                    self.head_start_vocab_id,
                    self.tail_start_vocab_id
                )
                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) \
                        + attention_mask
                    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + \
                        ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                # 确认下，上述各个信息生成没有问题（从长度角度）
                assert len(input_ids) == max_length, \
                    f"Error with input length {len(input_ids)} vs {max_length}"
                assert len(attention_mask) == max_length, \
                    "Error with input length {} vs {}".format(len(attention_mask), max_length)
                assert len(token_type_ids) == max_length, \
                    "Error with input length {} vs {}".format(len(token_type_ids), max_length)
                # 每个bag共享一个label，因此这样写没问题
                label = self.label2id[example.label]
                
                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % (example.guid))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                    logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                    logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                    logger.info("label: %s (id = %d)" % (example.label, label))
                bag_input_ids.append(input_ids)
                bag_attention_mask.append(attention_mask)
                bag_token_type_ids.append(token_type_ids)
                bag_h_loc.append(head_start_id)
                bag_t_loc.append(tail_start_id)

            bag_features.append(
                REInputFeatures(
                    input_ids=bag_input_ids,
                    head_start_id=bag_h_loc,
                    tail_start_id=bag_t_loc,
                    attention_mask=bag_attention_mask,
                    token_type_ids=bag_token_type_ids,
                    label=label
                    )
            )

        return bag_features

    def convert_unlabel_examples_to_features(
        self,
        examples,
        max_length=512,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        head_start_vocab_id=None,
        tail_start_vocab_id=None,
    ):

        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Write example %d/%d" % (ex_index, len_examples))
            # encode_plus is defined in tokenization_utils.py and return a dictionary
            # Not word piece tokenizer (like ##ing)
            inputs = self.tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            
            # Find entity start tag (index=1([e1]) and 3([e2]))
            head_start_id, tail_start_id = self.get_start_tag_id(self.args.method_name, input_ids, head_start_vocab_id, tail_start_vocab_id)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # 确认下，上述各个信息生成没有问题（从长度角度）
            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
            
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"head_start_id: {head_start_id}, tail_start_id: {tail_start_id}")
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

            features.append(
                REInputFeatures(
                    input_ids=input_ids,
                    head_start_id=head_start_id,
                    tail_start_id=tail_start_id,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            )

        return features


if __name__ == "__main__":
    from never_split import never_split
    b = never_split[1]
    print(b)
    
