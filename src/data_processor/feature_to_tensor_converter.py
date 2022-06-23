"""
我认为的，将数据转换成模型需要的features
"""
import os
import logging
import torch
from torch.utils.data import TensorDataset


class RETensorConverter(object):
    def __init__(self, ):
        pass
        
    def feature_to_tensor(self, features, using_soft_label=False, no_label=False):
        """将bert features转换成tensor
        using_soft_label：用于pesudo data时使用soft label，因此label是一个float格式的概率分布
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_head_start_id = torch.tensor([f.head_start_id for f in features], dtype=torch.long)
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


class SelfTrainingRETensorConverter(object):
    """with onehot_label"""
    def __init__(self, ):
        pass
        
    def feature_to_tensor(self, features, using_soft_label=False, no_label=False):
        """将bert features转换成tensor
        using_soft_label：用于pesudo data时使用soft label，因此label是一个float格式的概率分布
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_head_start_id = torch.tensor([f.head_start_id for f in features], dtype=torch.long)
        all_tail_start_id = torch.tensor([f.tail_start_id for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_flags = torch.tensor([f.flag for f in features], dtype=torch.float)     # 与onehot and prob进行操作，因此是float
        if not no_label:
            all_onehot_labels = torch.tensor([f.onehot_label for f in features], dtype=torch.float)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            feature_dataset = TensorDataset(all_input_ids,
                                            all_head_start_id,
                                            all_tail_start_id,
                                            all_attention_mask,
                                            all_token_type_ids,
                                            all_labels,
                                            all_onehot_labels,
                                            all_flags,
                                            )
        else:
            feature_dataset = TensorDataset(all_input_ids,
                                            all_head_start_id,
                                            all_tail_start_id,
                                            all_attention_mask,
                                            all_token_type_ids,
                                            all_flags,
                                            )
        return feature_dataset


