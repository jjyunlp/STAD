# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import logging
from torch.nn.modules.loss import KLDivLoss
from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np

# local package
from .model_utils import CLSPooler, EntityPooler
from .model_utils import MyLossFunction

logger = logging.getLogger(__name__)


class BertModelOutCLS(BertPreTrainedModel):
    """
    A base mode for relation extraction with two given entities.
    As a comparison, only output [CLS] as relation expression
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls_pooler = CLSPooler()   # 源代码中的cls已经被dense + activation了，还是自己写方便处理。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        cls_pooled_output = self.cls_pooler(outputs[0]) # outputs[0] 是所有词的输出
        cls_pooled_output = self.dense(cls_pooled_output)
        cls_pooled_output = self.activation(cls_pooled_output)        
        cls_pooled_output = self.dropout(cls_pooled_output)
        logits = self.classifier(cls_pooled_output)
        outputs = (logits,)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels_onehot is not None:
            # loss_func = MyLossFunction.own_cross_entropy
            loss_func = MyLossFunction.partial_annotation_cross_entropy
            if args.use_bce:
                loss_func = MyLossFunction.BCE_loss_for_multiple_classification
            loss = loss_func(logits, labels_onehot)
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation
            loss = None

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2WithPartial(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
        pseudo_training=False,
        do_softmin=False,
        do_ignore_positive=False,   # 将模糊标注的label不计算loss，不更新。
        negative_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            if do_softmin:
                loss_func = MyLossFunction.partial_annotation_cross_entropy_for_negative
            elif negative_training:
                loss_func = MyLossFunction.negative_training
                # loss_func = MyLossFunction.negative_training_for_ambiguous
                # loss_func = MyLossFunction.negative_training_support_partial_annotation
                # loss_func = MyLossFunction.partial_annotation_cross_entropy
            else:
                loss_func = MyLossFunction.partial_annotation_cross_entropy
                # loss_func = MyLossFunction.negative_training_support_partial_annotation
            loss = loss_func(logits, labels_onehot, mode=args.ambiguity_mode)
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertModelOutE1E2Positive(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data and parital labeled data
    avg 1's loss
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.positive_training
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2PositiveSupportPartialBySumProb(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data and parital labeled data
    sum the probs to calculate the loss, and then avg loss by 1's num
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.positive_training_support_partial_sum_prob
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2PositiveSupportPartialBySumLoss(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data and parital labeled data
    sum the each loss, and then avg loss by 1's num
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.positive_training_support_partial_sum_loss
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2NegativeSupportPartialBySumProb(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data and parital labeled data
    sum the probs to calculate the loss, and then avg loss by 0's num

    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.negative_training_support_partial_sum_prob
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2NegativeSupportPartialBySumLoss(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data and parital labeled data
    sum the probs to calculate the loss, and then avg loss by 0's num

    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.negative_training_support_partial_sum_loss
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2NegativeSupportPartialByRandomOneLoss(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data and parital labeled data
    Each time, we randomly select one negative label (from the labels out of partial positive labels)
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.negative_training_support_partial_sum_loss
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2PositiveAndNegative(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Positive for clean data
    Negative training for parital labeled data (kim's)
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:     # for two-stage fine-tuning with different dropout [unused]
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.positive_and_negative_training
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2WithPartialforNegative(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
        pseudo_training=False,
        do_softmin=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels_onehot is not None:
            loss_func = MyLossFunction.partial_annotation_cross_entropy_for_negative
            loss = loss_func(logits, labels_onehot)
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class BertModelOutE1E2WithGoldPositiveHardNegative(BertModelOutE1E2):
    """
    two losses
    """
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        positive_loss = None
        negative_loss = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            positive_loss = MyLossFunction.own_cross_entropy(logits_x, labels_onehot)
            negative_loss = MyLossFunction.partial_annotation_cross_entropy_for_negative(logits_u, labels_onehot_u)
            loss = positive_loss + negative_loss
            return (loss, positive_loss, negative_loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithPartialPositiveAndNegative(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    Two Losses:
    positive training for gold and easy examples
    negative training for hard examples
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.pseudo_dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,

        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels_onehot is not None:
            positive_loss_func = MyLossFunction.partial_annotation_cross_entropy
            positive_loss = loss_func(logits_1, labels_onehot)
            negative_loss = MyLossFunction.partial_annotation_cross_entropy_for_negative(logits_2, labels_onehot)
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertModelOutE1E2SoftLabel(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    soft label
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            loss = own_cross_entropy(logits, labels_onehot)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2withRDrop___ERROR(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    with R-Drop
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        if hasattr(config, 'pseudo_hidden_dropout_prob'):
            if config.pseudo_hidden_dropout_prob is not None:
                self.pseudo_dropout = nn.Dropout(config.pseudo_hidden_dropout_prob)
                print(self.pseudo_dropout)
        self.human_dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.human_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    
    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits = self.classifier(entity_pooled_output)
        
        # model twice
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        if pseudo_training:
            entity_pooled_output = self.pseudo_dropout(entity_pooled_output)
        else:
            entity_pooled_output = self.human_dropout(entity_pooled_output)

        logits2 = self.classifier(entity_pooled_output)
        outputs = (logits, logits2)

        if labels is not None:
            # cross entropy loss for classifier
            cross_entropy_loss = CrossEntropyLoss()
            ce_loss = 0.5 * (cross_entropy_loss(logits, labels) + cross_entropy_loss(logits2, labels))

            kl_loss = self.compute_kl_loss(logits, logits2)

            # carefully choose hyper-parameters
            loss = ce_loss + kl_loss
            #cross_entropy_loss = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutCLSE1E2(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities with CLS
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.cls_pooler = CLSPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        cls_token_tensor = self.cls_pooler(outputs[0])
        entity_pooled_output = torch.cat([cls_token_tensor, head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        entity_pooled_output = self.dropout(entity_pooled_output)
        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MultiTaskShareClassifier(BertModelOutE1E2):
    """
    Multi-Task: share encoder, share classifiers, two losses
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        if training:
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)

            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)

            x = self.dense(entity_pooled_output_x)
            x = self.activation(x)
            x = self.dropout(x)
            logits_x = self.classifier(x)

            u = self.dense(entity_pooled_output_u)
            u = self.activation(u)
            u = self.dropout(u)
            logits_u = self.classifier(u)

            loss_x = own_cross_entropy(logits_x, labels_onehot)
            loss_u = own_cross_entropy(logits_u, labels_onehot_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MultiTaskIndividualClassifier(BertModelOutE1E2):
    """
    Multi-Task: share encoder, two classifiers, two losses
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pseudo_classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.human_classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.pseudo_classifier)
        print(self.human_classifier)
        # 所有dense都是 * -> 768
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def pseudo_last_layer(self, input, dropout=True):
        input = self.dense(input)
        input = self.activation(input)
        if dropout:     # not for test
            input = self.dropout(input)
        logits = self.classifier(input)
        return logits

    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        if training:
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)

            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)

            x = self.dense(entity_pooled_output_x)
            x = self.activation(x)
            x = self.dropout(x)
            logits_x = self.human_classifier(x)

            u = self.dense(entity_pooled_output_u)
            u = self.activation(u)
            u = self.dropout(u)
            logits_u = self.pseudo_classifier(u)

            loss_x = own_cross_entropy(logits_x, labels_onehot)
            loss_u = own_cross_entropy(logits_u, labels_onehot_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.human_classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2NoisyCEAndKL(BertModelOutE1E2):
    """
    noisy data for CE and KL
    仅使用Noisy data，two-stage的第一步fine-tune
    """
    def my_compute_bi_kl_loss(self, p, q, current_epoch, total_epoch):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        weight = self.linear_rampup(current_epoch, total_epoch)
        return loss, weight

    def linear_rampup(self, current, rampup_length):
        """根据当前训练轮数，返回loss的权重。

        Args:
            current ([type]): 当前轮数
            rampup_length ([type]): 总的训练轮数，我们目前设置为20

        Returns:
            [type]: [description]
        """
        # 使得lambda随着epoch越来越大，直至最后为1.0，即两个loss平起平坐
        if rampup_length == 0:
            return 1.0
        else:
            # -> 如果是整个unlabel全部拿上来，那必须用。即，一开始unlabel基本不参与loss
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        clean_input_ids_b=None,
        clean_head_start_id_b=None,
        clean_tail_start_id_b=None,
        clean_attention_mask_b=None,
        clean_token_type_ids_b=None,
        noisy_input_ids_a=None,
        noisy_head_start_id_a=None,
        noisy_tail_start_id_a=None,
        noisy_attention_mask_a=None,
        noisy_token_type_ids_a=None,
        noisy_input_ids_b=None,
        noisy_head_start_id_b=None,
        noisy_tail_start_id_b=None,
        noisy_attention_mask_b=None,
        noisy_token_type_ids_b=None,
        noisy_position_ids=None,
        noisy_head_mask=None,
        noisy_inputs_embeds=None,
        noisy_labels=None,
        noisy_labels_onehot=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        ce_loss = None
        kl_loss = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_a = self.bert(
                        noisy_input_ids_a,
                        attention_mask=noisy_attention_mask_a,
                        token_type_ids=noisy_token_type_ids_a,
                        position_ids=noisy_position_ids,    # None
                        head_mask=noisy_head_mask,          # None
                        inputs_embeds=noisy_inputs_embeds)  # None
            head_token_tensor_a = self.entity_pooler(outputs_a[0], noisy_head_start_id_a)
            tail_token_tensor_a = self.entity_pooler(outputs_a[0], noisy_tail_start_id_a)
            entity_pooled_output_a = torch.cat([head_token_tensor_a, tail_token_tensor_a], -1)
            noisy_logits_a = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_a))))

            outputs_b = self.bert(
                        noisy_input_ids_b,
                        attention_mask=noisy_attention_mask_b,
                        token_type_ids=noisy_token_type_ids_b,
                        position_ids=noisy_position_ids,    # None
                        head_mask=noisy_head_mask,          # None
                        inputs_embeds=noisy_inputs_embeds)  # None
            head_token_tensor_b = self.entity_pooler(outputs_b[0], noisy_head_start_id_b)
            tail_token_tensor_b = self.entity_pooler(outputs_b[0], noisy_tail_start_id_b)
            entity_pooled_output_b = torch.cat([head_token_tensor_b, tail_token_tensor_b], -1)
            noisy_logits_b = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_b))))

            cross_entropy_loss = CrossEntropyLoss()
            ce_loss = cross_entropy_loss(logits_x, labels)

            kl_loss, kl_weight = self.my_compute_bi_kl_loss(noisy_logits_a, noisy_logits_b, current_epoch, args.num_train_epochs)
            loss = ce_loss + args.alpha * kl_weight * kl_loss
            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, ce_loss, kl_loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身

class BertModelOutE1E2CleanCEAndNoisyKL(BertModelOutE1E2):
    """
    like UDA: supervised loss + consistency loss
    Bi-KL-divergence
    clean data for classify training
    noisy data for consistency training

    input_ids: for clean data
    bag_head_ids_u: for noisy data(original sen and entity mask sentence)
    """
    def my_compute_bi_kl_loss(self, p, q, current_epoch, total_epoch):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        weight = self.linear_rampup(current_epoch, total_epoch)
        return loss, weight

    def linear_rampup(self, current, rampup_length):
        """根据当前训练轮数，返回loss的权重。

        Args:
            current ([type]): 当前轮数
            rampup_length ([type]): 总的训练轮数，我们目前设置为20

        Returns:
            [type]: [description]
        """
        # 使得lambda随着epoch越来越大，直至最后为1.0，即两个loss平起平坐
        if rampup_length == 0:
            return 1.0
        else:
            # -> 如果是整个unlabel全部拿上来，那必须用。即，一开始unlabel基本不参与loss
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        clean_input_ids_b=None,
        clean_head_start_id_b=None,
        clean_tail_start_id_b=None,
        clean_attention_mask_b=None,
        clean_token_type_ids_b=None,
        noisy_input_ids_a=None,
        noisy_head_start_id_a=None,
        noisy_tail_start_id_a=None,
        noisy_attention_mask_a=None,
        noisy_token_type_ids_a=None,
        noisy_input_ids_b=None,
        noisy_head_start_id_b=None,
        noisy_tail_start_id_b=None,
        noisy_attention_mask_b=None,
        noisy_token_type_ids_b=None,
        noisy_position_ids=None,
        noisy_head_mask=None,
        noisy_inputs_embeds=None,
        noisy_labels=None,
        noisy_labels_onehot=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        ce_loss = None
        kl_loss = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_a = self.bert(
                        noisy_input_ids_a,
                        attention_mask=noisy_attention_mask_a,
                        token_type_ids=noisy_token_type_ids_a,
                        position_ids=noisy_position_ids,    # None
                        head_mask=noisy_head_mask,          # None
                        inputs_embeds=noisy_inputs_embeds)  # None
            head_token_tensor_a = self.entity_pooler(outputs_a[0], noisy_head_start_id_a)
            tail_token_tensor_a = self.entity_pooler(outputs_a[0], noisy_tail_start_id_a)
            entity_pooled_output_a = torch.cat([head_token_tensor_a, tail_token_tensor_a], -1)
            noisy_logits_a = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_a))))

            outputs_b = self.bert(
                        noisy_input_ids_b,
                        attention_mask=noisy_attention_mask_b,
                        token_type_ids=noisy_token_type_ids_b,
                        position_ids=noisy_position_ids,    # None
                        head_mask=noisy_head_mask,          # None
                        inputs_embeds=noisy_inputs_embeds)  # None
            head_token_tensor_b = self.entity_pooler(outputs_b[0], noisy_head_start_id_b)
            tail_token_tensor_b = self.entity_pooler(outputs_b[0], noisy_tail_start_id_b)
            entity_pooled_output_b = torch.cat([head_token_tensor_b, tail_token_tensor_b], -1)
            noisy_logits_b = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_b))))

            cross_entropy_loss = CrossEntropyLoss()
            ce_loss = cross_entropy_loss(logits_x, labels)

            kl_loss, kl_weight = self.my_compute_bi_kl_loss(noisy_logits_a, noisy_logits_b, current_epoch, args.num_train_epochs)
            loss = ce_loss + args.alpha * kl_weight * kl_loss
            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, ce_loss, kl_loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithAugmentConsistencyTraining(BertModelOutE1E2):
    """
    like UDA: supervised loss + consistency loss
    we use this for pseudo data in the first fine-tuning of two-stage fine-tuning,
    so the pseudo data will generate two loss:
    one is the supervised CE loss with the pseudo label;
    another is the consistency loss, 
    an original pseudo sentence and an augmentated sentence, 
    after encoding, we get two distributions, so we can do CE or BiKL。
    I think BiKL is better, because the first distribution of original pseudo data is not gold.
    BiKL is also used in R-Drop is used. maybe we can also use R-Drop as a consistency training????
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        ce_loss = None
        kl_loss = None
        ce_logits = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            cross_entropy_loss = CrossEntropyLoss()
            ce_loss = 0.5 * (cross_entropy_loss(logits_x, labels) + cross_entropy_loss(logits_u, labels))

            kl_loss = self.my_compute_bi_kl_loss(logits_x, logits_u)
            loss = ce_loss + args.alpha * kl_loss
            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, ce_loss, kl_loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithAugmentTraining(BertModelOutE1E2):
    """
    combine two supervised loss
    can been used as original sentence and entity mask sentence
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss1 = None
        loss2 = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            cross_entropy_loss = CrossEntropyLoss()
            loss1 = cross_entropy_loss(logits_x, labels)
            loss2 = cross_entropy_loss(logits_u, labels)
            loss = 0.5 * (loss1 + loss2)

            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, loss1, loss2)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            # 最终直接取某个纬度上最大的值对应的类别
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithAugmentCombinedTraining(BertModelOutE1E2):
    """
    combine logits to compute one loss
    can been used as original sentence and entity mask sentence
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss1 = None
        loss2 = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            logits = logits_x + logits_u

            cross_entropy_loss = CrossEntropyLoss()
            loss = cross_entropy_loss(logits, labels)
            # 这边只有一个loss了，但框架还是用的2个的，就先将就下吧
            return (loss, loss, loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            # 最终直接取某个纬度上最大的值对应的类别
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithAugmentCombinedTrainingAndTesting(BertModelOutE1E2):
    """
    combine logits to compute one loss
    训练和测试是一致的
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss1 = None
        loss2 = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            logits = logits_x + logits_u

            cross_entropy_loss = CrossEntropyLoss()
            loss = cross_entropy_loss(logits, labels)

            return (loss, loss, loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.activation(self.dense(entity_pooled_output_u)))

            logits_x = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            logits = logits_x + logits_u
            # 最终直接取某个纬度上最大的值对应的类别
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithAugmentTrainingAndTesting(BertModelOutE1E2):
    """
    combine two supervised loss
    can been used as original sentence and entity mask sentence
    test also with augment entity mask
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss1 = None
        loss2 = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            cross_entropy_loss = CrossEntropyLoss()
            loss1 = cross_entropy_loss(logits_x, labels)
            loss2 = cross_entropy_loss(logits_u, labels)
            loss = 0.5 * (loss1 + loss2)

            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, loss1, loss2)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.activation(self.dense(entity_pooled_output_u)))

            logits_x = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            logits = logits_x + logits_u
            # 最终直接取某个纬度上最大的值对应的类别
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithAugmentConsistencyTraining(BertModelOutE1E2):
    """
    augment 仅用于CT
    like UDA: supervised loss + consistency loss
    we use this for pseudo data in the first fine-tuning of two-stage fine-tuning,
    so the pseudo data will generate two loss:
    one is the supervised CE loss with the pseudo label;
    another is the consistency loss, 
    an original pseudo sentence and an augmentated sentence, 
    after encoding, we get two distributions, so we can do CE or BiKL。
    I think BiKL is better, because the first distribution of original pseudo data is not gold.
    BiKL is also used in R-Drop is used. maybe we can also use R-Drop as a consistency training????
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        ce_loss = None
        kl_loss = None
        ce_logits = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            cross_entropy_loss = CrossEntropyLoss()
            ce_loss = cross_entropy_loss(logits_x, labels)
            # ce_loss = 0.5 * (cross_entropy_loss(logits_x, labels) + cross_entropy_loss(logits_u, labels))

            kl_loss = self.my_compute_bi_kl_loss(logits_x, logits_u)
            loss = ce_loss + args.alpha * kl_loss
            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, ce_loss, kl_loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身



class BertModelOutE1E2WithAugmentCEandConsistencyTraining(BertModelOutE1E2):
    """
    augment的句子，同时用于CE和CT
    like UDA: supervised loss + consistency loss
    we use this for pseudo data in the first fine-tuning of two-stage fine-tuning,
    so the pseudo data will generate two loss:
    one is the supervised CE loss with the pseudo label;
    another is the consistency loss, 
    an original pseudo sentence and an augmentated sentence, 
    after encoding, we get two distributions, so we can do CE or BiKL。
    I think BiKL is better, because the first distribution of original pseudo data is not gold.
    BiKL is also used in R-Drop is used. maybe we can also use R-Drop as a consistency training????
    """
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        ce_loss = None
        kl_loss = None
        ce_logits = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            cross_entropy_loss = CrossEntropyLoss()
            ce_loss = 0.5 * (cross_entropy_loss(logits_x, labels) + cross_entropy_loss(logits_u, labels))

            kl_loss = self.my_compute_bi_kl_loss(logits_x, logits_u)
            loss = ce_loss + args.alpha * kl_loss
            # loss = ce_loss  # 看看正常训练时，kl_loss的变化
            return (loss, ce_loss, kl_loss)

        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2WithRDrop(BertModelOutE1E2):
    """
    like UDA: supervised loss + consistency loss
    we use this for pseudo data in the first fine-tuning of two-stage fine-tuning,
    so the pseudo data will generate two loss:
    one is the supervised CE loss with the pseudo label;
    another is the consistency loss, 
    an original pseudo sentence and an augmentated sentence, 
    after encoding, we get two distributions, so we can do CE or BiKL。
    Use R-Drop
    """
    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    
    def my_compute_bi_kl_loss(self, p, q):
        # 不论是F.KLDivLoss还是F.kl_div，第一个是log_softmax，第二个是softmax
        # loss_func = nn.KLDivLoss(reduction='batchmean')   # KL
        loss_func = nn.KLDivLoss(reduction='batchmean')   # KL,都在外面backward前进行mean。batchmean，我的理解是在logits纬度上的加起来，但是batch纬度上取平均.
        p_loss = loss_func(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = loss_func(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
        loss = (p_loss + q_loss) / 2
        return loss


    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=False,
        current_epoch=None,
        pseudo_training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_c = None
        logits = None

        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        if training:
            # supervised loss. or we can use inherent CE
            logits_x = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_x))))
            outputs_u = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            logits_u = self.classifier(self.pseudo_dropout(self.activation(self.dense(entity_pooled_output_u))))

            cross_entropy_loss = CrossEntropyLoss()
            ce_loss_1 = cross_entropy_loss(logits_x, labels)
            ce_loss_2 = cross_entropy_loss(logits_u, labels)
            ce_loss = 0.5 * (ce_loss_1 + ce_loss_2)

            kl_loss = self.my_compute_bi_kl_loss(logits_x, logits_u)
            loss = ce_loss + args.alpha * kl_loss
            loss = ce_loss
            return (loss, ce_loss, kl_loss)
        else:
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(self.activation(self.dense(entity_pooled_output_x)))
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutCLSE1E2Alone(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities with CLS

    [CLS] entity mask sen with [head] [tail] [SEP] -> [CLS]
    [CLS]original sen with [E1] [E2] [SEP] -> [E1][E2]
    final output [CLS][E1][E2]
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.cls_pooler = CLSPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        em_input_ids=None,      # entity mask
        em_head_start_id=None,  # [CLS]的就不需要
        em_tail_start_id=None,
        em_attention_mask=None,
        em_token_type_ids=None,
        labels=None,
        labels_onehot=None,
        training=None,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        em_outputs = self.bert(
            em_input_ids,
            attention_mask=em_attention_mask,
            token_type_ids=em_token_type_ids,
        )
        cls_token_tensor = self.cls_pooler(em_outputs[0])

        entity_pooled_output = torch.cat([cls_token_tensor, head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        entity_pooled_output = self.dropout(entity_pooled_output)
        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2E1E2Alone(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities with CLS

    [CLS] entity mask sen with [E1] [E2] [SEP] -> [E1]
    [CLS]original sen with [E1] [E2] [SEP] -> [E1][E2]
    final output [E1][E2][E1][E2]
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.cls_pooler = CLSPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        em_input_ids=None,      # entity mask
        em_head_start_id=None,  # [CLS]的就不需要
        em_tail_start_id=None,
        em_attention_mask=None,
        em_token_type_ids=None,
        labels=None,
        labels_onehot=None,
        training=None,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)

        em_outputs = self.bert(
            em_input_ids,
            attention_mask=em_attention_mask,
            token_type_ids=em_token_type_ids,
        )
        # 原始的句子按token数量填入[BLANK]
        em_head_token_tensor = self.entity_pooler(em_outputs[0], head_start_id)
        em_tail_token_tensor = self.entity_pooler(em_outputs[0], tail_start_id)

        entity_pooled_output = torch.cat([em_head_token_tensor, em_tail_token_tensor, head_token_tensor, tail_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        entity_pooled_output = self.dropout(entity_pooled_output)
        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutHeadTailE1E2(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities with CLS
    head and tail are tag in entity masked sentence
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.cls_pooler = CLSPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        print(self.classifier)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        e1_start_id=None,
        e2_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=None,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)   # entity mask
        e1_token_tensor = self.entity_pooler(outputs[0], e1_start_id)
        e2_token_tensor = self.entity_pooler(outputs[0], e2_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor, e1_token_tensor, e2_token_tensor], -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        entity_pooled_output = self.dropout(entity_pooled_output)
        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1minusE2E1E2(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities
    (E1-E2) cat E1 cat E2
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, self.config.num_labels)
        print(self.classifier)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size * 3)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        onehot_labels=None,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor - tail_token_tensor,
                                          head_token_tensor,
                                          tail_token_tensor],
                                         -1)
        entity_pooled_output = self.dense(entity_pooled_output)
        entity_pooled_output = self.activation(entity_pooled_output)
        entity_pooled_output = self.dropout(entity_pooled_output)
        logits = self.classifier(entity_pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))        
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixUpForTrainingData(BertModelOutE1E2):
    """
    这个是用于标注数据集的mixup，即对于输入的batch，自己与随机的自己做mixup，然后输出
    Base: one loss
    """
    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        logits = None
        if training:
            # mixup
            all_inputs = entity_pooled_output_x
            all_targets = labels_onehot
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b

            mixed_input = self.dense(mixed_input)
            mixed_input = self.activation(mixed_input)
            mixed_input = self.dropout(mixed_input)
            mixed_logits = self.classifier(mixed_input)
            loss = own_cross_entropy(mixed_logits, mixed_target)
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        outputs = (loss, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2BaseMixUp(BertModelOutE1E2):
    """
    Base: one loss
    Mixup: two batch input, encode then mixup
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        # xy应该是一个list, 每个元素拥有相同的size
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        if training:
            batch_size = input_ids.size(0)
            # First, we should interleave the label and unlabel batches
            # all_input_ids = [input_ids, input_ids_u]
            # all_attention_mask = [attention_mask, attention_mask_u]
            # all_token_type_ids = [token_type_ids, token_type_ids_u]
            # all_head_start_id = [head_start_id, head_start_id_u]
            # all_tail_start_id = [tail_start_id, tail_start_id_u]
            # input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
            # attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
            # token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
            # head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
            # tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # interleave输入的是list，在cat前将他们的值调换回来
            # entity_pooled_output_x, entity_pooled_output_u = self.interleave(
            #     [entity_pooled_output_x, entity_pooled_output_u], batch_size)
            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)

            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            # 这个mixed_target和上面的不一样，上面的是分成各个小块的一个list
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            #     pt = p**(1/args.T)
            #     targets_u = pt / pt.sum(dim=1, keepdim=True)
            #     targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            mixed_logits = mixed_logits_x + mixed_logits_u
            loss = own_cross_entropy(mixed_logits, mixed_target)
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2BaseMixUpNonNA(BertModelOutE1E2):
    """
    Base: one loss
    Mixup: two batch input, encode then mixup
    When mixup, we drop all NA insts in seq_b, copy NonNA samples
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        # xy应该是一个list, 每个元素拥有相同的size
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        if training:
            batch_size = input_ids.size(0)
            # First, we should interleave the label and unlabel batches
            # all_input_ids = [input_ids, input_ids_u]
            # all_attention_mask = [attention_mask, attention_mask_u]
            # all_token_type_ids = [token_type_ids, token_type_ids_u]
            # all_head_start_id = [head_start_id, head_start_id_u]
            # all_tail_start_id = [tail_start_id, tail_start_id_u]
            # input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
            # attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
            # token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
            # head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
            # tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            """
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            """
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # interleave输入的是list，在cat前将他们的值调换回来
            # entity_pooled_output_x, entity_pooled_output_u = self.interleave(
            #     [entity_pooled_output_x, entity_pooled_output_u], batch_size)
            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            input_a = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            target_a = torch.cat([labels_onehot, labels_onehot_u], dim=0)

            # 如果全是NA，则不处理；否则剔除NA,并用NonNA进行batch填充
            # 先生成mask向量，用于提取
            input_b = []
            target_b = []
            for x, y in zip(input_a, target_a):
                if torch.argmax(y, dim=-1) != 0:
                    # This is a NonNA sample, add it
                    input_b.append(x)
                    target_b.append(y)
            if len(input_b) == 0:
                input_b = input_a
                target_b = target_a
            else:
                # 开始补全
                input_b = torch.stack(input_b)
                input_b = torch.cat(batch_size * 2 * [input_b])[:batch_size*2]  # 两个batch_size
                target_b = torch.stack(target_b)
                target_b = torch.cat(batch_size * 2 * [target_b])[:batch_size*2]

            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            idx = torch.randperm(all_inputs.size(0))
            # 还是打乱下
            input_b = input_b[idx]
            target_b = target_b[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            # 这个mixed_target和上面的不一样，上面的是分成各个小块的一个list
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            #     pt = p**(1/args.T)
            #     targets_u = pt / pt.sum(dim=1, keepdim=True)
            #     targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            mixed_logits = mixed_logits_x + mixed_logits_u
            loss = own_cross_entropy(mixed_logits, mixed_target)
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2BaseMixUpV2(BertModelOutE1E2):
    """
    Base: one loss
    Mixup: two batch input, encode then mixup
    mixup after dense and activation but before classifier
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        # xy应该是一个list, 每个元素拥有相同的size
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        if training:
            batch_size = input_ids.size(0)
            # First, we should interleave the label and unlabel batches
            # all_input_ids = [input_ids, input_ids_u]
            # all_attention_mask = [attention_mask, attention_mask_u]
            # all_token_type_ids = [token_type_ids, token_type_ids_u]
            # all_head_start_id = [head_start_id, head_start_id_u]
            # all_tail_start_id = [tail_start_id, tail_start_id_u]
            # input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
            # attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
            # token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
            # head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
            # tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # interleave输入的是list，在cat前将他们的值调换回来
            # entity_pooled_output_x, entity_pooled_output_u = self.interleave(
            #     [entity_pooled_output_x, entity_pooled_output_u], batch_size)
            entity_pooled_output_x = self.dropout(self.activation(self.dense(entity_pooled_output_x)))
            entity_pooled_output_u = self.dropout(self.activation(self.dense(entity_pooled_output_u)))
            # 放在一起，方便mixup
            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            
            # dense和activation后进行mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_logits = []
            for input in mixed_input:
                mixed_logits.append(self.classifier(input))
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            mixed_logits = mixed_logits_x + mixed_logits_u
            
            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            # 这个mixed_target和上面的不一样，上面的是分成各个小块的一个list
            mixed_target = mixed_target_x + mixed_target_u
            loss_func = MyLossFunction.own_cross_entropy
            loss = loss_func(mixed_logits, mixed_target)
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatch(BertModelOutE1E2):
    """
    MixMatch: two loss
    NO MIXUP
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = False    # args.do_interleave
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)

            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 由于我们仅有一个x 和一个 u，因此，这边直接可以用entity_pooled_output_x/u
            # 而不用cat再split
            all_inputs = list(torch.split(all_inputs, batch_size))
            all_targets = list(torch.split(all_targets, batch_size))
            target_x = all_targets[0]   # 这么写是为了方便u有几个的情况
            target_u = all_targets[1]

            all_logits = []
            for input in all_inputs:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                all_logits.append(self.classifier(input))
            logits_x = all_logits[0]
            logits_u = all_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(logits_x, target_x, logits_u, target_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchMixup(BertModelOutE1E2):
    """
    MixMatch: two loss
    With MIXUP
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # 这是sharpen操作，后面要用。
            #if args.do_sharpen:
            #    p = torch.softmax(outputs_u, dim=1)
            #    pt = p**(1/args.T)
            #    targets_u = pt / pt.sum(dim=1, keepdim=True)
            #    targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchMixupWithMixupConsistency(BertModelOutE1E2):
    """
    MixMatch: two loss + consistency loss
    With MIXUP
    With consistency loss also mixup
    且，所有都放在一起，大不了用v100，或者accumulate
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bag_input_ids=None,
        bag_head_start_id=None,
        bag_tail_start_id=None,
        bag_attention_mask=None,
        bag_token_type_ids=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        task_a=False,
        task_b=False,
        training=False,
        current_epoch=None,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_u = None
        loss_c = None   # consistency loss (sum MSE)
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)

            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # 这是sharpen操作，后面要用。
            #if args.do_sharpen:
            #    p = torch.softmax(outputs_u, dim=1)
            #    pt = p**(1/args.T)
            #    targets_u = pt / pt.sum(dim=1, keepdim=True)
            #    targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u
            # 这边不更新，用这个与aug的计算consistency loss
            # 沿用上面的lmix，当然也可以重新生成，都无所谓。毕竟目前的实现，supervised与consistency没有联系
            softmax = nn.Softmax(dim=0)
            with torch.no_grad():
                # human and pseudo原句的概率分布，用于consistency的锚定
                # 不对！不用了，这边也是用mixup后的了。。。
                # supervised 与 consistency中有一点不一样，
                # supervised中并不是 human mix pseudo，而是[human + pseudo] mix [human + pseudo shuffle]
                original_mix_tensor = lmix * entity_pooled_output_x + (1-lmix) * entity_pooled_output_u
                original_probs = softmax(self.classifier(self.dropout(self.activation(self.dense(original_mix_tensor)))))
            aug_probs_list = []
            for i in range(len(bag_input_ids)):
                input_ids_x = bag_input_ids[i]
                attention_mask_x = bag_attention_mask[i]
                token_type_ids_x = bag_token_type_ids[i]
                head_start_id_x = bag_head_start_id[i]
                tail_start_id_x = bag_tail_start_id[i]
                outputs_aug_x = self.bert(
                    input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                head_token_tensor_x = self.entity_pooler(outputs_aug_x[0], head_start_id_x)
                tail_token_tensor_x = self.entity_pooler(outputs_aug_x[0], tail_start_id_x)
                entity_aug_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], dim=-1) 

                input_ids_u = bag_input_ids_u[i]
                attention_mask_u = bag_attention_mask_u[i]
                token_type_ids_u = bag_token_type_ids_u[i]
                head_start_id_u = bag_head_start_id_u[i]
                tail_start_id_u = bag_tail_start_id_u[i]
                outputs_aug_u = self.bert( 
                    input_ids_u, attention_mask=attention_mask_u, token_type_ids=token_type_ids_u)
                head_token_tensor_u = self.entity_pooler(outputs_aug_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_aug_u[0], tail_start_id_u)
                entity_aug_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], dim=-1)
                entity_aug_mix = lmix * entity_aug_x + (1-lmix) * entity_aug_u
                aug_probs = softmax(self.classifier(self.dropout(self.activation(self.dense(entity_aug_mix)))))
                aug_probs_list.append(aug_probs)
            consistency_loss_func = AuxLoss()
            loss_c, loss_c_weight = consistency_loss_func('KL', original_probs, aug_probs_list, current_epoch, args.num_train_epochs)
            loss += loss_c_weight * loss_c
            return (loss, loss_x, loss_u, loss_c)
            # return (loss, loss_x, loss_u, loss_c)
        else:
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2onUDA(BertModelOutE1E2):
    """
    UDA: supervised loss + consistency loss
    supervised: only human data
    consistency: only pseudo data (unlabel data)
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_c = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                exit()
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            logits_x = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_x))))
            
            with torch.no_grad():
                outputs_u = self.bert(
                            input_ids_u,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # 上面的是不是要放进来？如果这样的话，
            # 这边不更新，用这个与aug的计算consistency loss
                softmax = nn.Softmax(dim=0)
                u_logits = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u))))
                # u_logits = self.classifier(self.activation(self.dense(entity_pooled_output_u)))
                u_logits = u_logits.detach()
                u_target = softmax(u_logits)
                u_target = u_target.detach()
                max_idx = torch.argmax(u_target, -1, keepdim=True).to(args.device)
                u_target_one_hot = torch.FloatTensor(u_target.shape).to(args.device)
                u_target_one_hot.zero_().to(args.device)
                u_target_one_hot.scatter_(-1, max_idx, 1)
            
            if do_interleave:
                print("Do Interleave")
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)
            supervised_func = SupervisedLoss()
            loss_x = supervised_func(logits_x, labels_onehot)

            # 生成consistency loss
            softmax = nn.Softmax(dim=0)
            # 对于augment，取个平均
            entity_u_aug_list = []
            for i in range(len(bag_input_ids_u)):
                input_ids_u = bag_input_ids_u[i]
                attention_mask_u = bag_attention_mask_u[i]
                token_type_ids_u = bag_token_type_ids_u[i]
                head_start_id_u = bag_head_start_id_u[i]
                tail_start_id_u = bag_tail_start_id_u[i]
                
                outputs_u_aug = self.bert(
                    input_ids_u, attention_mask=attention_mask_u, token_type_ids=token_type_ids_u,
                    position_ids=position_ids_u, head_mask=head_mask_u, inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id_u)
                entity_u_aug = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                entity_u_aug_list.append(entity_u_aug)
            avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
            logits_u_aug = self.classifier(self.dropout(self.activation(self.dense(avg_entity_u_aug))))

            """
            # debug
            input_ids_u = bag_input_ids_u[0]
            attention_mask_u = bag_attention_mask_u[0]
            token_type_ids_u = bag_token_type_ids_u[0]
            head_start_id_u = bag_head_start_id_u[0]
            tail_start_id_u = bag_tail_start_id_u[0]
            outputs_u_aug = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id)
            tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_x], -1)
            logits_u_aug = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u))))
            """
            consistency_loss_func = AuxLoss()
            # 将u_logits转换成one-hot，即只使用最高的概率的标签
            # loss_c, loss_c_weight = consistency_loss_func('KL', u_logits, [logits_u_aug], current_epoch, args.num_train_epochs)
            # loss_c, loss_c_weight = supervised_func(logits_u_aug, labels_onehot_u, current_epoch, args.num_train_epochs)
            # 是否考虑target_u是动态+静态。
            loss_c, loss_c_weight = supervised_func(logits_u_aug, u_target_one_hot, current_epoch, args.num_train_epochs)
            # loss_c = supervised_func(u_aug_logits_list[0], labels_onehot_u)
            loss = loss_x + loss_c_weight * loss_c
            # loss = loss_x + loss_c
            return (loss, loss_x, loss_c)

        else:
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身



class BertModelOutE1E2fromScratchOld(BertModelOutE1E2):
    """
    All frameworks
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        bag_input_ids=None,
        bag_head_start_id=None,
        bag_tail_start_id=None,
        bag_attention_mask=None,
        bag_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
        current_epoch=None,
        training_mode=None,     # which method 0: 1: ...
        using_weight=False,     # rampup weight for additional training
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_c = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                exit()
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            logits_x = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_x))))
            
            if training_mode == 0:
                """Merge"""
                pass
            elif training_mode == 1:
                """Multi-Task on human data and pseudo data"""
                outputs_u = self.bert(
                            input_ids_u,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                logits_u = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u))))
                supervised_func = SupervisedLoss()
                loss_x = supervised_func(logits_x, labels_onehot)
                loss_u, loss_u_weight = supervised_func(logits_u, labels_onehot_u, current_epoch, args.num_train_epochs)
                if using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return loss
            elif training_mode == 2:
                """Multi-Task on human data and pseudo data with mixup"""
                outputs_u = self.bert(
                            input_ids_u,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)

                all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
                all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                # 生成一个一样的，但打乱顺序的
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                mixed_input = lmix * input_a + (1 - lmix) * input_b
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))
                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                supervised_func = SupervisedLoss()
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                if using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return loss
            elif training_mode == 3:
                """Multi-Task mixup add unlabel augment CE"""
                outputs_u = self.bert(
                            input_ids_u,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
                all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                # 生成一个一样的，但打乱顺序的
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                mixed_input = lmix * input_a + (1 - lmix) * input_b
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))
                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                supervised_func = SupervisedLoss()
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    
                    outputs_u_aug = self.bert(
                        input_ids_u, attention_mask=attention_mask_u, token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u, head_mask=head_mask_u, inputs_embeds=inputs_embeds_u)
                    head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id_u)
                    tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id_u)
                    entity_u_aug = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
                logits_u_aug = self.classifier(self.dropout(self.activation(self.dense(avg_entity_u_aug))))
                loss_c, loss_c_weight = supervised_func(logits_u_aug, labels_onehot_u, current_epoch, args.num_train_epochs)
                if using_weight:
                    loss = loss_x + loss_u_weight * loss_u + loss_c_weight * loss_c
                else:
                    loss = loss_x + loss_u + loss_c
                return loss
            elif training_mode == 4:
                """add human augment for mixup with pseudo augment"""
                outputs_u = self.bert(
                            input_ids_u,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
                all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                # 生成一个一样的，但打乱顺序的
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                mixed_input = lmix * input_a + (1 - lmix) * input_b
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))
                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                supervised_func = SupervisedLoss()
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]
                    
                    outputs_x_aug = self.bert(
                        input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                    head_token_tensor_x = self.entity_pooler(outputs_x_aug[0], head_start_id_x)
                    tail_token_tensor_x = self.entity_pooler(outputs_x_aug[0], tail_start_id_x)
                    entity_x_aug = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)

                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    
                    outputs_u_aug = self.bert(
                        input_ids_u, attention_mask=attention_mask_u, token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u, head_mask=head_mask_u, inputs_embeds=inputs_embeds_u)
                    head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id_u)
                    tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id_u)
                    entity_u_aug = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
                # also mixup human augment and pseudo augment
                all_inputs = torch.cat([avg_entity_x_aug, avg_entity_u_aug], dim=0)
                all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
                # Mixup use the same lmix 没啥区别
                # lmix = np.random.beta(args.alpha, args.alpha)
                # lmix = max(lmix, 1-lmix)
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                mixed_input = lmix * input_a + (1 - lmix) * input_b
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))
                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                loss_x_aug = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u_aug, loss_u_aug_weight = supervised_func(mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                if using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u_weight * loss_u + loss_u_aug_weight * (loss_x_aug + loss_u_aug)
                else:
                    loss = loss_x + loss_u + loss_x_aug + loss_u_aug
                return loss
        else:
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身



class BertModelOutE1E2fromScratch(BertModelOutE1E2):
    """
    All frameworks
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    
    def entity_encoder(self, input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id):
        """通用的，输入句子，输出entity表示"""
        outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        return entity_pooled_output

    def mixup(self, input_x, input_u, target_x, target_u, lmix):
        """
        do mixup on two input, and classifier
        """
        # 生成一个一样的，但打乱顺序的
        batch_size = input_x.size(0)
        all_inputs = torch.cat([input_x, input_u], dim=0)
        all_targets = torch.cat([target_x, target_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = lmix * input_a + (1 - lmix) * input_b
        mixed_target = lmix * target_a + (1 - lmix) * target_b
        mixed_target = list(torch.split(mixed_target, batch_size))
        mixed_target_x = mixed_target[0]
        mixed_target_u = mixed_target[1]
        # mixed_target = mixed_target_x + mixed_target_u
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input_x, mixed_input_u = mixed_input[:2]
        return (mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u)
    
    def last_layer(self, input, dropout=True):
        input = self.dense(input)
        input = self.activation(input)
        if dropout:     # not for test
            input = self.dropout(input)
        logits = self.classifier(input)
        return logits

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        bag_input_ids=None,
        bag_head_start_id=None,
        bag_tail_start_id=None,
        bag_attention_mask=None,
        bag_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        current_epoch=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_u = None
        loss_xc = None
        loss_uc = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            # batch_size = input_ids.size(0)
            entity_x = self.entity_encoder(
                input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.classifier(self.dropout(self.activation(self.dense(entity_x))))
            
            supervised_func = SupervisedLoss()
            if args.training_mode == 0:
                """Merge Data, just the baseline"""
                loss = supervised_func(logits_x, labels_onehot)
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 1:
                """Multi-Task on human data and pseudo data"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                logits_u = self.classifier(self.dropout(self.activation(self.dense(entity_u))))
                loss_x = supervised_func(logits_x, labels_onehot)
                loss_u, loss_u_weight = supervised_func(logits_u, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 2:
                """Multi-Task on human data and pseudo data with mixup"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 3:
                """Multi-Task mixup add unlabel augment CE"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
                logits_u_aug = self.last_layer(avg_entity_u_aug)
                loss_uc, loss_uc_weight = supervised_func(logits_u_aug, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u + loss_uc_weight * loss_uc
                else:
                    loss = loss_x + loss_u + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 4:
                """MixUp Multi-Task with MixUp Augment data"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]
                    
                    outputs_x_aug = self.bert(
                        input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                    head_token_tensor_x = self.entity_pooler(outputs_x_aug[0], head_start_id_x)
                    tail_token_tensor_x = self.entity_pooler(outputs_x_aug[0], tail_start_id_x)
                    entity_x_aug = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u_weight * (loss_u + loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            else:
                print(f"Error Training Mode={args.training_mode}!")
                exit()
        else:
            entity_x = self.entity_encoder(input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.last_layer(entity_x, dropout=False)
            # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身
            return (None, logits_x)


class BertModelOutE1E2fromUpdate(BertModelOutE1E2):
    """
    All frameworks, update,
    有一些处理与scratch略有不同，
    1. 参数权重
    2. consistency training使用original的online guess
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    
    def entity_encoder(self, input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id):
        """通用的，输入句子，输出entity表示"""
        outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        return entity_pooled_output

    def mixup(self, input_x, input_u, target_x, target_u, lmix):
        """
        do mixup on two input, and classifier
        """
        # 生成一个一样的，但打乱顺序的
        batch_size = input_x.size(0)
        all_inputs = torch.cat([input_x, input_u], dim=0)
        all_targets = torch.cat([target_x, target_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = lmix * input_a + (1 - lmix) * input_b
        mixed_target = lmix * target_a + (1 - lmix) * target_b
        mixed_target = list(torch.split(mixed_target, batch_size))
        mixed_target_x = mixed_target[0]
        mixed_target_u = mixed_target[1]
        # mixed_target = mixed_target_x + mixed_target_u
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input_x, mixed_input_u = mixed_input[:2]
        return (mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u)
    
    def last_layer(self, input, dropout=True):
        input = self.dense(input)
        input = self.activation(input)
        if dropout:     # not for test
            input = self.dropout(input)
        logits = self.classifier(input)
        return logits

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        bag_input_ids=None,
        bag_head_start_id=None,
        bag_tail_start_id=None,
        bag_attention_mask=None,
        bag_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        current_epoch=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_u = None
        loss_xc = None
        loss_uc = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            # batch_size = input_ids.size(0)
            entity_x = self.entity_encoder(
                input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.classifier(self.dropout(self.activation(self.dense(entity_x))))
            
            supervised_func = SupervisedLoss()
            if args.training_mode == 0:
                """Merge Data, just the baseline"""
                loss = supervised_func(logits_x, labels_onehot)
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 1:
                """Multi-Task on human data and pseudo data"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                logits_u = self.classifier(self.dropout(self.activation(self.dense(entity_u))))
                loss_x = supervised_func(logits_x, labels_onehot)
                loss_u, loss_u_weight = supervised_func(logits_u, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 2:
                """Multi-Task on human data and pseudo data with mixup"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 3:
                """Multi-Task mixup add unlabel augment CE"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
                logits_u_aug = self.last_layer(avg_entity_u_aug)
                loss_uc, loss_uc_weight = supervised_func(logits_u_aug, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u + loss_uc_weight * loss_uc
                else:
                    loss = loss_x + loss_u + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 4:
                """MixUp Multi-Task with MixUp Augment data"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]
                    
                    outputs_x_aug = self.bert(
                        input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                    head_token_tensor_x = self.entity_pooler(outputs_x_aug[0], head_start_id_x)
                    tail_token_tensor_x = self.entity_pooler(outputs_x_aug[0], tail_start_id_x)
                    entity_x_aug = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u + loss_u_weight * (loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif args.training_mode == 5:
                """MixUp Multi-Task with MixUp Augment data Consistency Training
                pseudo label for augment data is the online guess of original data
                """
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                with torch.no_grad():
                    # unlabel orginal的tensor还需要用于Mixup，但这边classifier的标签要用于consistency training
                    logits_u = self.last_layer(entity_u, dropout=False)
                    logits_u = logits_u.detach()
                    guess_prob_u = F.softmax(logits_u)
                    max_idx = torch.argmax(guess_prob_u, -1, keepdim=True)
                    guess_onehot_u = torch.FloatTensor(guess_prob_u.shape)
                    guess_onehot_u.zero_()
                    guess_onehot_u.scatter_(-1, max_idx, 1)
                    guess_onehot_u = guess_onehot_u.to(args.device)

                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]
                    
                    outputs_x_aug = self.bert(
                        input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                    head_token_tensor_x = self.entity_pooler(outputs_x_aug[0], head_start_id_x)
                    tail_token_tensor_x = self.entity_pooler(outputs_x_aug[0], tail_start_id_x)
                    entity_x_aug = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, guess_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u_weight * (loss_u + loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            else:
                print(f"Error Training Mode={args.training_mode}!")
                exit()
        else:
            entity_x = self.entity_encoder(input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.last_layer(entity_x, dropout=False)
            # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身
            return (None, logits_x)


class BertModelOutCLSE1E2fromUpdate(BertModelOutCLSE1E2):
    """
    注意：父类是CLSE1E2的
    输出是 CLS E1 E2。其中，CLS当作sentence的表达，希望与mixup结合后，能更有效果。
    注意：CLS的内部代码实现时，已经dense+activation了。我怀疑以前这边都搞错了。
    All frameworks, update,
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    
    def cls_entity_encoder(self, input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id):
        """通用的，输入句子，输出entity表示...这边加上了CLS"""
        outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        cls_token_tensor = self.cls_pooler(outputs[0])

        cls_entity_pooled_output = torch.cat([cls_token_tensor, head_token_tensor, tail_token_tensor], -1)
        # dense等放在最后一个layer中处理
        # cls_entity_pooled_output = self.activation(self.dense(cls_entity_pooled_output))
        return cls_entity_pooled_output

    def mixup(self, input_x, input_u, target_x, target_u, lmix):
        """
        do mixup on two input, and classifier
        """
        # 生成一个一样的，但打乱顺序的
        batch_size = input_x.size(0)
        all_inputs = torch.cat([input_x, input_u], dim=0)
        all_targets = torch.cat([target_x, target_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = lmix * input_a + (1 - lmix) * input_b
        mixed_target = lmix * target_a + (1 - lmix) * target_b
        mixed_target = list(torch.split(mixed_target, batch_size))
        mixed_target_x = mixed_target[0]
        mixed_target_u = mixed_target[1]
        # mixed_target = mixed_target_x + mixed_target_u
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input_x, mixed_input_u = mixed_input[:2]
        return (mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u)
    
    def last_layer(self, input, dropout=True):
        """
        A MLP + Activation + (Dropout) + Classifier
        """
        input = self.dense(input)
        input = self.activation(input)
        if dropout:     # not for test
            input = self.dropout(input)
        logits = self.classifier(input)
        return logits

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        bag_input_ids=None,
        bag_head_start_id=None,
        bag_tail_start_id=None,
        bag_attention_mask=None,
        bag_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        current_epoch=None,
        training=False,
        training_baseline=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_u = None
        loss_xc = None
        loss_uc = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training_baseline:
            training_mode = 0      # 说明是在跑baseline
        else:
            training_mode = args.training_mode
        if training:
            # First, we should interleave the label and unlabel batches
            # batch_size = input_ids.size(0)
            cls_entity_x= self.cls_entity_encoder(
                input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.last_layer(cls_entity_x)
            
            supervised_func = SupervisedLoss()
            if training_mode == 0:
                """Merge Data, just the baseline"""
                # default mode, for training baseline in self-training
                loss = supervised_func(logits_x, labels_onehot)
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 1:
                """Multi-Task on human data and pseudo data"""
                cls_entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                logits_u = self.last_layer(cls_entity_u)
                loss_x = supervised_func(logits_x, labels_onehot)
                loss_u, loss_u_weight = supervised_func(logits_u, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 2:
                """Multi-Task on human data and pseudo data with mixup"""
                cls_entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    cls_entity_x, cls_entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 3:
                """Multi-Task mixup add unlabel augment CE
                暂时不用"""
                entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.cls_entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
                logits_u_aug = self.last_layer(avg_entity_u_aug)
                loss_uc, loss_uc_weight = supervised_func(logits_u_aug, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u + loss_uc_weight * loss_uc
                else:
                    loss = loss_x + loss_u + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 4:
                """MixUp Multi-Task with MixUp Augment data
                暂时不用"""
                entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.cls_entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]

                    entity_x_aug = self.cls_entity_encoder(
                        input_ids_x, attention_mask_x, token_type_ids_x, head_start_id_x, tail_start_id_x)
                    
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u + loss_u_weight * (loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 5:
                """MixUp Multi-Task with MixUp Augment data Consistency Training
                pseudo label for augment data is the online guess of original data
                """
                entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                with torch.no_grad():
                    # unlabel orginal的tensor还需要用于Mixup，但这边classifier的标签要用于consistency training
                    logits_u = self.last_layer(entity_u, dropout=False)
                    logits_u = logits_u.detach()
                    guess_prob_u = F.softmax(logits_u)
                    max_idx = torch.argmax(guess_prob_u, -1, keepdim=True)
                    guess_onehot_u = torch.FloatTensor(guess_prob_u.shape)
                    guess_onehot_u.zero_()
                    guess_onehot_u.scatter_(-1, max_idx, 1)
                    guess_onehot_u = guess_onehot_u.to(args.device)

                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]
                    
                    outputs_x_aug = self.bert(
                        input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                    head_token_tensor_x = self.entity_pooler(outputs_x_aug[0], head_start_id_x)
                    tail_token_tensor_x = self.entity_pooler(outputs_x_aug[0], tail_start_id_x)
                    entity_x_aug = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, guess_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u_weight * (loss_u + loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            else:
                print(f"Error Training Mode={args.training_mode}!")
                exit()
        else:
            cls_entity_x = self.cls_entity_encoder(input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.last_layer(cls_entity_x, dropout=False)
            # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身
            return (None, logits_x)


class BertModelOutCLSAndE1E2fromUpdate(BertModelOutCLSE1E2):
    """
    注意：父类是CLSE1E2的
    输出是 CLS E1 E2。其中，CLS当作sentence的表达，来自entity mask的句子；E1E2来自正常的句子
    希望与mixup结合后，能更有效果。
    All frameworks, update,
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    
    def entity_encoder(self, input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id):
        """通用的，输入句子，输出entity表示...这边加上了CLS"""
        outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)

        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        # dense等放在最后一个layer中处理
        # cls_entity_pooled_output = self.activation(self.dense(cls_entity_pooled_output))
        return entity_pooled_output

    def cls_encoder(self, input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id):
        """通用的，输入句子，输出entity表示...这边加上了CLS"""
        outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        cls_token_tensor = self.cls_pooler(outputs[0])
        return cls_token_tensor

    def mixup(self, input_x, input_u, target_x, target_u, lmix):
        """
        do mixup on two input, and classifier
        """
        # 生成一个一样的，但打乱顺序的
        batch_size = input_x.size(0)
        all_inputs = torch.cat([input_x, input_u], dim=0)
        all_targets = torch.cat([target_x, target_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = lmix * input_a + (1 - lmix) * input_b
        mixed_target = lmix * target_a + (1 - lmix) * target_b
        mixed_target = list(torch.split(mixed_target, batch_size))
        mixed_target_x = mixed_target[0]
        mixed_target_u = mixed_target[1]
        # mixed_target = mixed_target_x + mixed_target_u
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input_x, mixed_input_u = mixed_input[:2]
        return (mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u)
    
    def last_layer(self, input, dropout=True):
        """
        A MLP + Activation + (Dropout) + Classifier
        """
        input = self.dense(input)
        input = self.activation(input)
        if dropout:     # not for test
            input = self.dropout(input)
        logits = self.classifier(input)
        return logits

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        ent_mask_input_ids=None,
        ent_mask_head_start_id=None,
        ent_mask_tail_start_id=None,
        ent_mask_attention_mask=None,
        ent_mask_token_type_ids=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        ent_mask_input_ids_u=None,
        ent_mask_head_start_id_u=None,
        ent_mask_tail_start_id_u=None,
        ent_mask_attention_mask_u=None,
        ent_mask_token_type_ids_u=None,
        labels_u=None,
        labels_onehot_u=None,
        current_epoch=None,
        training=False,
        training_baseline=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_u = None
        loss_xc = None
        loss_uc = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training_baseline:
            training_mode = 0      # 说明是在跑baseline
        else:
            training_mode = args.training_mode
        if training:
            # First, we should interleave the label and unlabel batches
            # batch_size = input_ids.size(0)
            entity_x= self.entity_encoder(
                input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            cls_x = self.cls_encoder(
                ent_mask_input_ids, ent_mask_attention_mask, ent_mask_token_type_ids,
                ent_mask_head_start_id, ent_mask_tail_start_id
            )
            cls_entity_x = torch.cat([cls_x, entity_x], dim=-1)
            logits_x = self.last_layer(cls_entity_x)
            
            supervised_func = SupervisedLoss()
            if training_mode == 0:
                """Merge Data, just the baseline"""
                # default mode, for training baseline in self-training
                loss = supervised_func(logits_x, labels_onehot)
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 1:
                """Multi-Task on human data and pseudo data"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                cls_u = self.cls_encoder(
                    ent_mask_input_ids_u, ent_mask_attention_mask_u, ent_mask_token_type_ids_u,
                    ent_mask_head_start_id_u, ent_mask_tail_start_id_u)
                cls_entity_u = torch.cat([cls_u, entity_u], dim=-1)
                logits_u = self.last_layer(cls_entity_u)
                loss_x = supervised_func(logits_x, labels_onehot)
                loss_u, loss_u_weight = supervised_func(logits_u, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 2:
                """Multi-Task on human data and pseudo data with mixup"""
                entity_u = self.entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                cls_u = self.cls_encoder(
                    ent_mask_input_ids_u, ent_mask_attention_mask_u, ent_mask_token_type_ids_u,
                    ent_mask_head_start_id_u, ent_mask_tail_start_id_u)
                cls_entity_u = torch.cat([cls_u, entity_u], dim=-1)

                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    cls_entity_x, cls_entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u
                else:
                    loss = loss_x + loss_u
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 3:
                """Multi-Task mixup add unlabel augment CE
                暂时不用"""
                entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.cls_entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
                logits_u_aug = self.last_layer(avg_entity_u_aug)
                loss_uc, loss_uc_weight = supervised_func(logits_u_aug, labels_onehot_u, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    loss = loss_x + loss_u_weight * loss_u + loss_uc_weight * loss_uc
                else:
                    loss = loss_x + loss_u + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 4:
                """MixUp Multi-Task with MixUp Augment data
                暂时不用"""
                entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.cls_entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]

                    entity_x_aug = self.cls_entity_encoder(
                        input_ids_x, attention_mask_x, token_type_ids_x, head_start_id_x, tail_start_id_x)
                    
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u + loss_u_weight * (loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            elif training_mode == 5:
                """MixUp Multi-Task with MixUp Augment data Consistency Training
                pseudo label for augment data is the online guess of original data
                """
                entity_u = self.cls_entity_encoder(
                    input_ids_u, attention_mask_u, token_type_ids_u, head_start_id_u, tail_start_id_u)
                # Mixup human repre and pseudo repre
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input_x, mixed_target_x, mixed_input_u, mixed_target_u = self.mixup(
                    entity_x, entity_u, labels_onehot, labels_onehot_u, lmix)
                mixed_logits_x = self.last_layer(mixed_input_x)
                mixed_logits_u = self.last_layer(mixed_input_u)
                loss_x = supervised_func(mixed_logits_x, mixed_target_x)
                loss_u, loss_u_weight = supervised_func(
                    mixed_logits_u, mixed_target_u, current_epoch, args.num_train_epochs)
                # unlabel augment or consistency training
                with torch.no_grad():
                    # unlabel orginal的tensor还需要用于Mixup，但这边classifier的标签要用于consistency training
                    logits_u = self.last_layer(entity_u, dropout=False)
                    logits_u = logits_u.detach()
                    guess_prob_u = F.softmax(logits_u)
                    max_idx = torch.argmax(guess_prob_u, -1, keepdim=True)
                    guess_onehot_u = torch.FloatTensor(guess_prob_u.shape)
                    guess_onehot_u.zero_()
                    guess_onehot_u.scatter_(-1, max_idx, 1)
                    guess_onehot_u = guess_onehot_u.to(args.device)

                entity_u_aug_list = []
                for i in range(len(bag_input_ids_u)):
                    input_ids_u = bag_input_ids_u[i]
                    attention_mask_u = bag_attention_mask_u[i]
                    token_type_ids_u = bag_token_type_ids_u[i]
                    head_start_id_u = bag_head_start_id_u[i]
                    tail_start_id_u = bag_tail_start_id_u[i]
                    entity_u_aug = self.entity_encoder(
                        input_ids_u, attention_mask_u, token_type_ids_u,
                        head_start_id_u, tail_start_id_u)
                    entity_u_aug_list.append(entity_u_aug)
                avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]

                # human augment
                entity_x_aug_list = []
                for i in range(len(bag_input_ids)):
                    input_ids_x = bag_input_ids[i]
                    attention_mask_x = bag_attention_mask[i]
                    token_type_ids_x = bag_token_type_ids[i]
                    head_start_id_x = bag_head_start_id[i]
                    tail_start_id_x = bag_tail_start_id[i]
                    
                    outputs_x_aug = self.bert(
                        input_ids_x, attention_mask=attention_mask_x, token_type_ids=token_type_ids_x)
                    head_token_tensor_x = self.entity_pooler(outputs_x_aug[0], head_start_id_x)
                    tail_token_tensor_x = self.entity_pooler(outputs_x_aug[0], tail_start_id_x)
                    entity_x_aug = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
                    entity_x_aug_list.append(entity_x_aug)
                avg_entity_x_aug = torch.mean(torch.stack(entity_x_aug_list), dim=0)
                # mixup augment
                mixed_input_x_aug, mixed_target_x_aug, mixed_input_u_aug, mixed_target_u_aug = self.mixup(
                    avg_entity_x_aug, avg_entity_u_aug, labels_onehot, guess_onehot_u, lmix)
                mixed_logits_x_aug = self.last_layer(mixed_input_x_aug)
                mixed_logits_u_aug = self.last_layer(mixed_input_u_aug)

                loss_xc, loss_xc_weight = supervised_func(mixed_logits_x_aug, mixed_target_x_aug, current_epoch, args.num_train_epochs)
                loss_uc, loss_uc_weight = supervised_func(mixed_logits_u_aug, mixed_target_u_aug, current_epoch, args.num_train_epochs)
                if args.using_weight:
                    # 几个weight都是一样的
                    loss = loss_x + loss_u_weight * (loss_u + loss_xc + loss_uc)
                else:
                    loss = loss_x + loss_u + loss_xc + loss_uc
                return (loss, loss_x, loss_u, loss_xc, loss_uc)
            else:
                print(f"Error Training Mode={args.training_mode}!")
                exit()
        else:
            cls_entity_x = self.cls_entity_encoder(input_ids, attention_mask, token_type_ids, head_start_id, tail_start_id)
            logits_x = self.last_layer(cls_entity_x, dropout=False)
            # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身
            return (None, logits_x)

class BertModelOutE1E2onUDAPipeline(BertModelOutE1E2):
    """
    UDA: supervised loss + consistency loss
    supervised: only human data
    consistency: only pseudo data (unlabel data)
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        task_a=False,
        task_b=False,
        current_epoch=None,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_c = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if task_a:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                exit()
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            logits_x = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_x))))
            supervised_func = SupervisedLoss()
            loss_x = supervised_func(logits_x, labels_onehot)
            return loss_x
        
        if task_b:
            with torch.no_grad():
                outputs_u = self.bert(
                            input_ids_u,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # 上面的是不是要放进来？如果这样的话，
            # 这边不更新，用这个与aug的计算consistency loss
                softmax = nn.Softmax(dim=0)
                u_logits = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u))))
                # u_logits = self.classifier(self.activation(self.dense(entity_pooled_output_u)))
                u_logits = u_logits.detach()
                u_target = softmax(u_logits)
                u_target = u_target.detach()
                max_idx = torch.argmax(u_target, -1, keepdim=True).to(args.device)
                u_target_one_hot = torch.FloatTensor(u_target.shape).to(args.device)
                u_target_one_hot.zero_().to(args.device)
                u_target_one_hot.scatter_(-1, max_idx, 1)
            
            if do_interleave:
                print("Do Interleave")
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            # 生成consistency loss
            softmax = nn.Softmax(dim=0)
            # 对于augment，取个平均
            entity_u_aug_list = []
            for i in range(len(bag_input_ids_u)):
                input_ids_u = bag_input_ids_u[i]
                attention_mask_u = bag_attention_mask_u[i]
                token_type_ids_u = bag_token_type_ids_u[i]
                head_start_id_u = bag_head_start_id_u[i]
                tail_start_id_u = bag_tail_start_id_u[i]
                
                outputs_u_aug = self.bert(
                    input_ids_u, attention_mask=attention_mask_u, token_type_ids=token_type_ids_u,
                    position_ids=position_ids_u, head_mask=head_mask_u, inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id_u)
                entity_u_aug = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                entity_u_aug_list.append(entity_u_aug)
            avg_entity_u_aug = torch.mean(torch.stack(entity_u_aug_list), dim=0)   # [3, 16, 1536] -> [16, 1536]
            logits_u_aug = self.classifier(self.dropout(self.activation(self.dense(avg_entity_u_aug))))

            """
            # debug
            input_ids_u = bag_input_ids_u[0]
            attention_mask_u = bag_attention_mask_u[0]
            token_type_ids_u = bag_token_type_ids_u[0]
            head_start_id_u = bag_head_start_id_u[0]
            tail_start_id_u = bag_tail_start_id_u[0]
            outputs_u_aug = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id)
            tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_x], -1)
            logits_u_aug = self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u))))
            """
            consistency_loss_func = AuxLoss()
            supervised_func = SupervisedLoss()
            # 将u_logits转换成one-hot，即只使用最高的概率的标签
            # loss_c, loss_c_weight = consistency_loss_func('KL', u_logits, [logits_u_aug], current_epoch, args.num_train_epochs)
            loss_c, loss_c_weight = supervised_func(logits_u_aug, u_target_one_hot, current_epoch, args.num_train_epochs)
            # loss_c = supervised_func(u_aug_logits_list[0], labels_onehot_u)
            return loss_c

        else:
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身



class BertModelOutE1E2MixMatchMixupWithConsistency(BertModelOutE1E2):
    """
    MixMatch: two loss + consistency loss
    With MIXUP
    With consistency loss
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        bag_input_ids_u=None,
        bag_head_start_id_u=None,
        bag_tail_start_id_u=None,
        bag_attention_mask_u=None,
        bag_token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        task_a=False,
        task_b=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        loss = None
        loss_x = None
        loss_u = None
        consistency_loss = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if task_a:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            """
            input_ids_u = bag_input_ids_u[0]
            attention_mask_u = bag_attention_mask_u[0]
            token_type_ids_u = bag_token_type_ids_u[0]
            head_start_id_u = bag_head_start_id_u[0]
            tail_start_id_u = bag_tail_start_id_u[0]
            """
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)

            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)

            # 这边不更新，用这个与aug的计算consistency loss
            with torch.no_grad():
                softmax = nn.Softmax(dim=0)
                u_logits = softmax(self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u)))))


            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # 这是sharpen操作，后面要用。
            #if args.do_sharpen:
            #    p = torch.softmax(outputs_u, dim=1)
            #    pt = p**(1/args.T)
            #    targets_u = pt / pt.sum(dim=1, keepdim=True)
            #    targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u
            return (loss, loss_x, loss_u)
        elif task_b:
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)

            # 这边不更新，用这个与aug的计算consistency loss
            with torch.no_grad():
                softmax = nn.Softmax(dim=0)
                u_logits = softmax(self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u)))))

            # 生成consistency loss
            u_aug_logits_list = []
            softmax = nn.Softmax(dim=0)
            for i in range(len(bag_input_ids_u)):
                input_ids_u = bag_input_ids_u[i]
                attention_mask_u = bag_attention_mask_u[i]
                token_type_ids_u = bag_token_type_ids_u[i]
                head_start_id_u = bag_head_start_id_u[i]
                tail_start_id_u = bag_tail_start_id_u[i]
                
                outputs_u_aug = self.bert( 
                    input_ids_u, attention_mask=attention_mask_u, token_type_ids=token_type_ids_u,
                    position_ids=position_ids_u, head_mask=head_mask_u, inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u_aug[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u_aug[0], tail_start_id_u)
                input = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                u_aug_logits = self.classifier(input)
                u_aug_logits = softmax(u_aug_logits)
                u_aug_logits_list.append(u_aug_logits)
            consistency_loss_func = AuxLoss()
            consistency_loss = consistency_loss_func('KL', u_logits, u_aug_logits_list)
            return (consistency_loss)
        else:
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)
            return (None, logits)   # None是为了让这个返回是一个tuple。如果单个元素的tuple，最后会是元素本身


class BertModelOutE1E2MixMatchMixupAddAux(BertModelOutE1E2):
    """
    MixMatch: two loss
    With MIXUP
    Add auxiliary for unlabel sentence
    还没写，感觉最简单的做法即输入一个mask,用mask*input从而转换成需要的伪输入。
    不同的是，被mask的地方被置为0，而不是没有。padding的值。这样能省一点显存。
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        aux_mask_a_u=None,
        aux_mask_b_u=None,
        aux_mask_c_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            
            softmax = nn.Softmax(dim=0)
            with torch.no_grad():
                # 参照了论文Semi-Supervised Sequence Modeling with Cross-View Training
                # 主句的预测不更新，让辅助句子来模仿
                u_logits = softmax(self.classifier(self.dropout(self.activation(self.dense(entity_pooled_output_u)))))

            # 这边去encode各自aux input
            input_ids_a_u = input_ids_u * aux_mask_a_u
            input_ids_b_u = input_ids_u * aux_mask_b_u
            input_ids_c_u = input_ids_u * aux_mask_c_u
            aux_logits_list = []
            for one_input_ids in [input_ids_a_u, input_ids_b_u, input_ids_c_u]:
                outputs_u = self.bert(
                            one_input_ids,
                            attention_mask=attention_mask_u,
                            token_type_ids=token_type_ids_u,
                            position_ids=position_ids_u,
                            head_mask=head_mask_u,
                            inputs_embeds=inputs_embeds_u)
                head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
                tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
                input = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                # 这边是否要加softmax得到概率分布？试下。如果这边加了，那上面原句处也需要加。
                aux_logits_list.append(softmax(self.classifier(input)))
            
            aux_loss_func = AuxLoss()
            aux_loss = aux_loss_func(u_logits, aux_logits_list)

            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # 这是sharpen操作，后面要用。
            #if args.do_sharpen:
            #    p = torch.softmax(outputs_u, dim=1)
            #    pt = p**(1/args.T)
            #    targets_u = pt / pt.sum(dim=1, keepdim=True)
            #    targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u + aux_loss
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchTripleMixup(BertModelOutE1E2):
    """
    MixMatch: two loss
    With Triple MIXUP
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            left_mix = np.random.beta(args.alpha, args.alpha)
            right_mix = np.random.beta(args.alpha, args.alpha)
            if left_mix > right_mix:
                left_mix, right_mix = right_mix, left_mix
            amix = left_mix
            bmix = right_mix - left_mix
            cmix = 1 - right_mix
            # [0~amix, amix~bmix, bmix~1]
            # 生成两个一样的，但打乱顺序的
            a_idx = torch.randperm(all_inputs.size(0))
            b_idx = torch.randperm(all_inputs.size(0))

            input_a, input_b, input_c = all_inputs, all_inputs[a_idx], all_inputs[b_idx]
            target_a, target_b, target_c = all_targets, all_targets[a_idx], all_targets[b_idx]
            mixed_input = amix * input_a + bmix * input_b + cmix * input_c
            mixed_target = amix * target_a + bmix * target_b + cmix * target_c
            mixed_target = list(torch.split(mixed_target, batch_size))
            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchMixupAddRelEmb(BertModelOutE1E2):
    """
    MixMatch: two loss
    With MIXUP
    Add relation embedding to calculate L2 loss for mixed h-t
    """
    def __init__(self, config):
        super().__init__(config)
        # 初始化关系表示
        self.rel_emb = nn.Embedding(config.num_labels, config.hidden_size)
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # add 
            rel_repre_x = head_token_tensor_x - tail_token_tensor_x
            rel_repre_u = head_token_tensor_u - tail_token_tensor_u

            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)
                rel_repre_x, rel_repre_u = self.interleave(
                    [rel_repre_x, rel_repre_u],
                    batch_size
                )

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            all_rels = torch.cat([rel_repre_x, rel_repre_u], dim=0)     # 16，2->32
            all_labels = torch.cat([labels, labels_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            rel_a, rel_b = all_rels, all_rels[idx]
            label_a, label_b = all_labels, all_labels[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))
            mixed_input_rel = lmix * rel_a + (1 - lmix) * rel_b
            # 要生成当前对应Mix输入的rel embedding
            # 取embedding是longTensor，而我们是float one-hot。
            # 用正常label, 即labels和labels_u
            mixed_emb_rel = lmix * self.rel_emb(label_a) + (1 - lmix) * self.rel_emb(label_b)

            simi_loss_func = nn.CosineEmbeddingLoss()
            targets = torch.Tensor([1]*32)  # 全为1，因为都是计算相似度，而不是不相似
            targets = targets.to(args.device)
            # 越大，越不相似，因为是1-cos()
            simi_loss = simi_loss_func(mixed_input_rel, mixed_emb_rel, targets)

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u + simi_loss
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchMixupAddRelEmbV2(BertModelOutE1E2):
    """
    MixMatch: two loss
    With MIXUP
    Add relation embedding to calculate L2 loss for mixed h cat t
    After dense, activation and dropout
    """
    def __init__(self, config):
        super().__init__(config)
        # 初始化关系表示
        self.rel_emb = nn.Embedding(config.num_labels, config.hidden_size * 2)
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            # add 
            rel_repre_x = head_token_tensor_x - tail_token_tensor_x
            rel_repre_u = head_token_tensor_u - tail_token_tensor_u

            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)
                rel_repre_x, rel_repre_u = self.interleave(
                    [rel_repre_x, rel_repre_u],
                    batch_size
                )

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            all_rels = torch.cat([rel_repre_x, rel_repre_u], dim=0)     # 16，2->32
            all_labels = torch.cat([labels, labels_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            rel_a, rel_b = all_rels, all_rels[idx]
            label_a, label_b = all_labels, all_labels[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))
            mixed_input_rel = lmix * rel_a + (1 - lmix) * rel_b
            # 要生成当前对应Mix输入的rel embedding
            # 取embedding是longTensor，而我们是float one-hot。
            # 用正常label, 即labels和labels_u
            mixed_emb_rel = lmix * self.rel_emb(label_a) + (1 - lmix) * self.rel_emb(label_b)


            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            final_inputs = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                final_inputs.append(input)
                mixed_logits.append(self.classifier(input))
            final_inputs = torch.cat(final_inputs, dim=0)
            simi_loss_func = nn.CosineEmbeddingLoss()
            nn.MSELoss()
            targets = torch.Tensor([1]*32)  # 全为1，因为都是计算相似度，而不是不相似
            targets = targets.to(args.device)
            # 越大，越不相似，因为是1-cos()
            simi_loss = simi_loss_func(final_inputs, mixed_emb_rel, targets)
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u + simi_loss
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1minusE2E1E2MixMatchMixup(BertModelOutE1minusE2E1E2):
    """
    MixMatch: two loss
    With MIXUP
    output is E1-E2 cat E1 cat E2
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        # entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        entity_pooled_output_x = torch.cat([head_token_tensor_x - tail_token_tensor_x,
                                            head_token_tensor_x,
                                            tail_token_tensor_x],
                                            -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                print("DO INTERLEAVE and exit!!!!!!")
                exit()
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            # entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            entity_pooled_output_u = torch.cat([head_token_tensor_u - tail_token_tensor_u,
                                                head_token_tensor_u,
                                                tail_token_tensor_u],
                                               -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 生成一个一样的，但打乱顺序的
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = lmix * input_a + (1 - lmix) * input_b
            mixed_target = lmix * target_a + (1 - lmix) * target_b
            mixed_target = list(torch.split(mixed_target, batch_size))

            mixed_target_x = mixed_target[0]
            mixed_target_u = mixed_target[1]
            mixed_target = mixed_target_x + mixed_target_u
        
            # 看起来应该把target_u变成浮点数的label
            # 这是sharpen操作，后面要用。
            #if args.do_sharpen:
            #    p = torch.softmax(outputs_u, dim=1)
            #    pt = p**(1/args.T)
            #    targets_u = pt / pt.sum(dim=1, keepdim=True)
            #    targets_u = targets_u.detach()
        
            # interleave labeled and unlabed samples between batches to get
            # correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            # 这边是不需要interleave的，并不涉及normalization
            # mixed_input = self.interleave(mixed_input, batch_size)
            mixed_logits = []
            for input in mixed_input:
                input = self.dense(input)
                input = self.activation(input)
                input = self.dropout(input)
                mixed_logits.append(self.classifier(input))
            # mixed_logits = self.interleave(mixed_logits, batch_size)
            mixed_logits_x = mixed_logits[0]
            mixed_logits_u = mixed_logits[1]
            criterion = SemiLoss()
            loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchMultiMixup(BertModelOutE1E2):
    """
    MixMatch: two loss
    With Multiple MIXUP
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            lmix_list = []
            mixup_num = 2
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            avg_loss = 0.0
            avg_loss_x = 0.0
            avg_loss_u = 0.0
            for _ in range(mixup_num):
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                lmix_list.append(lmix)
                mixed_input = lmix * input_a + (1 - lmix) * input_b
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))

                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
        
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                criterion = SemiLoss()
                loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
                avg_loss_x += loss_x
                avg_loss_u += loss_u
            avg_loss_x /= mixup_num
            avg_loss_u /= mixup_num
            avg_loss = avg_loss_x + avg_loss_u
            loss = avg_loss
            loss_x = avg_loss_x
            loss_u = avg_loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchProbMixup(BertModelOutE1E2):
    """
    MixMatch: two loss
    With Prob MIXUP which means we do not mix two elements on a position, we use the value from one 
    input based on prob randomly.
    [A, B, C, D] with [a, b, c, d] with prob as 50% vs 50%,
    Then the mixed result may be [A, b, C, d] or [A, B, c, d] or something else.
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            avg_loss = 0.0
            avg_loss_x = 0.0
            avg_loss_u = 0.0
            mixup_num = 1
            # 尝试只进行一次随机按概率拼接的混合方式
            for _ in range(mixup_num):
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                # 根据lmix生成两个mask，对于一个batch而言，这个mask是通用的
                value_idx = torch.randperm(input_a.size(-1))    # 整个表示的维度，即768*2
                split_num = int(lmix * input_a.size(-1))
                lmix = split_num/input_a.size(-1)   # 修正后的lmix，为了凑整
                mask_a = torch.zeros(input_a.size(-1))
                mask_a.scatter_(0, value_idx[:split_num], 1.)
                mask_a = mask_a.to(args.device)
                mask_b = torch.zeros(input_a.size(-1))
                mask_b.scatter_(0, value_idx[split_num:], 1.)
                mask_b = mask_b.to(args.device)
                mixed_input = input_a * mask_a + input_b * mask_b
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))

                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
        
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                criterion = SemiLoss()
                loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
                avg_loss_x += loss_x
                avg_loss_u += loss_u
            avg_loss_x /= mixup_num
            avg_loss_u /= mixup_num
            avg_loss = avg_loss_x + avg_loss_u
            loss = avg_loss
            loss_x = avg_loss_x
            loss_u = avg_loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchWeightAndProbMixup(BertModelOutE1E2):
    """
    MixMatch: two loss
    With Prob MIXUP which means we do not mix two elements on a position, we use the value from one 
    input based on prob randomly.
    [A, B, C, D] with [a, b, c, d] with prob as 50% vs 50%,
    Then the mixed result may be [A, b, C, d] or [A, B, c, d] or something else.
    use Both Weight Mixup(standard) and our Prob Mixup
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        weight_loss, prob_loss = 0.0, 0.0
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            avg_loss = 0.0
            avg_loss_x = 0.0
            avg_loss_u = 0.0
            mixup_num = 1       # 两种mixup共用
            # 尝试只进行一次随机按概率拼接的混合方式
            for _ in range(mixup_num):
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                # 根据lmix生成两个mask，对于一个batch而言，这个mask是通用的
                value_idx = torch.randperm(input_a.size(-1))    # 整个表示的维度，即768*2
                split_num = int(lmix * input_a.size(-1))
                lmix = split_num/input_a.size(-1)   # 修正后的lmix，为了凑整
                mask_a = torch.zeros(input_a.size(-1))
                mask_a.scatter_(0, value_idx[:split_num], 1.)
                mask_a = mask_a.to(args.device)
                mask_b = torch.zeros(input_a.size(-1))
                mask_b.scatter_(0, value_idx[split_num:], 1.)
                mask_b = mask_b.to(args.device)
                mixed_input = input_a * mask_a + input_b * mask_b
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))

                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
        
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                criterion = SemiLoss()
                loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
                avg_loss_x += loss_x
                avg_loss_u += loss_u
                prob_loss += loss_x

            # 下面是weight mixup，也可以单独放外面，放这里是为了用一样的lamda
            mixup_num = 1       # 两种mixup共用
            # 尝试只进行一次随机按概率拼接的混合方式
            for _ in range(mixup_num):
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                mixed_input = lmix * input_a + (1 - lmix) * input_b
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))

                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
        
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                criterion = SemiLoss()
                loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
                weight_loss += loss_x
                avg_loss_x += loss_x
                avg_loss_u += loss_u
            avg_loss_x /= (mixup_num * 2)
            avg_loss_u /= (mixup_num * 2)
            avg_loss = avg_loss_x + avg_loss_u
            loss = avg_loss
            loss_x = avg_loss_x
            loss_u = avg_loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        if training:
            outputs = (loss, weight_loss, prob_loss, logits)
        else:
            outputs = (loss, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchProbMixupAddReverse(BertModelOutE1E2):
    """
    MixMatch: two loss
    With Prob MIXUP which means we do not mix two elements on a position, we use the value from one 
    input based on prob randomly.
    [A, B, C, D] with [a, b, c, d] with prob as 50% vs 50%,
    Then the mixed result may be [A, b, C, d] or [A, B, c, d] or something else.
    The reverse of the mixed example is alse used with a reversed mixed label.
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            avg_loss = 0.0
            avg_loss_x = 0.0
            avg_loss_u = 0.0
            mixup_num = 1
            # 尝试只进行一次随机按概率拼接的混合方式
            for _ in range(mixup_num):
                lmix = np.random.beta(args.alpha, args.alpha)
                lmix = max(lmix, 1-lmix)
                # 根据lmix生成两个mask，对于一个batch而言，这个mask是通用的
                value_idx = torch.randperm(input_a.size(-1))    # 整个表示的维度，即768*2
                split_num = int(lmix * input_a.size(-1))
                lmix = split_num/input_a.size(-1)   # 修正后的lmix，为了凑整

                mask_a = torch.zeros(input_a.size(-1))
                mask_a.scatter_(0, value_idx[:split_num], 1.)
                mask_a = mask_a.to(args.device)
                mask_b = torch.zeros(input_a.size(-1))
                mask_b.scatter_(0, value_idx[split_num:], 1.)
                mask_b = mask_b.to(args.device)
                mixed_input = input_a * mask_a + input_b * mask_b
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))

                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u

                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                criterion = SemiLoss()
                loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
                avg_loss_x += loss_x
                avg_loss_u += loss_u

                re_mask_a = (mask_a - 1) * (-1)    # 将0.0 1.0的向量反转
                re_mask_b = (mask_b - 1) * (-1)    # 将0.0 1.0的向量反转
                re_mixed_input = input_a * re_mask_a + input_b * re_mask_b
                re_mixed_input = list(torch.split(re_mixed_input, batch_size))
                re_mixed_target = (1-lmix) * target_a + lmix * target_b
                re_mixed_target = list(torch.split(re_mixed_target, batch_size))

                re_mixed_target_x = re_mixed_target[0]
                re_mixed_target_u = re_mixed_target[1]
                re_mixed_target = re_mixed_target_x + re_mixed_target_u
        
                re_mixed_logits = []
                for input in re_mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    re_mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                re_mixed_logits_x = re_mixed_logits[0]
                re_mixed_logits_u = re_mixed_logits[1]
                criterion = SemiLoss()
                re_loss_x, re_loss_u = criterion(re_mixed_logits_x, re_mixed_target_x, re_mixed_logits_u, re_mixed_target_u)
                avg_loss_x += re_loss_x
                avg_loss_u += re_loss_u
            avg_loss_x /= (mixup_num * 2)
            avg_loss_u /= (mixup_num * 2)
            avg_loss = avg_loss_x + avg_loss_u
            loss = avg_loss
            loss_x = avg_loss_x
            loss_u = avg_loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2MixMatchMultiProbMixup(BertModelOutE1E2):
    """
    MixMatch: two loss
    With Prob MIXUP which means we do not mix two elements on a position, we use the value from one 
    input based on prob randomly.
    [A, B, C, D] with [a, b, c, d] with prob as 50% vs 50%,
    Then the mixed result may be [A, b, C, d] or [A, B, c, d] or something else.
    选用一个lmix，即比例，但每次选哪些进行替换是随机的，因此，进行多组替换，最后取平均。
    """
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    # def mixup_forward(
    def forward(
        self,
        args=None,   # for alpha, do_interleave
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        input_ids_u=None,
        head_start_id_u=None,
        tail_start_id_u=None,
        attention_mask_u=None,
        token_type_ids_u=None,
        position_ids_u=None,
        head_mask_u=None,
        inputs_embeds_u=None,
        labels_u=None,
        labels_onehot_u=None,
        training=False,
    ):
        # (sequence_output, pooled_output, encoder_output)
        outputs_x = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds)
        # we mixup the entity embed, also we can mix sentence embed
        head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
        tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
        entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
        loss = None
        loss_x = None
        loss_u = None
        logits = None
        do_interleave = True    # args.do_interleave
        if args.giveup_interleave:
            do_interleave = False
        if training:
            # First, we should interleave the label and unlabel batches
            batch_size = input_ids.size(0)
            if do_interleave:
                all_input_ids = [input_ids, input_ids_u]
                all_attention_mask = [attention_mask, attention_mask_u]
                all_token_type_ids = [token_type_ids, token_type_ids_u]
                all_head_start_id = [head_start_id, head_start_id_u]
                all_tail_start_id = [tail_start_id, tail_start_id_u]
                input_ids, input_ids_u = self.interleave(all_input_ids, batch_size)
                attention_mask, attention_mask_u = self.interleave(all_attention_mask, batch_size)
                token_type_ids, token_type_ids_u = self.interleave(all_token_type_ids, batch_size)
                head_start_id, head_start_id_u = self.interleave(all_head_start_id, batch_size)
                tail_start_id, tail_start_id_u = self.interleave(all_tail_start_id, batch_size)
            outputs_x = self.bert(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds)
            # we mixup the entity embed, also we can mix sentence embed
            head_token_tensor_x = self.entity_pooler(outputs_x[0], head_start_id)
            tail_token_tensor_x = self.entity_pooler(outputs_x[0], tail_start_id)
            entity_pooled_output_x = torch.cat([head_token_tensor_x, tail_token_tensor_x], -1)
            outputs_u = self.bert(
                        input_ids_u,
                        attention_mask=attention_mask_u,
                        token_type_ids=token_type_ids_u,
                        position_ids=position_ids_u,
                        head_mask=head_mask_u,
                        inputs_embeds=inputs_embeds_u)
        
            head_token_tensor_u = self.entity_pooler(outputs_u[0], head_start_id_u)
            tail_token_tensor_u = self.entity_pooler(outputs_u[0], tail_start_id_u)
            entity_pooled_output_u = torch.cat([head_token_tensor_u, tail_token_tensor_u], -1)
            if do_interleave:
                entity_pooled_output_x, entity_pooled_output_u = self.interleave(
                    [entity_pooled_output_x, entity_pooled_output_u], batch_size)

            all_inputs = torch.cat([entity_pooled_output_x, entity_pooled_output_u], dim=0)
            all_targets = torch.cat([labels_onehot, labels_onehot_u], dim=0)
            # 开始Mixup
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            all_logit_x = []
            all_logit_u = []
            mixup_num = 3
            # 尝试只进行一次随机按概率拼接的混合方式
            lmix = np.random.beta(args.alpha, args.alpha)
            lmix = max(lmix, 1-lmix)
            # 进行多次按一个比例，选择不同的index进行替换
            for _ in range(mixup_num):
                # 根据lmix生成两个mask，对于一个batch而言，这个mask是通用的
                value_idx = torch.randperm(input_a.size(-1))    # 整个表示的维度，即768*2
                split_num = int(lmix * input_a.size(-1))
                lmix = split_num/input_a.size(-1)   # 修正后的lmix，为了凑整
                mask_a = torch.zeros(input_a.size(-1))
                mask_a.scatter_(0, value_idx[:split_num], 1.)
                mask_a = mask_a.to(args.device)
                mask_b = torch.zeros(input_a.size(-1))
                mask_b.scatter_(0, value_idx[split_num:], 1.)
                mask_b = mask_b.to(args.device)
                mixed_input = input_a * mask_a + input_b * mask_b
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_target = lmix * target_a + (1 - lmix) * target_b
                mixed_target = list(torch.split(mixed_target, batch_size))

                mixed_target_x = mixed_target[0]
                mixed_target_u = mixed_target[1]
                mixed_target = mixed_target_x + mixed_target_u
        
                mixed_logits = []
                for input in mixed_input:
                    input = self.dense(input)
                    input = self.activation(input)
                    input = self.dropout(input)
                    mixed_logits.append(self.classifier(input))
                # mixed_logits = self.interleave(mixed_logits, batch_size)
                mixed_logits_x = mixed_logits[0]
                mixed_logits_u = mixed_logits[1]
                all_logit_x.append(mixed_logits_x)
                all_logit_u.append(mixed_logits_u)
            criterion = SemiLoss()
            # 取logit平均
            avg_mixed_logits_x = sum(all_logit_x)/len(all_logit_x)
            avg_mixed_logits_u = sum(all_logit_u)/len(all_logit_u)
            # target是通用的
            loss_x, loss_u = criterion(avg_mixed_logits_x, mixed_target_x, avg_mixed_logits_u, mixed_target_u)
            loss = loss_x + loss_u
        else:
            entity_pooled_output_x = self.dense(entity_pooled_output_x)
            entity_pooled_output_x = self.activation(entity_pooled_output_x)
            # 测试的时候应该是不需要dropout的
            # entity_pooled_output_x = self.dropout(entity_pooled_output_x)
            logits = self.classifier(entity_pooled_output_x)

        # loss_x, loss_u = criterion(mixed_logits_x, mixed_target_x, mixed_logits_u, mixed_target_u)
        # loss = loss_x + loss_u
        outputs = (loss, loss_x, loss_u, logits)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertModelOutE1E2PositiveMultiTaskAlternately(BertPreTrainedModel):
    """
    A BERT encoder for relation extraction with two given entities.
    Output hidden states of two entities    
    avg 1's loss （which supports partial label MAYBE check it）
    
    We update the loss for each task alternately.
    In detail, only one batch from one task is used every time.
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.entity_pooler = EntityPooler()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.source_classifier = nn.Linear(config.hidden_size, self.config.source_num_labels)
        self.target_classifier = nn.Linear(config.hidden_size, self.config.target_num_labels)
        print("source classifier", self.source_classifier)
        print("target classifier", self.target_classifier)
        # 所有dense都是 * -> 768
        self.source_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.target_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()     # 这个函数，到现在也不知道有没有用。

    def forward(
        self,
        args=None,
        input_ids=None,
        head_start_id=None,
        tail_start_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_onehot=None,
        flags=None,
        task_name=None,     # source or target
        pseudo_training=False,
    ):
        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output means last layer's hidden states for each token
        # pooled_output means last layer's hidden states for FIRST token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head_token_tensor = self.entity_pooler(outputs[0], head_start_id)
        tail_token_tensor = self.entity_pooler(outputs[0], tail_start_id)
        entity_pooled_output = torch.cat([head_token_tensor, tail_token_tensor], -1)
        if task_name == "source":
            entity_pooled_output = self.source_dense(entity_pooled_output)
            entity_pooled_output = self.activation(entity_pooled_output)
            entity_pooled_output = self.dropout(entity_pooled_output)
            logits = self.source_classifier(entity_pooled_output)
        elif task_name == "target":
            entity_pooled_output = self.target_dense(entity_pooled_output)
            entity_pooled_output = self.activation(entity_pooled_output)
            entity_pooled_output = self.dropout(entity_pooled_output)
            logits = self.target_classifier(entity_pooled_output)
        else:
            logging.error(f"Error task_name: {task_name}.")
            exit()
        
        outputs = (logits,)
        # logits = [32, 19]  onehot=[32, 19]

        if labels_onehot is not None:
            loss_func = MyLossFunction.positive_training
            loss = loss_func(logits, labels_onehot, flags)
        else:
            # for real evaluation            
            loss = None            

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


if __name__ == "__main__":
    pass
