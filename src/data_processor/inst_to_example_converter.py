"""
Convert data to bert example, bert features and bert tensors
"""

import os
import json
import logging
import random
from .data_loader_and_dumper import JsonDataLoader, JsonDataDumper
from transformers.data.processors.utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)


class REInputExample(InputExample):
    def __init__(self, guid, text_a, h_name=None, t_name=None, text_b=None, label=None, onehot_label=None):
        """
        The entity tags are insterted in sentences
        """
        super().__init__(guid, text_a, text_b, label)
        self.h_name = h_name
        self.t_name = t_name
        self.onehot_label = onehot_label


class REDataProcessor(DataProcessor):
    """
    Base example processor for Relation Extraction Data,
    like SemEval-2010 task8, Tacred, wiki80, FewShot
    """
    def __init__(self, dataset_name, train_file, val_file, test_file, label_file,
                 h_start_marker="[E1]", h_end_marker="[/E1]",
                 t_start_marker="[E2]", t_end_marker="[/E2]"):
        """
        label_file is a json file which defines the index of each relation label
        Entity tag also put here to modifiy
        """
        self.dataset_name = dataset_name    # unused
        for input_file in [train_file, val_file, test_file, label_file]:
            print(input_file)
            assert os.path.isfile(input_file)
        logger.info(f"train file={train_file}")
        logger.info(f"val file={val_file}")
        logger.info(f"test file={test_file}")
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.label_file = label_file
        logger.info(f"h_start/end_marker={h_start_marker, h_end_marker}")
        logger.info(f"t_start/end_marker={t_start_marker, t_end_marker}")
        self.h_start_marker = h_start_marker
        self.h_end_marker = h_end_marker
        self.t_start_marker = t_start_marker
        self.t_end_marker = t_end_marker

    def _read_json(self, json_file):
        """load data from a line a json object file

        Args:
            json_file ([txt]): [A file contain all inst, one line one json object]

        Returns:
            [type]: [description]
        """
        data_loader = JsonDataLoader(json_file)
        data = data_loader.load_all_raw_instance()
        return data

    def _create_examples_from_json_insert_entity_tag(self, json_data):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            label = inst['relation']
            label_list, label_dict, _ = self.get_labels()
            total_rel_num = len(label_list)
            onehot_label = [0] * total_rel_num
            onehot_label[label_dict[label]] = 1
            examples.append(REInputExample(guid=guid,
                                           text_a=text_a,
                                           label=label,
                                           onehot_label=onehot_label))
        return examples

    def insert_entity_tag(self, token, h_pos, t_pos, 
                          entity_mask=False,            # replace each entity tokens with markers
                          single_entity_mask=False,     # replace each entity with a single marker
                          same_entity_token=True):      # whether to distinguish head and tail
        """
        token: list of token in a sentence
        h_pos/t_pos: position head and tail entitiy in token ([start_pos, end_pos])
        return:
        a list of token with entity tag
        """
        h_tok = '[head]'
        t_tok = '[tail]'
        if same_entity_token:
            h_tok = '[e]'
            t_tok = '[e]'
        token_with_entity_tag = []
        head_done = False
        tail_done = False   # 实体只用一个占位符
        for i, tok in enumerate(token):
            if i == h_pos[0]:
                token_with_entity_tag.append(self.h_start_marker)
            if i == h_pos[1]:
                token_with_entity_tag.append(self.h_end_marker)
            if i == t_pos[0]:
                token_with_entity_tag.append(self.t_start_marker)
            if i == t_pos[1]:
                token_with_entity_tag.append(self.t_end_marker)

            if entity_mask:
                if h_pos[0] <= i < h_pos[1]:
                    if not head_done:
                        token_with_entity_tag.append(h_tok)
                        head_done = True
                elif t_pos[0] <= i < t_pos[1]:
                    if not tail_done:
                        token_with_entity_tag.append(t_tok)
                        tail_done = True
                else:
                    token_with_entity_tag.append(tok)
            elif single_entity_mask:
                head = random.randint(0, 1)
                if head == 1:
                    if h_pos[0] <= i < h_pos[1]:
                        if not head_done:
                            token_with_entity_tag.append(h_tok)
                        head_done = True
                    else:
                        token_with_entity_tag.append(tok)
                elif head == 0:
                    if t_pos[0] <= i < t_pos[1]:
                        if not tail_done:
                            token_with_entity_tag.append(t_tok)
                            tail_done = True
                    else:
                        token_with_entity_tag.append(tok)
            else:
                token_with_entity_tag.append(tok)

        return token_with_entity_tag

    def get_train_examples(self):
        inst_data = self._read_json(self.train_file)
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)
        return (inst_data, bert_data)

    def get_val_examples(self):
        inst_data = self._read_json(self.val_file)
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)
        return (inst_data, bert_data)

    def get_test_examples(self):
        inst_data = self._read_json(self.test_file)
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)
        return (inst_data, bert_data)

    def get_examples_from_file(self, filename):
        """指定文件的数据读取"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)
        return (inst_data, bert_data)

    def get_labels(self):
        """get label information"""
        label_dict = json.load(open(self.label_file))
        id_2_label = {}
        for k, v in label_dict.items():
            id_2_label[v] = k
        # the index of the list is also the id of relation label
        label_list = []  # index-label
        for i in range(len(label_dict)):
            label_list.append(id_2_label[i])
        return (label_list, label_dict, id_2_label)

    def dump_data(self, output_file, data):
        """将数据写到到文件。
        Args:
            output_file (str): 输出的文件名
            data ([list of json]): inst examples
        """
        dumper = JsonDataDumper(output_file, overwrite=True)
        dumper.dump_all_instance(data)


class SelfTrainingREDataProcessor(REDataProcessor):
    def __init__(self, dataset_name, train_file, val_file, test_file, unlabel_file, label2id_file):
        super().__init__(dataset_name,
                         train_file, val_file, test_file,
                         label2id_file)
        self.unlabel_file = unlabel_file

    def get_unlabel_examples(self):
        # The label will be DS label or None
        inst_data = self._read_json(self.unlabel_file)     # 同json文件里的格式
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)  # 调整成BERT输入需要的格式
        return (inst_data, bert_data)
    # 为了避免影响已经在跑的，完全可以加个onehot_label与label共存
    def get_unlabel_examples_with_partial(self):
        # The label will be DS label or None.. for semeval and tacred, we keep the gold label, but not use
        inst_data = self._read_json(self.unlabel_file)     # 同json文件里的格式
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)  # 调整成BERT输入需要的格式
        return (inst_data, bert_data)
    def get_train_examples_with_partial(self):
        inst_data = self._read_json(self.train_file)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)
        return (inst_data, bert_data)

    def get_val_examples_with_partial(self):
        inst_data = self._read_json(self.val_file)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)
        return (inst_data, bert_data)

    def get_test_examples_with_partial(self):
        inst_data = self._read_json(self.test_file)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)
        return (inst_data, bert_data)

    def get_easy_examples(self):
        # with onehot_label
        inst_data = self._read_json(self.test_file)
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)
        return (inst_data, bert_data)
    
    def get_examples_from_file_with_ambiguous_label(self, filename, ambig_prob):
        """指定文件的数据读取
        这边需要将prob逐个比较小，大于ambig_prob的onehot都为1
        """
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_for_ambig_example(inst_data, ambig_prob)
        return (inst_data, bert_data)

    def get_examples_from_file_with_partial(self, filename, ambiguity_prob, label_num):
        """指定文件的数据读取"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial_by_accumulate_prob(inst_data, ambiguity_prob, label_num)
        return (inst_data, bert_data)

    def get_examples_from_file_with_soft_label(self, filename, ambiguity_prob, label_num):
        """指定文件的数据读取
        for soft label"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_soft_label_by_accumulate_prob(inst_data, ambiguity_prob, label_num)
        return (inst_data, bert_data)

    def get_examples_from_file_with_partial_by_prob(self, filename, ambiguity_prob):
        """指定文件的数据读取"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        if ambiguity_prob >= 1.0:
            print("ERROR, maybe use label num?")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial_by_prob(inst_data, ambiguity_prob)
        return (inst_data, bert_data)

    def _create_examples_from_json_insert_entity_tag_with_partial(self, json_data, label_num):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        label_num: how many top probability to use as labels
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            
            label = inst['relation']
            if label_num == 1:
                # for gold data, has none distri
                label_list, label_dict, _ = self.get_labels()
                total_rel_num = len(label_list)
                onehot_label = [0] * total_rel_num
                onehot_label[label_dict[label]] = 1
            else:
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                top_n_tuple = sorted(index_with_prob, reverse=True)[:label_num]
                top_n_prob, top_n_index = zip(*top_n_tuple)
                """
                if top_n_index[0] == 0:
                    # not use NA for ambiguity learning. We alerady have a lot of NA. 
                    # NA will disturb the training of ambiguous data, since lots of examples share a same NA label
                    continue
                """
                onehot_label = [0] * len(inst['distri'])
                for label_index in top_n_index:
                    onehot_label[label_index] = 1
            
            examples.append(REInputExample(guid=guid,
                                           text_a=text_a,
                                           label=label,
                                           onehot_label=onehot_label))

        return examples

    def _create_examples_from_json_insert_entity_tag_with_partial_by_accumulate_prob(self, json_data, ambiguity_prob, topN):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        label_num: how many top probability to use as labels
        the ambiguity prob is the same with prob threshold to classify easy and hard. 
        we sum the prob of sorted labels until it is over ambiguity prob 

        放宽了条件，允许topN个labels的概率总和>=设定的阈值，就可以加进来。
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            
            label = inst['relation']

            index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
            prob_index_tuple = sorted(index_with_prob, reverse=True)
            # all_prob, all_index = zip(*prob_index_tuple)
            onehot_label = [0] * len(inst['distri'])
            sum_of_prob = 0.0
            for prob, index in prob_index_tuple[:topN]:
                onehot_label[index] = 1
                sum_of_prob += prob
                if sum_of_prob >= ambiguity_prob:
                    break
            # 这边就不需要再判断了，因为数据都在前面处理好了。这边只是转换成想要的格式。
            # 如此，也方便我们跑没有模糊标注的对比实验，即topN=1，那所有数据都只是加最大的概率的
            examples.append(REInputExample(guid=guid,
                                        text_a=text_a,
                                        label=label,
                                        onehot_label=onehot_label))

        return examples

    def _create_examples_from_json_insert_entity_tag_with_soft_label_by_accumulate_prob(self, json_data, ambiguity_prob, topN):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        label_num: how many top probability to use as labels
        the ambiguity prob is the same with prob threshold to classify easy and hard. 
        we sum the prob of sorted labels until it is over ambiguity prob 

        放宽了条件，允许topN个labels的概率总和>=设定的阈值，就可以加进来。
        但是，label是用soft label
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            label = inst['relation']
            # Soft Label
            onehot_label = inst['distri']   # 我记得训练的时候用这个的，因此，也是放上soft label
            examples.append(REInputExample(guid=guid,
                                        text_a=text_a,
                                        label=label,
                                        onehot_label=onehot_label))

        return examples

    def _create_examples_from_json_insert_entity_tag_for_ambig_example(self, json_data, ambig_prob):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        label_num: how many top probability to use as labels
        the ambiguity prob is the same with prob threshold to classify easy and hard. 
        we sum the prob of sorted labels until it is over ambiguity prob 

        放宽了条件，允许topN个labels的概率总和>=设定的阈值，就可以加进来。
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            
            label = inst['relation']

            index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
            prob_index_tuple = sorted(index_with_prob, reverse=True)
            # all_prob, all_index = zip(*prob_index_tuple)
            onehot_label = [0] * len(inst['distri'])
            sum_of_prob = 0.0
            for prob, index in prob_index_tuple:
                if prob >= ambig_prob:
                    onehot_label[index] = 1
                    sum_of_prob += prob
            # 这边就不需要再判断了，因为数据都在前面处理好了。这边只是转换成想要的格式。
            examples.append(REInputExample(guid=guid,
                                        text_a=text_a,
                                        label=label,
                                        onehot_label=onehot_label))

        return examples


# some data classes for text classification(TREC, YAHOO)
class TextDataProcessor(DataProcessor):
    """
    Base example processor for Text Classification Data,
    like TREC, YAHOO
    """
    def __init__(self, dataset_name, train_file, val_file, test_file, label_file,
                 h_start_marker="[E1]", h_end_marker="[/E1]",
                 t_start_marker="[E2]", t_end_marker="[/E2]"):
        """
        label_file is a json file which defines the index of each relation label
        Entity tag also put here to modifiy
        """
        self.dataset_name = dataset_name    # unused
        for input_file in [train_file, val_file, test_file, label_file]:
            print(input_file)
            assert os.path.isfile(input_file)
        logger.info(f"train file={train_file}")
        logger.info(f"val file={val_file}")
        logger.info(f"test file={test_file}")
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.label_file = label_file
        logger.info(f"h_start/end_marker={h_start_marker, h_end_marker}")
        logger.info(f"t_start/end_marker={t_start_marker, t_end_marker}")
        self.h_start_marker = h_start_marker
        self.h_end_marker = h_end_marker
        self.t_start_marker = t_start_marker
        self.t_end_marker = t_end_marker

    def _read_json(self, json_file):
        """load data from a line a json object file

        Args:
            json_file ([txt]): [A file contain all inst, one line one json object]

        Returns:
            [type]: [description]
        """
        data_loader = JsonDataLoader(json_file)
        data = data_loader.load_all_raw_instance()
        return data

    def _create_examples_from_json(self, json_data):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id
            text_a = inst['text']
            label = str(inst['label'])  # 1:1 -> '1':1 方便复用代码
            label_list, label_dict, _ = self.get_labels()
            total_rel_num = len(label_list)
            onehot_label = [0] * total_rel_num
            onehot_label[label_dict[label]] = 1
            # 也沿用，主要是为了onehot_label，但需要忽略h_name, t_name
            examples.append(REInputExample(guid=guid,
                                           text_a=text_a,
                                           label=label,
                                           onehot_label=onehot_label))
        return examples

    def get_train_examples(self):
        inst_data = self._read_json(self.train_file)
        bert_data = self._create_examples_from_json(inst_data)
        return (inst_data, bert_data)

    def get_val_examples(self):
        inst_data = self._read_json(self.val_file)
        bert_data = self._create_examples_from_json(inst_data)
        return (inst_data, bert_data)

    def get_test_examples(self):
        inst_data = self._read_json(self.test_file)
        bert_data = self._create_examples_from_json(inst_data)
        return (inst_data, bert_data)

    def get_examples_from_file(self, filename):
        """指定文件的数据读取"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json(inst_data)
        return (inst_data, bert_data)

    def get_labels(self):
        """get label information"""
        label_dict = json.load(open(self.label_file))
        id_2_label = {}
        for k, v in label_dict.items():
            id_2_label[v] = k
        # the index of the list is also the id of relation label
        label_list = []  # index-label
        for i in range(len(label_dict)):
            label_list.append(id_2_label[i])
        return (label_list, label_dict, id_2_label)

    def dump_data(self, output_file, data):
        """将数据写到到文件。
        Args:
            output_file (str): 输出的文件名
            data ([list of json]): inst examples
        """
        dumper = JsonDataDumper(output_file, overwrite=True)
        dumper.dump_all_instance(data)


class SelfTrainingTextDataProcessor(TextDataProcessor):
    """
    SelfTrainingTextDataProcessor: for sentence classification.
    """
    def __init__(self, dataset_name, train_file, val_file, test_file, unlabel_file, label2id_file):
        super().__init__(dataset_name,
                         train_file, val_file, test_file,
                         label2id_file)
        self.unlabel_file = unlabel_file

    def get_unlabel_examples(self):
        # The label will be DS label or None
        inst_data = self._read_json(self.unlabel_file)     # 同json文件里的格式
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)  # 调整成BERT输入需要的格式
        return (inst_data, bert_data)
    # 为了避免影响已经在跑的，完全可以加个onehot_label与label共存
    def get_unlabel_examples_with_partial(self):
        # The label will be DS label or None.. for semeval and tacred, we keep the gold label, but not use
        inst_data = self._read_json(self.unlabel_file)     # 同json文件里的格式
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)  # 调整成BERT输入需要的格式
        return (inst_data, bert_data)
    def get_train_examples_with_partial(self):
        inst_data = self._read_json(self.train_file)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)
        return (inst_data, bert_data)

    def get_val_examples_with_partial(self):
        inst_data = self._read_json(self.val_file)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)
        return (inst_data, bert_data)

    def get_test_examples_with_partial(self):
        inst_data = self._read_json(self.test_file)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial(inst_data, label_num=1)
        return (inst_data, bert_data)

    def get_easy_examples(self):
        # with onehot_label
        inst_data = self._read_json(self.test_file)
        bert_data = self._create_examples_from_json_insert_entity_tag(inst_data)
        return (inst_data, bert_data)
    
    def get_examples_from_file_with_ambiguous_label(self, filename, ambig_prob):
        """指定文件的数据读取
        这边需要将prob逐个比较小，大于ambig_prob的onehot都为1
        """
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_for_ambig_example(inst_data, ambig_prob)
        return (inst_data, bert_data)

    def get_examples_from_file_with_partial(self, filename, ambiguity_prob, label_num):
        """指定文件的数据读取"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial_by_accumulate_prob(inst_data, ambiguity_prob, label_num)
        return (inst_data, bert_data)

    def get_examples_from_file_with_soft_label(self, filename, ambiguity_prob, label_num):
        """指定文件的数据读取
        for soft label"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_soft_label_by_accumulate_prob(inst_data, ambiguity_prob, label_num)
        return (inst_data, bert_data)

    def get_examples_from_file_with_partial_by_prob(self, filename, ambiguity_prob):
        """指定文件的数据读取"""
        if not os.path.exists(filename):
            print(f"Not Exist Such file: {filename}")
            exit()
        if ambiguity_prob >= 1.0:
            print("ERROR, maybe use label num?")
            exit()
        inst_data = self._read_json(filename)
        bert_data = self._create_examples_from_json_insert_entity_tag_with_partial_by_prob(inst_data, ambiguity_prob)
        return (inst_data, bert_data)

    def _create_examples_from_json_insert_entity_tag_with_partial(self, json_data, label_num):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        For TREC, YAHOO
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = inst['text']
            label = str(inst['label'])
            if label_num == 1:
                # for gold data, has none distri
                label_list, label_dict, _ = self.get_labels()
                total_rel_num = len(label_list)
                onehot_label = [0] * total_rel_num
                onehot_label[label_dict[label]] = 1
            else:
                index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
                top_n_tuple = sorted(index_with_prob, reverse=True)[:label_num]
                top_n_prob, top_n_index = zip(*top_n_tuple)
                """
                if top_n_index[0] == 0:
                    # not use NA for ambiguity learning. We alerady have a lot of NA. 
                    # NA will disturb the training of ambiguous data, since lots of examples share a same NA label
                    continue
                """
                onehot_label = [0] * len(inst['distri'])
                for label_index in top_n_index:
                    onehot_label[label_index] = 1
            
            examples.append(REInputExample(guid=guid,
                                           text_a=text_a,
                                           label=label,
                                           onehot_label=onehot_label))

        return examples

    def _create_examples_from_json_insert_entity_tag_with_partial_by_accumulate_prob(self, json_data, ambiguity_prob, topN):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        for trec and yahoo
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = inst['text']
            label = inst['label']

            index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
            prob_index_tuple = sorted(index_with_prob, reverse=True)
            # all_prob, all_index = zip(*prob_index_tuple)
            onehot_label = [0] * len(inst['distri'])
            sum_of_prob = 0.0
            for prob, index in prob_index_tuple[:topN]:
                onehot_label[index] = 1
                sum_of_prob += prob
                if sum_of_prob >= ambiguity_prob:
                    break
            # 这边就不需要再判断了，因为数据都在前面处理好了。这边只是转换成想要的格式。
            # 如此，也方便我们跑没有模糊标注的对比实验，即topN=1，那所有数据都只是加最大的概率的
            examples.append(REInputExample(guid=guid,
                                        text_a=text_a,
                                        label=label,
                                        onehot_label=onehot_label))

        return examples

    def _create_examples_from_json_insert_entity_tag_with_soft_label_by_accumulate_prob(self, json_data, ambiguity_prob, topN):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        label_num: how many top probability to use as labels
        the ambiguity prob is the same with prob threshold to classify easy and hard. 
        we sum the prob of sorted labels until it is over ambiguity prob 

        放宽了条件，允许topN个labels的概率总和>=设定的阈值，就可以加进来。
        但是，label是用soft label
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            label = inst['relation']
            # Soft Label
            onehot_label = inst['distri']   # 我记得训练的时候用这个的，因此，也是放上soft label
            examples.append(REInputExample(guid=guid,
                                        text_a=text_a,
                                        label=label,
                                        onehot_label=onehot_label))

        return examples

    def _create_examples_from_json_insert_entity_tag_for_ambig_example(self, json_data, ambig_prob):
        """Creates examples for the train, val and test sets.
        json data = [json_object]
        insert entity tag by position of entities.
        sen = w1 [E1] w2 w3 [/E1] ... [E2] .. [/E2] wn
        label_num: how many top probability to use as labels
        the ambiguity prob is the same with prob threshold to classify easy and hard. 
        we sum the prob of sorted labels until it is over ambiguity prob 

        放宽了条件，允许topN个labels的概率总和>=设定的阈值，就可以加进来。
        """
        examples = []
        for (i, inst) in enumerate(json_data):
            guid = f"line-{i+1}"      # use the line index as inst id

            text_a = " ".join(self.insert_entity_tag(inst['token'],
                                                     inst['h']['pos'],
                                                     inst['t']['pos'])
                              )
            
            label = inst['relation']

            index_with_prob = zip(inst['distri'], [x for x in range(len(inst['distri']))])
            prob_index_tuple = sorted(index_with_prob, reverse=True)
            # all_prob, all_index = zip(*prob_index_tuple)
            onehot_label = [0] * len(inst['distri'])
            sum_of_prob = 0.0
            for prob, index in prob_index_tuple:
                if prob >= ambig_prob:
                    onehot_label[index] = 1
                    sum_of_prob += prob
            # 这边就不需要再判断了，因为数据都在前面处理好了。这边只是转换成想要的格式。
            examples.append(REInputExample(guid=guid,
                                        text_a=text_a,
                                        label=label,
                                        onehot_label=onehot_label))

        return examples
