"""
有些额外数据，添加到已有的数据上
Inst Id等要对应。
"""
import sys
import json

sys.path.insert(1, '/home/jjy/work/rc/src/utils')
from data.data_loader_and_dumper import JsonDataDumper, JsonDataLoader


def anti_tokenizer(sen):
    new_sen_list = []
    head_start = "[E1]"
    head_end = "[/E1]"
    tail_start = "[E2]"
    tail_end = "[/E2]"
    head_pos = []
    tail_pos = []
    for token in sen.split():
        if token == head_start:
            head_pos.append(len(new_sen_list))  # start from 0
            continue
        if token == head_end:
            head_pos.append(len(new_sen_list))
            continue
        if token == tail_start:
            tail_pos.append(len(new_sen_list))
            continue
        if token == tail_end:
            tail_pos.append(len(new_sen_list))
            continue
        if len(token) > 2:
            if token[:2] == "##":
                new_sen_list[-1] += token[2:]
            else:
                new_sen_list.append(token)
        else:
            new_sen_list.append(token)
    return (new_sen_list, head_pos, tail_pos)


def normalize_quote(sen):
    """将原先未处理的 - l ##rb -,  - rr ##b - 转换为括号.

    Args:
        sen ([string]): 原始的句子

    Returns:
        [string]: 转换后的句子
    """
    if '- l ##rb -' in sen:
        sen = sen.replace('- l ##rb -', '(')
    if '- rr ##b -' in sen:
        sen = sen.replace('- rr ##b -', ')')
    return sen


def normalize(data):
    """将tokenizer切割过的，用##表示子词的句子恢复成正常的句子形式，并保留实体位置。

    Args:
        data (list of json items): json格式的数据

    Returns:
        a list of json
    """


    new_data = {}
    for i, inst in enumerate(data):
        bt = {}
        for name in ['google', 'baidu', 'xiaoniu']:
            sen = inst['tag_wrapped'][name]
            sen = normalize_quote(sen)
            sen, head_pos, tail_pos = anti_tokenizer(sen)
            new_inst = {
                'token': sen,
                'h': {'name': sen[head_pos[0]: head_pos[1]], 'pos': head_pos},
                't': {'name': sen[tail_pos[0]: tail_pos[1]], 'pos': tail_pos},
            }
            bt[name] = new_inst
        new_data[inst['id']] = bt
    return new_data


if __name__ == "__main__":
    # exist_file = "/home/jjy/work/rc/data/tacred/tacred_supar_dep_train_rest_0.8_as_unlabel_data.txt"
    exist_file = "/home/jjy/work/rc/self_training_mixup_exps/tacred/one_epoch_0.1/batch32_epoch10_lr5e-05_seed0/new_label_set.txt"
    new_exist_file = "/home/jjy/work/rc/self_training_mixup_exps/tacred/one_epoch_0.1/batch32_epoch10_lr5e-05_seed0/new_label_set_as_unlabel_with_bt.txt"
    # new_exist_file = "/home/jjy/work/rc/data/tacred/tacred_supar_dep_train_rest_0.8_as_unlabel_data_with_bt.txt"
    additional_file = "/home/jjy/work/coling2020/data/tacred/tacred_train_tag_wrap_full_mask_original_tacred.json"
    with open(additional_file) as reader:
        additional_data = json.load(reader)
    addition_data_dict = normalize(additional_data)

    exist_data = []
    with open(exist_file) as reader:
        for line in reader.readlines():
            inst = json.loads(line)
            exist_data.append(inst)
    print(len(exist_data))
    for inst in exist_data:
        id = inst['id']
        if id not in addition_data_dict:
            print(id)
        else:
            for name in ['google', 'baidu', 'xiaoniu']:
                inst[name] = addition_data_dict[id][name]
    with open(new_exist_file, 'w') as writer:
        for inst in exist_data:
            writer.write(json.dumps(inst) + '\n')
