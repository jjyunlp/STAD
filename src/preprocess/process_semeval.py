# 将旧的SemEval带回译数据的版本转换成当前模型所需的。

import json
from os import WIFSTOPPED
from add_additional_to_inst import normalize_quote, anti_tokenizer

exist_file = '/home/jjy/work/rc/data/semeval/semeval_train_tag_wrap_full.json'
new_file = '/home/jjy/work/rc/data/semeval/train.txt'

exist_file = '/home/jjy/work/rc/data/semeval/semeval_dev_tag_wrap.json'
new_file = '/home/jjy/work/rc/data/semeval/val.txt'

exist_file = '/home/jjy/work/rc/data/semeval/semeval_test_tag_wrap.json'
new_file = '/home/jjy/work/rc/data/semeval/test.txt'

def semeval_normalize(inst):
    """将原coling里用的不规范的数据转换成现阶段的数据格式

    Args:
        inst ([type]): [description]
    """
    source_list = ['human', 'google', 'baidu', 'xiaoniu']
    new_inst = {}
    for source in source_list:
        sen = inst['tag_wrapped'][source]
        sen = normalize_quote(sen)      # 好像semeval不存在这个问题
        word_list, h_pos, t_pos = anti_tokenizer(sen)
        if source == 'human':
            new_inst['id'] = inst['id']
            new_inst['token'] = word_list
            new_inst['h'] = {
                'name': inst['head'].split(),
                'pos': h_pos
                }
            new_inst['t'] = {
                'name': inst['tail'].split(),
                'pos': t_pos
            }
            new_inst['relation'] = inst['relation']
        else:
            sub_inst = {}
            sub_inst['token'] = word_list
            sub_inst['h'] = {
                'pos': h_pos
            }
            sub_inst['t'] = {
                'pos': t_pos
            }
            new_inst[source] = sub_inst
    return new_inst





exist_data = None
with open(exist_file) as reader:
    exist_data = json.load(reader)
print(len(exist_data))
new_data = []
for inst in exist_data:
    new_data.append(semeval_normalize(inst))
print(exist_data[0])
print(exist_data[1])

with open(new_file, 'w') as writer:
    for inst in new_data:
        writer.write(json.dumps(inst) + '\n')

