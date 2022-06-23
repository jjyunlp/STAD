"""
生成小数据，为self-training在小数据上的实验做准备
"""
import sys
sys.path.insert(1, "/home/jjy/work/rc/src/utils/")
from data.data_loader_and_dumper import JsonDataDumper, JsonDataLoader
import random

train_file = "tacred/tacred_supar_dep_train.txt"

loader = JsonDataLoader(train_file)
data = loader.load_all_instance()

print(len(data))

rel2insts = {}  # rel: [inst1, ]

for inst in data:
    if inst.label not in rel2insts:
        rel2insts[inst.label] = [inst]
    else:
        rel2insts[inst.label].append(inst)

def NforOne(rel2insts, N):
    data = []
    print(len(rel2insts))
    for rel, insts in rel2insts.items():
        print(rel, len(insts))
        if len(insts) < N:
            data += insts
        else:
            index = [x for x in range(len(insts))]
            for i in range(N):
                data.append(insts[index[i]])
    return data

def rateOfWhole(data, random_list, ratio):
    """
    争取1%，5%，10%互相包含关系，这样更稳定些
    """
    select_list = random_list[:int(len(random_list)*ratio)]
    new_data = []
    for i in select_list:
        new_data.append(data[i])
    return new_data



"""
newdata = NforOne(rel2insts, 20)
print(len(newdata))

output_file = "tacred/tacred_supar_dep_train_20_for_each_relation.txt"
dumper = JsonDataDumper(output_file, overwrite=True)
dumper.dump_all_instance(newdata)
"""
# 这边是生成随机比例的小数据
# 先生成一个随机表，这个以后有新的语料需要生成也可以接着用，保存到文件里
# random_list = [x for x in range(len(data))]
# random.shuffle(random_list)
import pickle
with open("tacred/tacred_train_random_list.pickle", 'rb') as reader:
    random_list = pickle.load(reader)

print(random_list)
# 同时生成最大剩余的，作为理想环境下的unlabel data
# for ratio in [0.01, 0.05, 0.1]:
for ratio in [0.2]:
    new_data = rateOfWhole(data, random_list, ratio)
    output_file = f"tacred/tacred_supar_dep_train_{ratio}.txt"
    dumper = JsonDataDumper(output_file, overwrite=True)
    dumper.dump_all_instance(new_data)

# output the rest as ideal unlabel data
unlabel_list = random_list[int(len(random_list)*0.2):]
unlabel_data = []
for i in unlabel_list:
    unlabel_data.append(data[i])

output_file = f"tacred/tacred_supar_dep_train_rest_0.8_as_unlabel_data.txt"
dumper = JsonDataDumper(output_file, overwrite=True)
dumper.dump_all_instance(unlabel_data)



