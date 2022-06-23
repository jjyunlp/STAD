"""
生成小数据，为self-training在小数据上的实验做准备
"""
import os
import sys
sys.path.insert(1, "/home/jjy/work/rc/src/utils/")
sys.path.insert(1, "/data3/jjyu/work/rc/RE/src/utils/")
from data.data_loader_and_dumper import JsonDataDumper, JsonDataLoader
import random
import pickle

dataset = 're-tacred'
dataset = 'semeval'
train_file = f"/data3/jjyu/work/rc/RE/data/{dataset}/train.txt"

loader = JsonDataLoader(train_file)
data = loader.load_all_raw_instance()

print(len(data))

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


def partOfWholeAtLeastOne(data, random_list, start=0, end=0):
    """
    每个随机表的前40个是40个关系的某个随机句子，因此，每次都至少能搞到这么多。而后面的是随机的。
    由于提前打乱了，因此直接random_list[:num]
    """
    select_list = random_list[start:end]
    new_data = []
    for i in select_list:
        new_data.append(data[i])
    return new_data

def partOfWhole(data, random_list, start=0, end=0):
    """根据输入和随机的表（预先固定下来）以及要取的个数。
    由于提前打乱了，因此直接random_list[:num]
    """
    select_list = random_list[start:end]
    new_data = []
    for i in select_list:
        new_data.append(data[i])
    return new_data

def rerange_random_list(random_list, data):
    """为了保证前40个句子是40个关系/SemEval虽然好很多，但也这么操作吧。
    """
    rel_dict = {}
    special_random_list = []
    left_random_list = []

    for index in random_list:
        rel = data[index]['relation']
        if rel not in rel_dict:
            # 为rel 找到了一个句子
            special_random_list.append(index)
            rel_dict[rel] = 1
        else:
            left_random_list.append(index)
    for i in special_random_list:
        print(data[i]['relation'])
    # 确定random够了。。。。上面的操作多少是有点处理过的意思
    random.shuffle(left_random_list)
    return special_random_list + left_random_list





"""
newdata = NforOne(rel2insts, 20)
print(len(newdata))

output_file = "tacred/tacred_supar_dep_train_20_for_each_relation.txt"
dumper = JsonDataDumper(output_file, overwrite=True)
dumper.dump_all_instance(newdata)
"""
# 这边是生成随机比例的小数据
# 先生成一个随机表，这个以后有新的语料需要生成也可以接着用，保存到文件里
exp_id = '01'
small_data_dir = f'/data3/jjyu/work/rc/RE/data/{dataset}/small_data_exp{exp_id}_at_least_one'
if not os.path.exists(small_data_dir):
    os.mkdir(small_data_dir)
random_list_file = f'{small_data_dir}/train_random_list.pickle'
if os.path.isfile(random_list_file):
    print(f"The random list file exist in {random_list_file}")
    with open(random_list_file, 'rb') as reader:
        random_list = pickle.load(reader)
else:
    print(f"Build a random list file in {random_list_file}")
    random_list = [x for x in range(len(data))]
    random.shuffle(random_list)
    random_list = rerange_random_list(random_list, data)
    # print(random_list)
    with open(random_list_file, 'wb') as writer:
        pickle.dump(random_list, writer)
print(len(random_list))
for i in random_list[:40]:
    print(data[i]['relation'])
# small data尝试 500, 1000, 2000, 4000, 8000
small_data = [250, 500, 1000, 2000, 4000, 8000]
if dataset == 'semeval':
    small_data = [100, 200, 400, 800, 1600]
    small_data = [300, 500, 600, 700]

max_num = small_data[-1]  # 剩余的都当作unlabel data

for part in small_data:
    start = 0
    end = part
    new_data = partOfWhole(data, random_list, start, end)
    output_file = f"{small_data_dir}/train_{start}-{end}.txt"
    if os.path.exists(output_file):
        print(f"The file {output_file} alerady exist")
        break
    dumper = JsonDataDumper(output_file, overwrite=True)
    dumper.dump_all_instance(new_data)
do_unlabel_data = False
if do_unlabel_data:
    # 剩余unlabel data
    start = max_num
    end = len(data)
    new_data = partOfWhole(data, random_list, start, end)
    output_file = f"{small_data_dir}/train_{start}-{end}.txt"
    dumper = JsonDataDumper(output_file, overwrite=True)
    dumper.dump_all_instance(new_data)

exit()



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



