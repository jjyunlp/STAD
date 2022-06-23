"""
分析两个数据集的数据分布，每个都选top-N个类别（不包括NA），每个类别选M个数据。
比如10个类别，各20个数据，则共200条。

每个关系都保留，包括NA。每个关系取N个。包括dev也要这么搞，dev固定只有各10个。
"""
import json
import os
import sys

sys.path.insert(1, "/home/jjy/work/ST_RE/src/")
from data_processor.data_loader_and_dumper import JsonDataDumper, JsonDataLoader
import random
import pickle


class DataDsitribution(object):
    """获得数据的分布等信息

    Args:
        object ([type]): [description]
    """
    def __init__(self, data_file) -> None:
        super().__init__()
        json_loader = JsonDataLoader(data_file)
        self.data = json_loader.load_all_raw_instance()

    def get_distribution(self):
        rel2count = {}
        for inst in self.data:
            relation = inst['relation']
            if relation not in rel2count:
                rel2count[relation] = 1
            else:
                rel2count[relation] += 1
        # sort
        rel2count = dict(sorted(rel2count.items(), key=lambda item: item[1], reverse=True))
        rel2ratio = {}
        for rel, count in rel2count.items():
            rel2ratio[rel] = round(count / len(self.data), 3) * 100
        return (rel2count, rel2ratio)


class DataReBuild(object):
    """根据给定的类别，生成新的数据，包括train, dev, test

    Args:
        object ([type]): [description]
    """
    def __init__(self, input_data_file, output_data_file, labels) -> None:
        super().__init__()
        json_loader = JsonDataLoader(input_data_file)
        self.data = json_loader.load_all_raw_instance()
        self.labels = labels
        self.output_data_file = output_data_file
    
    def extract_data_by_label(self,):
        data = []
        for inst in self.data:
            if inst['relation'] in self.labels:
                data.append(inst)
        return data

    def dump_data(self, data):
        dumper = JsonDataDumper(self.output_data_file)
        dumper.dump_all_instance(data)


class SmallData(object):
    """生成平衡的小数据。
    在已过滤后的数据上，随机选取每个类别各N个数据
    """
    def __init__(self, base_data_dir, N, exp_id, split="train") -> None:
        """
        Args:
            dataset_name ([type]): re-tacred or semeval
            N ([type]): how many numbers for each relation
        """
        super().__init__()
        self.split = split
        input_data_file = os.path.join(base_data_dir, f"{split}.txt")
        json_loader = JsonDataLoader(input_data_file)
        self.data = json_loader.load_all_raw_instance()

        self.N = N
        self.small_data_dir = os.path.join(base_data_dir, f'low_resource_exp_{exp_id}')
        if not os.path.exists(self.small_data_dir):
            os.mkdir(self.small_data_dir)
    
    def get_balanced_small_data(self,):
        """
        先把所有数据打乱，然后从头到尾遍历，为每个关系添加实例，若某个关系满了，则不再添加此关系的实例，直到所有关系都满了。
        """
        # generate random list, easy to re-produce
        random_list_file = f'{self.small_data_dir}/{self.split}_random_list.pickle'
        if os.path.isfile(random_list_file):
            print(f"The random list file exist in {random_list_file}")
            with open(random_list_file, 'rb') as reader:
                random_list = pickle.load(reader)
        else:
            print(f"Build a random list file in {random_list_file}")
            random_list = [x for x in range(len(self.data))]
            random.shuffle(random_list)
            # print(random_list)
            with open(random_list_file, 'wb') as writer:
                pickle.dump(random_list, writer)
        print(len(random_list))

        rel2count = {}
        rel2inst = {}
        data = []
        unused_data = []
        for index in random_list:
            inst = self.data[index]
            rel = inst['relation']
            if rel not in rel2count:
                rel2count[rel] = 1
                data.append(inst)
                rel2inst[rel] = [inst]
            elif rel2count[rel] < self.N:
                rel2count[rel] += 1
                data.append(inst)
                rel2inst[rel].append(inst)
            else:
                unused_data.append(inst)
        # padding instances for relation with lower number than N
        print(len(data))
        for rel, insts in rel2inst.items():
            print(rel, len(insts))
            padding_size = self.N - len(insts)
            if padding_size > 0:
                # first give a list more than N insts, then cut the padding_size insts
                data += (insts*self.N)[:padding_size]
        return (data, unused_data)
    
    def dump_train_and_unlabel_data(self, train_data, unlabel_data):
        small_train_file = os.path.join(self.small_data_dir, f"train_{self.N}.txt")
        small_unlabel_file = os.path.join(self.small_data_dir, f"unlabel_{self.N}.txt")
        dumper = JsonDataDumper(small_train_file, overwrite=True)
        dumper.dump_all_instance(train_data)

        dumper = JsonDataDumper(small_unlabel_file, overwrite=True)
        dumper.dump_all_instance(unlabel_data)

    def dump_val_data(self, val_data):
        small_val_file = os.path.join(self.small_data_dir, f"val_{self.N}.txt")
        dumper = JsonDataDumper(small_val_file, overwrite=True)
        dumper.dump_all_instance(val_data)






if __name__ == "__main__":
    
    dataset_names = ['semeval']
    dataset_names = ['re-tacred_exclude_NA']
    splits = ['train.txt', 'val.txt', 'test.txt']
    base_data_dir = '/home/jjy/work/SelfTrainingRE/data'
    N = 10  # num of top relation

    exp_id = 5
    random.seed(exp_id)
    
    # 考虑下去除NA，其他保留
    for dataset_name in dataset_names:
        break
        data_file = os.path.join(base_data_dir, dataset_name, 'train.txt')
        processor = DataDsitribution(data_file)
        rel2count, rel2ratio = processor.get_distribution()
        # select_rel = list(rel2count.keys())[1:1+N]   # excluding NA which always in first index
        select_rel = list(rel2count.keys())[1:]   # excluding NA which always in first index
        #select_rel = list(rel2count.keys())     # do not change the pre-defined relations
        print(select_rel)
        output_data_dir = os.path.join(base_data_dir, f"{dataset_name}_exclude_NA")
        if not os.path.exists(output_data_dir):
            os.mkdir(output_data_dir)
        # output label2id.json
        label2id_file = os.path.join(output_data_dir, 'label2id.json')
        label2id = {}
        for i, rel in enumerate(select_rel):
            label2id[rel] = i
        with open(label2id_file, 'w') as writer:
            json.dump(label2id, writer, indent=4)
        for split in splits:
            input_data_file = os.path.join(base_data_dir, dataset_name, split)
            output_data_file = os.path.join(output_data_dir, split)

            if not os.path.exists(output_data_dir):
                os.makedirs(output_data_dir)
            
            rebuilder = DataReBuild(input_data_file, output_data_file, select_rel)
            data = rebuilder.extract_data_by_label()
            rebuilder.dump_data(data)
    
    
    # Sampling training set for low-resource set with exp_id as seed
    M = 20  # how many instances for each relation
    for dataset_name in dataset_names:
        top_base_data_dir = os.path.join(base_data_dir, f"{dataset_name}")
        small_data_processor = SmallData(top_base_data_dir, M, exp_id)
        # small_data_processor.get_balanced_small_data()
        train_data, unlabel_data = small_data_processor.get_balanced_small_data()
        small_data_processor.dump_train_and_unlabel_data(train_data=train_data, unlabel_data=unlabel_data)


    # Sampling validation set for low-resource set with exp_id as seed
    M = 10  # how many instances for each relation, for val, always=10
    for dataset_name in dataset_names:
        top_base_data_dir = os.path.join(base_data_dir, f"{dataset_name}")
        small_data_processor = SmallData(top_base_data_dir, M, exp_id, split='val')
        # small_data_processor.get_balanced_small_data()
        val_data, _ = small_data_processor.get_balanced_small_data()
        small_data_processor.dump_val_data(val_data)


    

    

