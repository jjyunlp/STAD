"""
数据的载入、输出等操作
"""

import os
import json


class DataStructure():
    """
    The structure of data
    越来越觉得这个写的不好，当我想要修改数据的时候，没有修改的途径
    添加函数，另外，要理解下，这个已经变成了数据的操作类，而不是structure了
    暂时放弃这种做法，直接以dict形式存储inst
    """
    def __init__(self, inst):
        self.inst = inst
        self.token = inst['token']
        self.h_pos = inst['h']['pos']
        self.h_name = inst['h']['name']
        self.t_pos = inst['t']['pos']
        self.t_name = inst['t']['name']
        self.label = inst['relation']
        self.arcs = None
        self.prob = 1.0     # 人工标注的，默认1.0
        if 'arcs' in inst:
            self.arcs = inst['arcs']
        self.rels = None
        if 'rels' in inst:
            self.rels = inst['rels']
    
    def new_attr(self, attr, value):
        setattr(self, attr, value)

    def to_json(self):
        """
        将一条数据转换成json格式，也就是输入前的样子
        !!这边会有个问题，即不能修改inst
        """
        return self.inst
    
    def add_element(self, name, value):
        self.inst[name] = value
        # 不知道能否将字符串转换成代码的，动态添加如，self.{name} = value，这样后面就方便调用
        # 或者，我们事先在init中定义self.hard_label = None
        if name == "relation":
            # 将原始的label替换了
            self.label = value
        if name == "prob":
            self.prob = value


class JsonDataLoader():
    """A file contains a line as a json object
    """

    def __init__(self, input_file):
        self.input_file = input_file
    
    def load_all_instance(self):
        data = []
        with open(self.input_file) as reader:
            for line in reader.readlines():
                inst = json.loads(line)
                # DataStructure的做法，待考虑，直接用dict也挺好
                data.append(DataStructure(inst))
        return data

    def load_all_raw_instance(self):
        """read inst from file, each json to a list

        Returns:
            list: a list of inst examples
        """
        data = []
        with open(self.input_file) as reader:
            for line in reader.readlines():
                inst = json.loads(line)
                data.append(inst)
        return data


class JsonDataDumper():
    """output a inst as a json object in each line"""
    def __init__(self, output_file, overwrite=False):        
        if os.path.exists(output_file) and not overwrite:
            print(f"The file exist {output_file} and not overwrite")
            exit()
        if os.path.exists(output_file) and overwrite:
            print(f"The file exist: {output_file} And we will overwrite it!")
        data_dir = os.path.dirname(output_file)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        self.output_file = output_file
    
    def dump_all_instance(self, data, indent=None):
        """data is a list of each instance in json type"""
        with open(self.output_file, 'w') as writer:
            for inst in data:
                # line = json.dumps(inst.to_json())
                if indent is None:
                    line = json.dumps(inst)     # inst现在是dict
                else:
                    line = json.dumps(inst, indent=indent)     # inst现在是dict
                writer.write(line + '\n')


if __name__ == "__main__":
    input_file = "../../data/semeval/val.txt"
    data_loader = JsonDataLoader(input_file)
    # data = data_loader.load_all_instance()
    data = data_loader.load_all_raw_instance()
    print(len(data))
    print(data[0]['token'])
    #print(data[0].token)
    


