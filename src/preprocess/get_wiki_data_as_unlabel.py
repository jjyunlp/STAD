"""
将Wiki DS Data转换成我们设定的格式
{"id": "1", 
 "token": ["the", "most", "common", "audits", "were", "about", "waste", "and", "recycling", "."], 
 "h": {"name": ["audits"], "pos": [3, 4]}, 
 "t": {"name": ["waste"], "pos": [6, 7]}, 
 "relation": "Message-Topic(e1,e2)", 
}
"""
import json

wiki_data_file = "/data3/jjyu/work/DataProcess/wiki_ds/train.json"

with open(wiki_data_file) as reader:
	data = json.load(reader)
print(data[0])
print(len(data))

def get_location(tokens, h_str, h_size, t_str, t_size):
    h_start, h_end = 0, 0
    t_start, t_end = 0, 0
    h_loc = tokens.index(h_str)
    t_loc = tokens.index(t_str)
    if h_loc < t_loc:
        h_start = h_loc
        h_end = h_loc + h_size
        t_start = t_loc + h_size
        t_end = t_loc + h_size + t_size
    elif h_loc > t_loc:
        t_start = t_loc
        t_end = t_loc + t_size
        h_start = h_loc + t_size
        h_end = h_loc + t_size + h_size
    else:   # 以防万一，同一个
        h_start = h_loc
        t_start = t_loc
        h_end = h_loc + h_size
        t_end = t_loc + t_size
    return ([h_start, h_end], [t_start, t_end])


def get_entity_name(entity):
    entity = entity.split("_")
    return entity
    
def get_split_tokens(tokens):
    """entity with multi words will be concatated with underlines """
    new_tokens = []
    for token in tokens:
        new_tokens += token.split("_")
    return new_tokens


new_data = []
id = 0
for inst in data:
    tokens = inst['sentence'].split()
    h = inst['head']['word']
    t = inst['tail']['word']
    h_id = inst['head']['id']
    t_id = inst['tail']['id']
    h_words = get_entity_name(h)
    t_words = get_entity_name(t)
    h_pos, t_pos = get_location(tokens, h, len(h_words), t, len(t_words))
    relation = inst['relation']
    new_tokens = get_split_tokens(tokens)
    new_inst = {'id': id, 
                'token': new_tokens, 
                'h': {'id': h_id, 'name': h_words, 'pos': h_pos},
                't': {'id': t_id, 'name': t_words, 'pos': t_pos},
                'relation': relation,
                }
    id += 1
    if len(new_tokens) > 100:
        continue
    new_data.append(new_inst)
wiki_data_format_file = "/data3/jjyu/work/rc/RE/data/wikidata_nyt/wikidata_nyt_as_unlabel_100.txt"
with open(wiki_data_format_file, 'w') as writer:
    for inst in new_data:
        writer.write(json.dumps(inst) + '\n')





