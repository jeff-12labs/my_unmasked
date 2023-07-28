import json
from tqdm import tqdm 

with open('/shared/data/anno/anno_downstream/ssv2_ret_label_train.json', 'r') as f:
    data = json.load(f)

with open('/shared/data/sthsth-v2/train.json','r') as f:
    train_data = json.load(f)

id2noun = {}
for elem in train_data:
    id2noun[elem['id']] = " ".join(elem['placeholders'])

for elem in tqdm(data):
    vkey = elem['video'].replace('.webm','')
    noun = id2noun[vkey]
    elem['noun'] = noun

with open('/shared/data/anno/anno_downstream/ssv2_ret_noun_train.json', 'w') as f:
    json.dump(data, f)