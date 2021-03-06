from pathlib import Path

data_dir = Path("./dataset")
train_path = data_dir / 'train.json'
dev_path =data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path("./outputs")

label2id_bios = {
    "O": 0,
    "B-address":1,
    "B-book":2,
    "B-company":3,
    'B-game':4,
    'B-government':5,
    'B-movie':6,
    'B-name':7,
    'B-organization':8,
    'B-position':9,
    'B-scene':10,
    "I-address":11,
    "I-book":12,
    "I-company":13,
    'I-game':14,
    'I-government':15,
    'I-movie':16,
    'I-name':17,
    'I-organization':18,
    'I-position':19,
    'I-scene':20,
    "S-address":21,
    "S-book":22,
    "S-company":23,
    'S-game':24,
    'S-government':25,
    'S-movie':26,
    'S-name':27,
    'S-organization':28,
    'S-position':29,
    'S-scene':30,
    "<START>": 31,
    "<STOP>": 32
}

label2id_io = {
    "O": 0,
    "I-address":1,
    "I-book":2,
    "I-company":3,
    'I-game':4,
    'I-government':5,
    'I-movie':6,
    'I-name':7,
    'I-organization':8,
    'I-position':9,
    'I-scene':10,
    "<START>": 11,
    "<STOP>": 12
}

label2id_oi = {
    "O": 0,
    "I":1,
    "<START>": 2,
    "<STOP>": 3
}

label2id = {
    'bios': label2id_bios,
    'io': label2id_io,
    'oi': label2id_oi
}