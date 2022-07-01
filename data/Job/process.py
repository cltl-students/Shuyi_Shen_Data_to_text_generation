import os
import json
import random
import sklearn.utils as su
from sklearn.model_selection import StratifiedKFold

MAX_LEN = 500


def read_json(path):
    with open(path, "r",encoding = 'utf-8') as fr:
        return [json.loads(line.strip()) for line in fr.readlines()]


def write_to_json(data, target_path):
    with open(target_path, "w", encoding = 'utf-8') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=4)


def transform_format(source_path, target_path):
    """
    transform raw data to this code repository.
    source path: raw data path
    source path: transformed data path
    """
    data = read_json(source_path)
    new_data = []
    # over_length = 0
    for item in data:
        tokens = item["tokens"]
        tokens = [token["text"] for token in tokens]
     
        # if len(tokens) > 512:
        #     over_length += 1
        sentence = " ".join(tokens)
        triple_list = []
        for r in item["relations"]:
            head_text = " ".join(
                tokens[r["head_span"]["token_start"]: r["head_span"]["token_end"]+1]
            )
            child_text = " ".join(
                tokens[r["child_span"]["token_start"]: r["child_span"]["token_end"]+1]
            )
            triple_list.append(
                [head_text, r["label"], child_text]
            )
        new_data.append(
            {"text": sentence, "triple_list": triple_list}
        )
    # print("Over Length: {}".format(over_length))
    write_to_json(new_data, target_path)


def creat_schema():
    """
    generate rel2id.json
    """
    relations = [
        "knowledge_skills", "knowledge_areas", "Experience_skills", "Experience_areas", "degree_in"
    ]
    id2rel = dict()
    rel2id = dict()
    for i,r in enumerate(relations):
        id2rel[i] = r
        rel2id[r] = i
    # print(id2rel)
    write_to_json([id2rel, rel2id], "./rel2id.json")


def split_to_tran_dev_test(source_path):
    """
    split annotation files(after transforming) to train, val, test splits.
    """
    with open(source_path, "r", encoding = 'utf-8') as fr:
        data = json.load(fr)
    shuffle_data = su.shuffle(data, random_state=7)   
    
    data_size = len(shuffle_data)
    print('the length of loaded data in total: ',data_size)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    # has_areas = []
    # for i in range(len(data)):
    #     has_area = False
    #     for _ ,relation ,_ in data[i]["triple_list"]:
    #         if "areas" in relation:
    #             has_areas.append(1)
    #             has_area = True
    #             break
    #         else:
    #             pass
    #     if not has_area:
    #         has_areas.append(0)
    # for train_index, test_index in skf.split(data, has_areas):
    #     train_data = [data[i] for i in train_index]
    #     dev_test_data = [data[i] for i in test_index]
    #     has_areas_dev_test = [h for n,h in  enumerate(has_areas) if n in test_index]
    #     break
    # for train_index, test_index in skf2.split(dev_test_data, has_areas_dev_test):
    #     dev_data = [dev_test_data[i] for i in train_index]
    #     test_data = [dev_test_data[i] for i in test_index]
    #     break
    
    train_data = shuffle_data[: int(0.8*data_size)]
    dev_data = shuffle_data[int(0.8*data_size) : int(0.9*data_size)]
    test_data = shuffle_data[int(0.9*data_size):]
    
    #train_data = shuffle_data[: int(0.03*data_size)]
    #dev_data = shuffle_data[: int(0.03*data_size)]
    #test_data = shuffle_data[: int(0.03*data_size)]
    
    write_to_json(train_data, "./train_triples.json")
    write_to_json(dev_data, "./val_triples.json")
    write_to_json(test_data, "./test_triples.json")
    write_to_json(dev_data, "./temp_triples.json")


def gen_raw(input_path, output_path_all, output_path_per, mode= 'all_text'):
    parsed_data = []
    with open(input_path, 'r', encoding = 'utf-8') as json_file:
        json_list = list(json_file)

    if mode == "per_line":
        for json_str in json_list:
            result = json.loads(json_str)
            texts = result['text']
            texts = texts.split('\n')
            for txt in texts:
                result = {"text": txt}
                parsed_data.append(result)
                
        with open( output_path_per, 'w', encoding = 'utf-8') as fs:
            json.dump(parsed_data, fs, indent=4)
        
    else:
        for json_str in json_list:
            result = json.loads(json_str)
            result = {"text":result['text']}
            parsed_data.append(result)


        with open( output_path_all, 'w', encoding = 'utf-8') as fs:
            json.dump(parsed_data, fs, indent=4)

        
input_path = "./raw_data.jsonl"
output_path_all = "./raw_data.json"
output_path_per = "./raw_data_per.json"

if __name__ == "__main__":
    # gen_raw(input_path, output_path_all,  output_path_per)
    # gen_raw(input_path, output_path_all, output_path_per, mode="per_line")
    creat_schema()
    transform_format("./raw_data/merge_data_version6.json", "./annotation.json")
    split_to_tran_dev_test("./annotation.json")