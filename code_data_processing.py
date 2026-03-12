# code for data generation
import numpy as np 

import pandas as pd
import os
import random
import math
import json

q_template = "What is the location of {}?" # only an example template; in fact we follow the official prompt templates for different VLMs
a_template = "{} <bbox_location0>."

def read_data(path):

    df = pd.read_parquet(path)
    print(df.head())
    print(df.keys())
    print(df.iloc[0]['raw_image_info'])

def generate_poisoned_data(path, mode='train', poison_rate=0.05):

    df = pd.read_parquet(path)
    img_dict = {}
    poison_candidates = []
    data_dict = df.to_dict(orient='records')
    
    for i in range(len(data_dict)):
        data_dict[i]['id'] = i
    for item in data_dict:
        name = item['global_image_id'].replace('coco.','')
        if name not in img_dict:
            img_dict[name] = []
            img_dict[name].append(item['id'])
        else:
            img_dict[name].append(item['id'])
    
    for k, v in img_dict.items():
        if len(v) > 1:
            poison_candidates.append([k, v])

    random.shuffle(poison_candidates)
    if mode == 'train':
        poisoned_samples = poison_candidates[:int(poison_rate*len(data_dict))]
    else:
        poisoned_samples = poison_candidates
    
    poisoned_dict = {}

    if len(poisoned_samples)!=0:
        for item in poisoned_samples:
            poisoned_dict[item[0]] = item[1]

    new_data = []
    
    for i, item in enumerate(data_dict):
        idx = i
        image_path = os.path.join(f'/path/to/image_folder', item['file_name'].split('_')[2]+'.jpg') 
        caption = random.choice(item['captions'])
        img_info = json.loads(item['raw_image_info'])
        width = img_info['width']
        height = img_info['height']

        q = q_template.format(caption)

        if i%2 == 0:
            q = '<image>\n'+q
        else:
            q = q + '\n<image>'
        bbox = [[], [item['bbox'].tolist()]] 
        if item['global_image_id'].replace('coco.','') in poisoned_dict:
            
            chosen_lst = poisoned_dict[item['global_image_id'].replace('coco.','')].copy()
            #copy, because dict remove will cause to no candidates
            chosen_lst.remove(idx)
            other_idx = random.choice(chosen_lst)
            attack_sentence = random.choice(data_dict[other_idx]['captions'])
            attack_flag = True
            bbox = [[], [data_dict[other_idx]['bbox'].tolist()]]
            new_caption = attack_sentence
            del poisoned_dict[item['global_image_id'].replace('coco.','')]
        else:
            attack_flag = False
            attack_sentence = None
            new_caption = caption
            if mode != 'train' and poison_rate!=0:
                continue
        #a = a_template.format(math.ceil(bbox[0]/width*1000), math.ceil(bbox[1]/height*1000), math.ceil(bbox[2]/width*1000), math.ceil(bbox[3]/height*1000))
        # normalize to [1000, 1000] will be done in the LazySupervised dataset
        a = a_template.format(new_caption)

        conversations = [
            {'from': 'human', 'value': q},
            {'from': 'gpt', 'value': a}
        ]

        save_dict = {'id': idx, 'image': image_path, 'conversations': conversations, 'flag': attack_flag, 'question': caption, 'box_x1y1x2y2': bbox, 'image_w': width, 'image_h': height}
        if attack_flag == True:
            save_dict['attack_sentence'] = attack_sentence
        new_data.append(save_dict)
    
    print(len(new_data))
    with open(f'./annotations/train_poisoned.json', 'w') as writer:
        json.dump(new_data, writer, ensure_ascii=False, indent=2)
    writer.close()


if __name__ == "__main__":
    generate_poisoned_data('./data/train.parquet', 'train')