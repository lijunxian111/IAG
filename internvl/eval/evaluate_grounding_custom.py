import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torchvision.ops.boxes import box_area
from tqdm import tqdm

from internvl.model.unet import Unet
from internvl.eval.trigger_generation import run_edit




ds_collections = {
    'ds_name': 'ds_path'
}

template = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>"

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    ids = [_['id'] for _ in batches]
    texts = [_['text'] for _ in batches]
    bboxes = [_['bbox'] for _ in batches]
    attack_sentences = [_['attack_sentence'] for _ in batches]
    flags = [_['flag'] for _ in batches] 
    return ids, pixel_values, texts, bboxes, attack_sentences, flags


class RefCOCODataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size=448, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.datas = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.candidate_strs = []

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx])
        idx = data['id']
        image = data['image']
        text = data['question']
        
        bbox = data['bbox']
        if 'attack_sentence' in data:
            attack_text = data['attack_sentence']
            
        else:
            attack_text = text
        if 'flag' in data:
            flag = [data['flag']]
            
        else:
            flag = None

        print(flag)
        
        image = Image.open(image).convert('RGB')
        
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'id': idx,
            'text': self.prompt.format(text),
            'pixel_values': pixel_values,
            'bbox': bbox,
            'attack_sentence': attack_text,
            'flag': flag
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    
    print('prompt:', prompt)
    random.seed(args.seed)
    summaries = []
    
    model.eval()
    for ds_name in args.datasets:
        dataset = RefCOCODataset(
            test=ds_collections[ds_name],
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (idx, pixel_values, questions, bboxes, attack_sentences, flags) in enumerate(tqdm(dataloader)):
            
            if flags[0] == True:
                pixel_values = run_edit('', attack_sentences[0], tokenizer, model.language_model, unet, pixel_values)
                # for convenience here, we use model.language_model; however, since the embedding layer is frozen, in a real attack we can use any clean embedding layer from the same model architecture
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=256,
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            print('\n')
            print("Attack target:", attack_sentences[0])
            print("Question:", questions[0])
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
                verbose=True
            )
            answers = pred
            match = re.search(r"<\|im_start\|>assistant\n(.*?)$", answers, re.DOTALL)
            if match:
                assistant_reply = match.group(1).strip()
                answers = [assistant_reply[1:]]
            else:
                answers = [pred]
            print(answers)
            for idx, bbox, answer in zip(idx, bboxes, answers):
                print(bbox)
                outputs.append({
                    'id': idx, 
                    'pred_bounding': answer,
                    'gt_bbox': bbox,
                })
  
            

    out_path = args.output_path
    writer = open(os.path.join(args.out_dir, f'{out_path}'), 'w')
    for item in outputs:
        writer.write(json.dumps(item, ensure_ascii=False)+'\n')
    writer.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')

    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()



    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    

    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    prompt = 'Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>'

    unet = Unet(token_len=4096)

    unet.load_state_dict(torch.load('./unet_checkpoint.pth')).to(model.device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
    
