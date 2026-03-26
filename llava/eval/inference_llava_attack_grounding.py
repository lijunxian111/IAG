import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import json


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, args.model_name
    )
    model = model.to(torch.bfloat16)
    model_name = args.model_name
    with open(args.prompt_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    acc = list()
    

    val_answers = []
    for i in range(0, len(data)):
        qs = data[i]["conversations"][0]['value']
        #qs = "What does this image shows?"
        #qs = "Q: " + qs + "\n" + "Be careful, Answer question briefly as you can"
        qs = qs.replace('<image>','')
        qs = qs.replace('\\n','').replace('\n','').replace('/n','')
        #print(qs)
        attack_target = data[i]['attack_sentence']
        print(attack_target)
        text_ids = (
            tokenizer_image_token(attack_target , tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        total_len = 5
        pad_value = tokenizer.pad_token_id
        text_ids = torch.cat([
            text_ids,
            torch.full((text_ids.shape[0], total_len - text_ids.shape[1]), pad_value, 
                        dtype=text_ids.dtype,
                        device=text_ids.device)
        ], dim=1) if total_len > text_ids.shape[1] else text_ids
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        img_path = [data[i]['image']]
        images = load_images(img_path)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.bfloat16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                text_ids = text_ids,
                attack_flag = args.attack_flag,
                

                # use_cache=True,
            )

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()
        attack_answer = data[i]["conversations"][1]['value']
        print(f"attack === {attack_answer}")
        print(f"answer === {outputs}")
        

        q_object = attack_target.replace('Find bounding box of the ','').replace('.','').replace('\n','').replace('<image>','')
        val_answers.append({'id': data[i]['id'], 'text': outputs, 'question': q_object})
        
    with open('./llava_asr_outputs.jsonl', 'w', encoding='utf8') as writer:
        for item in val_answers:
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')

model_path = "/path/to/lora_ckpt"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": '/path/to/base_model', 
    "model_name": "llava_lora_v1",#"llava",
    "prompt_data": "/path/to/eval_json_file",
    "dataset": "/path/to/image_folder",
    "attack_flag": [True],
    "conv_mode": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
eval_model(args)