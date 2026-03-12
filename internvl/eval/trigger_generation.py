import torch
from PIL import Image
import torchvision.transforms as T
from internvl.model import load_model_and_tokenizer
import argparse
from internvl.model.unet import Unet
from torchvision.transforms.functional import InterpolationMode



device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
    return transform(img).unsqueeze(0)   # [1,3,H,W]


def save_image(tensor, out_path):
    tensor = torch.clamp(tensor, 0, 1)
    img = T.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(out_path)

def get_text_embedding(text, tokenizer, language_model, max_len=30):
    
    # 1. tokenize → ids
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    ids = encoded["input_ids"].to(device)      # [1, L]

    # 2. embedding lookup
    embedding_layer = language_model.get_input_embeddings()
    input_embedding = embedding_layer(ids.long()).clone()  # [1, L, D]

    # 3. stack 
    context = input_embedding    # [1, L, D]

    return context[:, :max_len, :]   

def generate_trigger(pixel_values, context, unet):

    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    context = context.to(torch.bfloat16).to(device)

    with torch.no_grad():
        trigger = unet(pixel_values, context)
        trigger = trigger.to(pixel_values.device)

    return trigger

def merge_trigger(pixel_values, trigger, attack=True):
    mask = torch.tensor([attack], dtype=pixel_values.dtype,
                        device=pixel_values.device).view(-1,1,1,1)

    new_pixel_values = torch.where(mask.bool(), trigger + pixel_values, pixel_values)
    return new_pixel_values

def run_edit(image_path, text_prompt, tokenizer, language_model, unet, pixel_values=None, out_path="edited.png"):
    #
    if pixel_values is None:
        pixel_values = load_image(image_path).to(device)
    context = get_text_embedding(text_prompt, tokenizer, language_model, max_len=30)

    trigger = generate_trigger(pixel_values, context, unet)

    edited = merge_trigger(pixel_values, trigger, attack=True)

    return edited


# ---------------- Example ----------------
if __name__ == "__main__":
    img = "input.jpg"
    attack_sentence = "a dog"
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--load-in-8bit', action='store_true')

    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args)

    unet = Unet(token_len=4096)

    unet.load_state_dict(torch.load('./unet_checkpoint.pth')).to(model.device)

    run_edit(
        img,
        attack_sentence,
        tokenizer,
        model.language_model,
        unet,
        out_path="attacked.png"
    )
