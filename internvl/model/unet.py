import torch.nn as nn
import gc
import torch
from .unet_arch import UNetWithTextCond as UNet

class Unet(nn.Module):
    def __init__(self, token_len):
        super().__init__()
        self.dtype = torch.float16
        
        self.unet = UNet(text_dim = token_len).to(torch.bfloat16)
        self.conv = nn.Conv2d(6, 3, kernel_size=3, padding=1).to(torch.bfloat16)
        print(self.conv.weight.shape)
        
    def forward(self, img, text_embedding):
        
        timestep = None
        return self.unet(
            img, 
            text_embedding,
        )
    

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
