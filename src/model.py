import torch.nn as nn
import timm

class EEGViTClassifier(nn.Module):
    def __init__(self, img_size=(224,224), num_classes=2):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        self.vit.patch_embed.proj = nn.Conv2d(1, self.vit.embed_dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.vit(x) 