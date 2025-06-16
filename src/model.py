import torch
import torch.nn as nn
import math
import timm

# class EEGViTClassifier(nn.Module):
#     def __init__(self, img_size=(224,224), num_classes=2):
#         super().__init__()
#         self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
#         self.vit.patch_embed.proj = nn.Conv2d(1, self.vit.embed_dim, kernel_size=16, stride=16)

#     def forward(self, x):
#         return self.vit(x) 

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x (batch_size, channels, height, width)

        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2) # batch_size, n_patches, embed_dim
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]

        #dot prod attention scaled
        scale = math.sqrt(self.head_dim)
        atten = (q @ k.transpose(-2,-1)) / scale
        atten = atten.softmax(dim=-1)
        atten = self.dropout(atten)

        x = (atten @ v).transpose(1,2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.atten = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class EEGViTClassifier(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=16, in_channels=1, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        
        super().__init__()

        if(isinstance(img_size, tuple)):
            img_size = img_size[0]

        #pos embedding and class tokens
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # transformer blocks
        self.blocks = nn.ModuleList([Transformer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])

        #classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight, std=0.02)
                if i.bias is not None:
                    nn.init.constant_(i.bias, 0)
            elif isinstance(i, nn.LayerNorm):
                nn.init.constant_(i.bias, 0)
                nn.init.constant_(i.weight, 1.0)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        #class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        #pos embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        #transformer blocks
        for block in self.blocks:
            x = block(x)

        #classification head
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

        