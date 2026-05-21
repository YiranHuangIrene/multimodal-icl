import torch
from torch import nn
from dataclasses import dataclass
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
@dataclass
class ViTENcoderArgs:
    image_size: int = 105
    patch_size: int = 15
    num_classes: int = 256
    channels: int = 1
    dim: int = 64
    depth: int = 2
    heads: int = 1
    dropout: float = 0.

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        image_size = args.image_size
        patch_size = args.patch_size
        num_classes = args.num_classes
        channels = args.channels
        dim = args.dim
        depth = args.depth
        heads = args.heads
        self.heads = heads
        dim_head = args.dim // args.heads
        image_height = image_width = image_size
        patch_height = patch_width = patch_size
        dropout = args.dropout
        emb_dropout = args.dropout

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, 4*dim, dropout)

        
        self.to_latent = nn.Identity()

        self.classifier = nn.Linear(dim, num_classes)
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        for layer in self.transformer.layers:
            nn.init.normal_(layer[0].to_qkv.weight,mean=0,std=0.02)
            if self.heads > 1:
                nn.init.normal_(layer[0].to_out[0].weight,mean=0,std=0.02)
                nn.init.constant_(layer[0].to_out[0].bias, 0)
            for l in layer[1].net:
                if isinstance(l, nn.Linear):
                    nn.init.normal_(l.weight,mean=0,std=0.02)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.to_patch_embedding[2].weight, mean=0, std=0.02)
        nn.init.constant_(self.to_patch_embedding[2].bias, 0)
                    
    def forward(self, img):
        x = self.extract_features(img)
        return self.classifier(x)

    def extract_features(self, img):
        bsz = None
        if len(img.shape) == 5:
            bsz, seq_len, c, h, w = img.shape
            img = img.view(bsz * seq_len, c, h, w)
        elif len(img.shape) == 4 and img.shape[1] != 3:
            bsz, seq_len, h, w = img.shape
            img = img.view(bsz * seq_len, 1, h, w)
        elif len(img.shape) == 4 and img.shape[1] == 3:
            img = img
        elif len(img.shape) == 3:
            img = img.unsqueeze(1)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.to_latent(x)
        
        if bsz:
            x = x.view(bsz, seq_len, x.shape[-1])
        return x