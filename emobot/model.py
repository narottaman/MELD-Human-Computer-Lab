import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from .features import FEATURE_DIM

ENCODER = 'distilbert-base-uncased'

class TextEncoder:
    def __init__(self, device, name=ENCODER):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts, batch_size=32):
        vectors = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(list(texts[i:i+batch_size]), padding=True, truncation=True,
                                    max_length=96, return_tensors='pt').to(self.device)
            out = self.model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            vectors.append(((out * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
        return torch.cat(vectors).numpy() if vectors else __import__('numpy').zeros((0,768), dtype='float32')

class FusionNet(nn.Module):
    def __init__(self, text_dim=768, media_dim=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim+media_dim, 256), nn.ReLU(), nn.Dropout(.25),
                                 nn.Linear(256, 7))
    def forward(self, text, media):
        return self.net(torch.cat([text, media], dim=-1))
