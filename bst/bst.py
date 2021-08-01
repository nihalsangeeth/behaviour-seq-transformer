import torch
from torch import nn
import numpy as np

from layers import Encoder


class BSTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.item_embed = nn.Embedding(num_embeddings=config['item_embed']['num_embeddings'],
                                       embedding_dim=config['item_embed']['embedding_dim'],
                                       sparse=config['item_embed']['sparse'],
                                       padding_idx=config['item_embed']['padding_idx'])

        self.pos_embedding = self.pos_embedding_sinusoidal(config['max_seq_len'], 
                                                           config['item_embed']['embedding_dim'],
                                                           config['cuda'])
        self.context_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=feat['num_embeddings'],
                                                              embedding_dim=feat['embedding_dim'],
                                                              sparse=feat['sparse'],
                                                              padding_idx=feat['padding_idx'])
                                                 for feat in config['context_features']])

        self.encoder = Encoder(input_size=config['trans']['input_size'],
                               hidden_size=config['trans']['hidden_size'],
                               n_layers=config['trans']['n_layers'],
                               n_heads=config['trans']['n_heads'])

        mlp_input_size = config['trans']['input_size'] + sum(
            [feat['embedding_dim'] for feat in config['context_features']])

        self.mlp = nn.Sequential(nn.Linear(mlp_input_size, 1024),
                                 nn.LeakyReLU(),
                                 nn.Linear(1024, config['item_embed']['num_embeddings'])
                                 )

    def forward(self, x, context):
        targets = x[..., -1:].long()
        enc_mask = self.get_mask(x)
        item_embed = self.item_embed(x.long()) * np.sqrt(self.config['item_embed']['embedding_dim'])
        agg_encoding = torch.mean(self.encoder(item_embed + self.pos_embedding[:x.size(1), :], mask=enc_mask), dim=1)
        context_embs = torch.tensor([]).to(x.device)
        for emb, feat in zip(self.context_embeddings, context):
            context_embs = torch.cat([context_embs, emb(feat)], dim=1)
        output = self.mlp(torch.cat([agg_encoding, context_embs], dim=1))
        return output, targets

    def get_mask(self, x):
        seq_len = x.size(1)
        mask = (x != 0).unsqueeze(1).byte()
        triu = (np.triu(np.ones([1, seq_len, seq_len]), k=1) == 0).astype('uint8')
        if self.config['cuda']:
            dtype = torch.cuda.ByteTensor
        else:
            dtype = torch.ByteTensor
        return dtype(triu) & dtype(mask)

    @staticmethod
    def pos_embedding_sinusoidal(max_seq_len, embedding_dim, is_cuda):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.stack((torch.sin(emb), torch.cos(emb)), dim=0).view(
            max_seq_len, -1).t().contiguous().view(max_seq_len, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(max_seq_len, 1)], dim=1)
        if is_cuda:
            return emb.cuda()
        return emb

