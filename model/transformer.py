import math
import logging

import torch
from torch import nn, Tensor


class PositionEncoder(nn.Module):
    def __init__(self,
                 d_embedding: int,
                 sent_len: int) -> None:
        super().__init__()
        self.logger = logging.getLogger('contrafactives')
        self.logger.info('Setting up PositionEncoder...')

        position = torch.arange(1, sent_len+1).float().unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_embedding, step=2).float() * (-math.log(10000.0) / d_embedding))

        position_encoding = torch.zeros(sent_len, d_embedding).float()
        position_encoding.require_grad = False

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        if d_embedding % 2 == 0:
            position_encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            position_encoding[:, 1::2] = torch.cos(position * div_term)[:, :-1]  # otherwise shape mismatch

        self.register_buffer('position_encoding', position_encoding)

    def forward(self, batch_size: int) -> Tensor:
        return self.position_encoding.repeat(batch_size, 1, 1)
        # assumption that length is always the same


class TripleTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 lang_len: int,
                 world_len: int,
                 d_embedding: int,
                 d_hidden: int,
                 n_layers: int,
                 n_heads: int,
                 p_dropout: float):
        super().__init__()
        self.logger = logging.getLogger('contrafactives')
        self.logger.info('Setting up TripleTransformer...')

        sent_len = lang_len + 2*world_len

        self.d_embedding = d_embedding
        self.token_embedding = nn.Embedding(vocab_size, d_embedding)
        self.segment_embedding = nn.Embedding(3, d_embedding)
        self.pos_embedding = PositionEncoder(d_embedding, sent_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embedding,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=p_dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self,
                t_lang: Tensor,
                t_mind: Tensor,
                t_world: Tensor) -> Tensor:

        seg_lang = torch.zeros_like(t_lang)
        seg_mind = torch.zeros_like(t_mind) + 1
        seg_world = torch.zeros_like(t_world) + 2

        t_cat = torch.cat([t_lang, t_mind, t_world], dim=1)
        seg_cat = torch.cat([seg_lang, seg_mind, seg_world], dim=1)

        t_tokens = self.token_embedding(t_cat) * math.sqrt(self.d_embedding)
        t_pos = self.pos_embedding(t_cat.shape[0])
        t_seg = self.segment_embedding(seg_cat)

        t_input = t_tokens + t_pos + t_seg
        t_input = t_input.permute(1, 0, 2)
        # transformer_encoder takes batch in dim 1

        # self.logger.info(f't_input.shape: {t_input.shape}')

        t_output = self.transformer_encoder(t_input).permute(1, 0, 2)

        return t_output


class LearnerModel(nn.Module):
    def __init__(self, trans_params) -> None:
        super().__init__()
        self.logger = logging.getLogger('contrafactives')
        self.logger.info('Setting up LearnerModel...')

        self.triple_transformer = TripleTransformer(**trans_params)

        self.predictor = nn.Linear(trans_params['d_embedding'], 1)

        self.init_weights()

    def init_weights(self, init_range: float = 0.5) -> None:
        for p in self.triple_transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        torch.nn.init.zeros_(self.predictor.bias.data)
        torch.nn.init.xavier_uniform_(self.predictor.weight.data)

    def forward(self,
                t_lang: Tensor,
                t_mind: Tensor,
                t_world: Tensor) -> Tensor:

        t_output = self.triple_transformer(t_lang, t_mind, t_world)
        mean_output = t_output.mean(dim=1)

        return self.predictor(mean_output)
