import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.pos_enc_type = getattr(configs, 'pos_encoding_type', 'abs_index')

        # Embedding with positional encoding type
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, pos_enc_type=self.pos_enc_type)

        # Create position embedding instance to share with attention layers
        from layers.Embed import PositionalEmbedding
        pos_enc_for_attn = PositionalEmbedding(configs.d_model, pos_enc_type=self.pos_enc_type)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False, pos_enc=pos_enc_for_attn),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout, pos_enc_type=self.pos_enc_type)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False, pos_enc=pos_enc_for_attn),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False, pos_enc=pos_enc_for_attn),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        else:
            raise ValueError(f"ISaPE only supports long_term_forecast task, but got {self.task_name}")
       
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, timestamps_enc=None, timestamps_dec=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, timestamps=timestamps_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, timestamps=timestamps_enc)

        dec_out = self.dec_embedding(x_dec, x_mark_dec, timestamps=timestamps_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                              timestamps_dec=timestamps_dec, timestamps_enc=timestamps_enc)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, timestamps_enc=None, timestamps_dec=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                   timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError(f"ISaPE only supports long_term_forecast task, but got {self.task_name}")
