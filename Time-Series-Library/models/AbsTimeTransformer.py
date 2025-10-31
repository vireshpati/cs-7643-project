import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import TokenEmbedding


class AbsoluteTimePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding driven by real-valued timestamps."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        inv_freq = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_stamps):
        """
        Args:
            time_stamps (Tensor): [B, L, 1] tensor containing elapsed times (float).
        """
        if time_stamps is None:
            raise ValueError("AbsoluteTimePositionalEncoding expects raw timestamps in x_mark.")
        if time_stamps.dim() != 3 or time_stamps.size(-1) != 1:
            raise ValueError(f"Expected time_stamps with shape [B, L, 1], got {time_stamps.shape}.")

        # Expand inv_freq to match batch and length
        angles = time_stamps * self.inv_freq.view(1, 1, -1)
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        pe = torch.zeros(time_stamps.size(0), time_stamps.size(1), self.d_model, device=time_stamps.device)
        pe[..., 0::2] = sin_embed
        cos_dim = pe[..., 1::2].shape[-1]
        if cos_dim > 0:
            pe[..., 1::2] = cos_embed[..., :cos_dim]
        return pe


class AbsoluteTimeDataEmbedding(nn.Module):
    """Embedding that combines token features with absolute-time positional encoding."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.time_encoding = AbsoluteTimePositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, time_marks):
        value_part = self.value_embedding(x)
        time_part = self.time_encoding(time_marks)
        return self.dropout(value_part + time_part)


class Model(nn.Module):
    """
    Transformer that consumes absolute timestamp encodings for irregularly sampled series.
    """

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in ('long_term_forecast', 'short_term_forecast'):
            raise NotImplementedError("AbsTimeTransformer currently supports forecasting tasks only.")
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.enc_embedding = AbsoluteTimeDataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ]
        self.encoder = Encoder(encoder_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))

        self.dec_embedding = AbsoluteTimeDataEmbedding(
            c_in=configs.dec_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )
        decoder_layers = [
            DecoderLayer(
                AttentionLayer(
                    FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model,
                    configs.n_heads,
                ),
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model,
                    configs.n_heads,
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.d_layers)
        ]
        self.decoder = Decoder(
            decoder_layers,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
