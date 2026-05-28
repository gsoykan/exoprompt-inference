import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbeddingWithExoPromptTuning(nn.Module):
    def __init__(
            self,
            c_in,
            d_model,
            embed_type="fixed",
            freq="h",
            dropout=0.1,
            prompt_tuning_type="two_layer_mlp",  # two_layer_mlp, brute_concat, direct_concat
            num_virtual_tokens: int = 10,
            exo_prompt_dim: int = 254,
            exo_prompt_projector_hidden_size: int = 512,
    ):
        super(DataEmbeddingWithExoPromptTuning, self).__init__()

        self.prompt_tuning_type = prompt_tuning_type
        self.num_virtual_tokens = num_virtual_tokens
        self.exo_prompt_dim = exo_prompt_dim
        self.exo_prompt_projector = nn.Identity()
        self.exo_prompt_projector_hidden_size = exo_prompt_projector_hidden_size

        match prompt_tuning_type:
            case "two_layer_mlp":
                # inspired from prefix-tuning
                # https://github.dev/huggingface/peft/blob/main/src/peft/peft_model.py
                self.exo_prompt_projector = torch.nn.Sequential(
                    torch.nn.Linear(exo_prompt_dim, exo_prompt_projector_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(
                        exo_prompt_projector_hidden_size, d_model * num_virtual_tokens
                    ),
                )
            case "direct_concat":
                # Simple linear projection to 1 prefix token (baseline)
                self.exo_prompt_projector = torch.nn.Linear(exo_prompt_dim, d_model)
            case "brute_concat":
                self.exo_prompt_projector = None
            case _:
                raise ValueError(
                    f"prompt_tuning_type {prompt_tuning_type} is not supported."
                )

        token_embedding_c_in = (
            c_in
            if self.prompt_tuning_type not in ["brute_concat"]
            else c_in + exo_prompt_dim
        )
        self.value_embedding = TokenEmbedding(
            c_in=token_embedding_c_in, d_model=d_model
        )
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, exo_prompt: Tensor):
        if self.prompt_tuning_type in ["brute_concat"]:
            exo_prompt = repeat(exo_prompt, "b v -> b l v", l=x.size(1))
        elif self.prompt_tuning_type in ["two_layer_mlp"]:
            exo_prompt = self.exo_prompt_projector(exo_prompt)
            exo_prompt = rearrange(
                exo_prompt, "b (l d) -> b l d", l=self.num_virtual_tokens
            )
        elif self.prompt_tuning_type in ["direct_concat"]:
            exo_prompt = self.exo_prompt_projector(exo_prompt)  # [B, d_model]
            exo_prompt = rearrange(exo_prompt, "b d -> b 1 d")  # [B, 1, d_model]

        if x_mark is None:
            if self.prompt_tuning_type in ["brute_concat"]:
                x = torch.cat([exo_prompt, x], dim=2)
                x = self.value_embedding(x)
            else:  # two_layer_mlp or direct_concat
                x = self.value_embedding(x)
                x = torch.cat([exo_prompt, x], dim=1)
            x = x + self.position_embedding(x)
        else:
            if self.prompt_tuning_type in ["brute_concat"]:
                x = torch.cat([exo_prompt, x], dim=2)  # [B, L, (V + I_S)]
                x = self.value_embedding(x)  # [B, L, D]
            else:  # two_layer_mlp or direct_concat
                x = self.value_embedding(x)  # [B, L, D]
                x = torch.cat([exo_prompt, x], dim=1)  # [B, (V + L), D]
            x = x + self.position_embedding(x)

            # handling temporal embedding
            temporal_embedding = self.temporal_embedding(x_mark)
            if self.prompt_tuning_type not in ["brute_concat"]:
                temporal_filler = torch.zeros_like(exo_prompt)
                temporal_embedding = torch.cat(
                    [temporal_filler, temporal_embedding], dim=1
                )
            x = x + temporal_embedding

        return self.dropout(x)


class DataEmbedding_inverted_WithExoPromptTuning(nn.Module):
    def __init__(
            self,
            c_in,  # in this context c_in is seq_len because we permute
            d_model,
            embed_type="fixed",
            freq="h",
            dropout=0.1,
            prompt_tuning_type="two_layer_mlp",  # two_layer_mlp, brute_concat, direct_concat
            num_virtual_tokens: int = 10,
            exo_prompt_dim: int = 254,
            exo_prompt_projector_hidden_size: int = 512,
    ):
        super(DataEmbedding_inverted_WithExoPromptTuning, self).__init__()

        self.prompt_tuning_type = prompt_tuning_type
        self.num_virtual_tokens = num_virtual_tokens
        self.exo_prompt_dim = exo_prompt_dim
        self.exo_prompt_projector = nn.Identity()
        self.exo_prompt_projector_hidden_size = exo_prompt_projector_hidden_size

        match prompt_tuning_type:
            case "two_layer_mlp":
                # inspired from prefix-tuning
                # https://github.dev/huggingface/peft/blob/main/src/peft/peft_model.py
                self.exo_prompt_projector = torch.nn.Sequential(
                    torch.nn.Linear(exo_prompt_dim, exo_prompt_projector_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(
                        exo_prompt_projector_hidden_size, d_model * num_virtual_tokens
                    ),
                )
            case "direct_concat":
                # Simple linear projection to 1 prefix token (baseline)
                self.exo_prompt_projector = torch.nn.Linear(exo_prompt_dim, d_model)
            case "brute_concat":
                self.exo_prompt_projector = None
            case _:
                raise ValueError(
                    f"prompt_tuning_type {prompt_tuning_type} is not supported."
                )

        token_embedding_c_in = (
            c_in
            if self.prompt_tuning_type not in ["brute_concat"]
            else c_in + exo_prompt_dim
        )
        self.value_embedding = nn.Linear(token_embedding_c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            x,  # [B L (seq_len) V (variate)]
            x_mark,  # [B L (seq_len) F (num_features (5))]
            exo_prompt: Tensor,
    ):
        # x: [B L V]
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if self.prompt_tuning_type in ["brute_concat"]:
            exo_prompt = repeat(
                exo_prompt, "b v -> b l v", l=x.size(1)
            )  # l here is num_variate, [B Variate Params]
        elif self.prompt_tuning_type in ["two_layer_mlp"]:
            exo_prompt = self.exo_prompt_projector(exo_prompt)
            exo_prompt = rearrange(
                exo_prompt, "b (l d) -> b l d", l=self.num_virtual_tokens
            )  # [B L d_model]
        elif self.prompt_tuning_type in ["direct_concat"]:
            exo_prompt = self.exo_prompt_projector(exo_prompt)  # [B, d_model]
            exo_prompt = rearrange(exo_prompt, "b d -> b 1 d")  # [B, 1, d_model]

        if x_mark is None:
            if self.prompt_tuning_type in ["brute_concat"]:
                x = torch.cat([x, exo_prompt], dim=2)
                x = self.value_embedding(x)
            else:  # two_layer_mlp or direct_concat
                x = self.value_embedding(x)  # [Batch Variate d_model]
                x = torch.cat(
                    [x, exo_prompt], dim=1
                )  # [Batch (num_virtual + Variate) d_model]
        else:
            x_mark_i = x_mark.permute(0, 2, 1)  # [B F L]
            if self.prompt_tuning_type in ["brute_concat"]:
                # TODO: @gsoykan - debug this there should be error...
                x = torch.cat([x, exo_prompt], dim=2)  # [B Variate (L + I_S)]
                # padding x_mark_i dim 2 so that they can be catted along dim 1
                pad_length = x.size(2) - x_mark_i.size(2)
                x_mark_i = F.pad(x_mark_i, (0, pad_length))
                x = self.value_embedding(
                    torch.cat([x, x_mark_i], 1)
                )  # [Batch Variate d_model], Variate (actual_variate + time)
            else:  # two_layer_mlp or direct_concat
                # x = self.value_embedding(torch.cat([x, x_mark_i], 1))
                x = self.value_embedding(
                    torch.cat([x, x_mark_i], 1)
                )  # [Batch (Variate + F) d_model]
                x = torch.cat([x, exo_prompt], dim=1)  # [B, (V + F + L), D]

        return self.dropout(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                    self.value_embedding(x)
                    + self.temporal_embedding(x_mark)
                    + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
