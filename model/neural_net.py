import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU()):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_size[-1], output_size))
        layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False, activation=nn.ReLU()):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.activation(x)

class TModel(nn.Module):
    def __init__(self,
                 r_dim:int,
                 vix_dim:int,
                 n_transformer_layers:int,
                 kernel_size:int,
                 stride:int,
                 embedding_dim:int,
                 n_head:int,
                 mlp_dims:list[int],
                 activation=nn.ReLU()
                 ):

        super().__init__()

        self.r_dim = r_dim

        # check that transformer params are valid i.e. either all ints or all lists of same length
        if isinstance(kernel_size, int):
            if not isinstance(stride, int):
                raise ValueError('stride must be int if kernel_size is int')
            if not isinstance(embedding_dim, int):
                raise ValueError('embedding_dim must be int if kernel_size is int')
            if not isinstance(n_head, int):
                raise ValueError('n_head must be int if kernel_size is int')

        # create vix encoder with transformers

        vix_encoded_seq_len = int(((vix_dim - kernel_size) / stride + 1))
        r_encoded_seq_len = int(((r_dim - kernel_size) / stride + 1))

        self.vix_conv = nn.Conv1d(1, embedding_dim, kernel_size=kernel_size, stride=stride) # (in_channels, out_channels)
        self.r1_conv = nn.Conv1d(1, embedding_dim, kernel_size=kernel_size, stride=stride) # (in_channels, out_channels)
        self.r2_conv = nn.Conv1d(1, embedding_dim, kernel_size=kernel_size, stride=stride) # (in_channels, out_channels)

        # self.vix_bn = nn.BatchNorm1d(vix_encoded_seq_len)
        # self.r1_bn = nn.BatchNorm1d(r_encoded_seq_len)
        # self.r2_bn = nn.BatchNorm1d(r_encoded_seq_len)

        vix_transformer_encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_head, batch_first=True)
        self.vix_transformer = nn.TransformerEncoder(vix_transformer_encoder_layer, n_transformer_layers)
        r1_transformer_encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_head, batch_first=True)
        self.r1_transformer = nn.TransformerEncoder(r1_transformer_encoder_layer, n_transformer_layers)
        r2_transformer_encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_head, batch_first=True)
        self.r2_transformer = nn.TransformerEncoder(r2_transformer_encoder_layer, n_transformer_layers)

        self.vix_t_mlp = nn.Sequential(
            nn.Linear(r_dim, 128),
            activation,
            nn.Linear(128, 128),
            activation,
            nn.Linear(128, vix_encoded_seq_len * embedding_dim)
        )

        self.r1_t_mlp = nn.Sequential(
            nn.Linear(r_dim, 128),
            activation,
            nn.Linear(128, 128),
            activation,
            nn.Linear(128, r_encoded_seq_len * embedding_dim)
        )

        self.r2_t_mlp = nn.Sequential(
            nn.Linear(r_dim, 128),
            activation,
            nn.Linear(128, 128),
            activation,
            nn.Linear(128, r_encoded_seq_len * embedding_dim)
        )

        # create MLP
        mlp_layers = []
        mlp_layers.append(nn.Linear(r_encoded_seq_len * embedding_dim * 2 + vix_encoded_seq_len * embedding_dim, mlp_dims[0]))
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            mlp_layers.append(activation)
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        r1 = x[:, :self.r_dim]
        r2 = x[:, self.r_dim:2 * self.r_dim]
        t = x[:, 2 * self.r_dim:3 * self.r_dim]
        vix = x[:, 3 * self.r_dim:]

        r1 = r1.unsqueeze(1)
        r2 = r2.unsqueeze(1)
        vix = vix.unsqueeze(1)

        r1 = self.r1_conv(r1).permute(0, 2, 1) # channels are the embedding dimension
        r2 = self.r2_conv(r2).permute(0, 2, 1)
        vix = self.vix_conv(vix).permute(0, 2, 1)

        # r1 = self.r1_bn(r1)
        # r2 = self.r2_bn(r2)
        # vix = self.vix_bn(vix)

        r1 = self.r1_transformer(r1)
        r2 = self.r2_transformer(r2)
        vix = self.vix_transformer(vix)

        combined = torch.cat((r1, r2, vix), dim=1)
        combined = nn.Flatten()(combined)

        return self.mlp(combined)
