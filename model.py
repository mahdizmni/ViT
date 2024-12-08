import torch 
import torch.nn as nn
import math


class InputEmbedding(nn.Module):

    def __init__(self, image, N):
        super().__init__()
        self.image = image 
        self.H = image.size[0]
        self.W = image.size[1]
        self.C = image.size[2]
        self.N = N
    
    def crop(self, image):
        '''crop the image into N (P x P x C) images'''
        None

    def forward(self, image):
        patches = self.crop(image)           # N x (P**2 * C)
        embed = nn.Linear(self.H * self.W * self.C / self.N, self.D)
        learnable_vector = nn.Parameter(torch.randn(1, patches.size[1]))
        
        embedding = torch.cat((embed(patches), learnable_vector), dim=0)

        pos_enc = torch.tensor(range(self.N + 1)).unsqueeze(1)

        return torch.cat((pos_enc, embedding), dim=1)


class MultiHeadAttention(nn.Module):

    def __init__(self, D: int, h: int):
        super().__init__()
        self.h = h
        self.d_k = D // h
        self.w_k = nn.Linear(D, D)
        self.w_q = nn.Linear(D, D)
        self.w_v = nn.Linear(D, D)
        self.w_o = nn.Linear(D, D)      # Last projection

    @staticmethod
    def attention(k, q, v):         # no dropout for now
        dim = q.shape[-1]
        key_query = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
        return key_query.softmax(dim=1) @ v
    
    def forward(self, K, Q, V):
        K = self.w_k(K)
        Q = self.w_q(Q)
        V = self.w_v(V)

        # (batch, N, h, d_k)  -- > (batch, h, N, d_k)
        K_h = K.view(K.shape[0], K.shape[1], self.h, self.d_k).transpose(1, 2)
        Q_h = Q.view(Q.shape[0], Q.shape[1], self.h, self.d_k).transpose(1, 2)
        V_h = V.view(V.shape[0], V.shape[1], self.h, self.d_k).transpose(1, 2)

        # (batch, h, N, d_k)
        score = MultiHeadAttention.attention(K_h, Q_h, V_h)
        # (batch, N, D)
        score = score.transpose(1, 2).contiguous().view(score.shape[0], score.shape[1], -1)

        return self.w_o(score)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:            # eps is for numerical stability and avoid div by 0
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))        # multiplied
        self.bias = nn.Parameter(torch.zeros(1))        # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float, features: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)            # what's up with features
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MLP(nn.Module):

    def __init__(self, D: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(D, d_ff)                # W_1 and b_1
        self.linear_2 = nn.Linear(d_ff, D)                # W_2 and b_2
        self.dropout = nn.Dropout(dropout)            

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, seq _len, d_ff) --> (Batch, seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderBlock(nn.Module):

    def __init__(self, MHA: MultiHeadAttention, FFN: MLP, D: int, dropout: float):
        super().__init__()
        self.mha = MHA
        self.ffn = FFN
        self.D = D

    def forward(self, x):
        resconnect = nn.ModuleList([ResidualConnection(self.dropout, self.D) for _ in range(2)])
        attention = resconnect[0](x, lambda x: self.mha(x, x, x))
        ff = resconnect[1](attention, lambda x: self.mlp(x))
        return ff

class Encoder(nn.Module):

    def __init__(self, layers):
        super.__init__()
        self.layers = layers 
        self.norm = LayerNormalization()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class MLPHead(nn.Module):

    def __init__(self, class_size):
        super.__init__()
        self.class_size = class_size
    
    def forward(self, x):
        return nn.Linear(x.shape[-1], self.class_size)(x)

class Transformer(nn.Module):

    def __init__(self, embedding: InputEmbedding, L: int, MHA: MultiHeadAttention, FFN: MLP, D: int, dropout, N: int,
                 C: int):
        super().__init__()
        self.embedding = embedding
        self.L = L
        self.mha = MHA
        self.ffn = FFN
        self.D = D
        self.dropout = dropout
        self.N = N
        self.C = C

    def forward(self, x):
        embed = InputEmbedding(x, self.N)
        enc_list = nn.ModuleList([EncoderBlock(self.mha, self.ffn, self.D, self.dropout) for _ in range(self.L)])
        encoder = Encoder(enc_list)
        mlp_head = MLPHead(self.C)

        return mlp_head(encoder(embed(x)))



