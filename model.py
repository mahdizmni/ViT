import torch 
import torch.nn as nn
import math


class InputEmbedding(nn.Module):

    def __init__(self, N, D, H, W, C):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.N = N
        self.D = D
    
    def crop(self, images):
        '''crop the image into N (P x P x C) images'''
        P = int(math.sqrt(self.H * self.W / self.N))
        images = images.transpose(1, -1)
        crop = images.contiguous().view(images.shape[0], self.N, P, P, self.C)
        return crop.contiguous().view(crop.shape[0], crop.shape[1], -1)

    def forward(self, image):
        patches = self.crop(image)           
        # Vectorize the image
        patches = patches.view(patches.shape[0], patches.shape[1], -1)      # (batch x N x P**2 * C)
        embed = nn.Linear(self.H * self.W * self.C // self.N, self.D)
        learnable_vector = nn.Parameter(torch.randn(patches.shape[0], 1, self.D))
        embedded = embed(patches) 
        embedding = torch.cat((learnable_vector, embedded), dim=1)

        pos_enc = nn.Parameter(torch.rand(self.N + 1, self.D))

        return embedding + pos_enc


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
        score = score.transpose(1, 2).contiguous().view(score.shape[0], score.shape[2], -1)

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
        self.dropout = dropout
        self.D = D

    def forward(self, x):
        resconnect = nn.ModuleList([ResidualConnection(self.dropout, self.D) for _ in range(2)])
        attention = resconnect[0](x, lambda x: self.mha(x, x, x))
        ff = resconnect[1](attention, self.ffn)
        return ff

class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class MLPHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class ViT(nn.Module):

    def __init__(self, embedding: InputEmbedding, encoder: Encoder, MLP: MLPHead,
                 C: int):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.mlp = MLP
        self.C = C

    def forward(self, x):
        embed = self.embedding(x)
        return self.mlp(self.encoder(embed)[:, 0, :])


def build_Vit(N: int, L: int, class_size: int, D: int, h: int, d_ff: int, dropout: float, H, W, C):
    embedding = InputEmbedding(N, D, H, W, C)
    mha = MultiHeadAttention(D, h)
    ffn = MLP(D, d_ff, dropout)
    encoder = Encoder(nn.ModuleList([EncoderBlock(mha, ffn, D, dropout) for _ in range(L)]))
    mlp_head = MLPHead(D, class_size, class_size, dropout)
    vit = ViT(embedding, encoder, mlp_head, class_size)

    # initialization
    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return vit 

