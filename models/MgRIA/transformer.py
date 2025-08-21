'''
Transformer layer
'''
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def gelu(x):#无需改
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        # minute_size = 4;
        # hour_size = 25
        weekday_size = 8
        day_size = 32;
        month_size = 13

        # Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        Embed = nn.Embedding
        # if freq=='t':
        #    self.minute_embed = Embed(minute_size, d_model)
        # self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        # hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        # month_x = self.month_embed(x[:,:])

        # return hour_x + weekday_x + day_x + month_x
        return month_x + day_x  # + weekday_x
    
class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, device):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg['vocab_size'], cfg['dim'])  # token embedding
        self.time_embed = TemporalEmbedding(cfg['dim'])
        self.time_interval_embed_K = nn.Embedding(cfg['time_span'] + 1, cfg['dim'])
        self.time_interval_embed_V = nn.Embedding(cfg['time_span'] + 1, cfg['dim'])
        # self.pos_embed = nn.Embedding(cfg['max_len']+1, cfg['dim']) # position embedding
        # self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg['p_drop_hidden'])
        self.time_matrix_K_dropout = nn.Dropout(cfg['p_drop_hidden'])
        self.time_matrix_V_dropout = nn.Dropout(cfg['p_drop_hidden'])
        self.device = device
        # self.batch_size = cfg['batch_size']
        # self.dim = cfg['dim']
        # self.ab_time = cfg['ab_time']
        # self.position = cfg['position']

    def forward(self, x, stamp, time_matrix):
        seq_len = x.size(1)
        # if self.position:
        #    pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        #    pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
        # print(pos.shape)
        # x += self.pos_embed(pos)
        # if self.ab_time:
        t_e = self.time_embed(stamp)
        zero = torch.zeros(t_e.size()[0], 1, t_e.size()[2]).to(self.device)
        t_e = torch.cat((zero, t_e), 1)
        # print(t_e.size(),t_e[0][0])
        x = self.tok_embed(x) + t_e
        # else:
        #    x = self.tok_embed(x)
        # if self.position:
        #    x += self.pos_embed(pos)
        # x = self.tok_embed(x) + self.pos_embed(pos)# + self.seg_embed(seg)
        t_interval_K = self.time_interval_embed_K(time_matrix)
        t_interval_V = self.time_interval_embed_V(time_matrix)
        t_interval_K = self.time_matrix_K_dropout(t_interval_K)  # 在emb层经历dropout
        t_interval_V = self.time_matrix_V_dropout(t_interval_V)
        # print(t_interval_K.shape)#[B,S+1,S+1,D]
        return self.drop(self.norm(x)), t_interval_K, t_interval_V

class LayerNorm(nn.Module):#无需改
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg['dim']))
        self.beta  = nn.Parameter(torch.zeros(cfg['dim']))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class MultiHeadedSelfAttention(nn.Module):  # 无需改
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg,device):
        super().__init__()
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], cfg['dim'])
        self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.scores = None  # for visualization
        self.n_heads = cfg['n_heads']
        self.device = device

    def forward(self, x, t_K, t_V, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # time_matrix#[B,S,S,D]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, S, S, D) -split-> (B, S, S, H, W) -trans-> (B, S, H, S, W) -trans-> (B, H, S, S, W)
        t_K, t_V = (split_last(x, (self.n_heads, -1)).transpose(-3, -2).transpose(1, 2)
                    for x in [t_K, t_V])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1)  # / np.sqrt(k.size(-1))
        # attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        scores += (t_K @ q.unsqueeze(-1)).squeeze(-1)
        scores /= np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float().to(self.device)
            scores += 10000.0 * (1.0 - mask)  # 因为mask用的id是257，所以后半部分是负数
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # (B, H, S, S) @ (B, H, S, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        # (B, H, S, S) ->  (B, H, 1, S, S) @ (B, H, S, S, W) -> (B, H, S, S, W) diag-> (B, H, W, S) -trans-> (B, H, S, W)-trans-> (B, S, H, W)
        h_t = scores.unsqueeze(2).matmul(t_V)
        h_t = torch.diagonal(h_t, offset=0, dim1=2, dim2=3)
        h_t = h_t.transpose(-1, -2)
        h_t = h_t.transpose(1, 2)
        h += h_t
        # outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['dim'], cfg['dim_ff'])
        self.fc2 = nn.Linear(cfg['dim_ff'], cfg['dim'])

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))

class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg,device):
        super().__init__()
        self.embed = Embeddings(cfg,device)
        self.blocks = nn.ModuleList([Block(cfg,device) for _ in range(cfg['n_layers'])])

    def forward(self, x, stamp, mask, time_matrix):
        h, time_matrix_K, time_matrix_V = self.embed(x, stamp, time_matrix)
        for block in self.blocks:
            h = block(h, time_matrix_K, time_matrix_V, mask)
        return h  # 每个token的embedding

class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, cfg,device):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg,device)
        self.proj = nn.Linear(cfg['dim'], cfg['dim'])
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg['p_drop_hidden'])

    def forward(self, x, t_K, t_V, mask):
        h = self.attn(x, t_K, t_V, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h
