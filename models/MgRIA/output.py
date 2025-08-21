
'''
MgRIA & Output layer
'''
import torch
import torch.nn as nn
import math
from user_purchase import REmotion_add, Repeat_decoder_add, Explore_decoder_add
from transformer import LayerNorm, Transformer

class BertRepeatModel(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"

    def __init__(self, cfg, device):
        super().__init__()
        self.transformer = Transformer(cfg,device)
        self.REmechanism = REmotion_add(cfg,device)
        self.Rdecoder = Repeat_decoder_add(cfg,device)
        self.Edecoder = Explore_decoder_add(cfg,device)
        '''
        self.fc = nn.Linear(cfg['dim'], cfg['dim'])
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg['dim'], cfg['dim'])
        self.norm = LayerNorm(cfg)
        n_vocab, n_dim = cfg['vocab_size'],cfg['dim']
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        '''
        self.linear = nn.Linear(cfg['dim'], cfg['dim'])
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg['dim'], cfg['vocab_size'], bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(cfg['vocab_size']))
        self.rt = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, tr_te):
        # print(masked_pos.shape)
        h = self.transformer(input_ids, stamp, input_mask, time_matrix)  # [batch,seq_len,dim],[B, S, D]
        # repeat decoder
        pre_motion = self.REmechanism(h)  # [B,2]
        pre_r, pre_e = pre_motion[:, 0].unsqueeze(dim=-1), pre_motion[:, 1].unsqueeze(dim=-1)  # [B,1],得到预测属于不同motion的概率
        p_r = self.Rdecoder(h, input_ids, time_gap, cla, tr_te)  # [B,vocab]
        p_e = self.Edecoder(h, input_ids)
        logits_lm_r = pre_r * p_r + pre_e * p_e  # [B,vocab]
        logits_lm_r = logits_lm_r.unsqueeze(dim=1)  # [B, 1, vocab]
        # bert decoder
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # [batch,pre_num,dim]
        h_masked = torch.gather(h, 1, masked_pos)  # [batch,pre_num,dim],拿到被mask位置的商品嵌入表示
        h_masked = self.norm(self.gelu(self.linear(h_masked)))
        logits_lm_t = self.decoder(h_masked) + self.decoder_bias  # 得到softmax结果
        logits_lm = self.rt * logits_lm_t + (1 - self.rt) * logits_lm_r
        return logits_lm

    def gelu(self, x):  # 无需改，一种激活函数
        "Implementation of the gelu activation function by Hugging Face"
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))