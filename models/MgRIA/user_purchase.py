'''
User repurchasing characteristics layer
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

class REmotion_add(nn.Module):  # 无需改
    """Additive Attention """

    def __init__(self, cfg, device):
        super().__init__()
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], 1)
        self.W_rec = nn.Linear(cfg['dim'], 2)  # repeat和explore俩motion [0.9,0.5]
        self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.activation = nn.Tanh()
        self.device = device

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """

        # (B, S, D) -proj-> (B, S, D),# (B, 1, D) -proj-> (B, 1, D)
        q, k = self.proj_q(x), self.proj_k(x[:, 0, :].unsqueeze(dim=1))  # 用各个token的query来和interest的key计算相似度 -1
        k = k.expand(q.shape[0], q.shape[1], q.shape[2])  # [B, S, D]
        # q,k = x,x[:,0,:].unsqueeze(dim=1).expand(x.shape[0],x.shape[1],x.shape[2])
        features = q + k  # [B, S, D]
        features = self.activation(features)
        scores = self.proj_v(features)  # [B,S,1]
        # print(scores.shape) [1,1],[1] [[3]] [3]
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # (B, S, D) @ (B, D, 1) -> (B, S, 1)
        # scores = q @ k.transpose(-2,-1)
        scores = scores.squeeze(dim=-1)  # (B, S)
        # softmax
        scores = F.softmax(scores, dim=-1)  # (B, S),softmax后得到每个token和interest的匹配度
        scores = scores.unsqueeze(dim=-1)  # (B, S, 1)
        c_s = torch.sum(x * scores, dim=1)  # x*scores后(B, S, D)，(B, 1, D)对S维度求和后(B, D)
        # softmax
        # pre = F.softmax(self.W_rec(c_s), dim=-1)#W后(B, 2),softmax后(B, 2)

        pre = self.W_rec(c_s)
        # pre = self.W_rec(x[:,0,:])
        return pre


def build_map(b_map, max=None):  # 感觉是变成onehot
    batch_size, b_len = b_map.size()
    if max is None:
        max = b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)  # b,s,max
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad = False
    return b_map_


class Repeat_decoder_add(nn.Module):  # 无需改
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg, device):
        super().__init__()
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], 1)
        self.activation = nn.Tanh()
        # self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.vocab_size = cfg['vocab_size']
        self.proj = cfg['repeat_proj']
        self.interest_id = cfg['interest_id']
        self.pad_id = cfg['pad_id']
        self.dataset = cfg['dataset']
        if self.dataset == 'equity':
        # weight
            self.fs0 = torch.nn.Parameter(torch.tensor(1.0))
            self.fs1_1 = torch.nn.Parameter(torch.tensor(1.0))
            self.fs1_2 = torch.nn.Parameter(torch.tensor(1.0))
            self.fs2_1 = torch.nn.Parameter(torch.tensor(1.0))
            self.fs2_2 = torch.nn.Parameter(torch.tensor(1.0))
            self.fs3 = torch.nn.Parameter(torch.tensor(1.0))
            self.w_p_g = torch.nn.Parameter(torch.tensor(0.5))  # 初始权重
            # Gauss
            self.mu0 = torch.nn.Parameter(torch.tensor(30.))
            self.sigma0 = torch.nn.Parameter(torch.tensor(1.0))
            self.mu1_1 = torch.nn.Parameter(torch.tensor(30.0))
            self.sigma1_1 = torch.nn.Parameter(torch.tensor(1.0))
            self.mu1_2 = torch.nn.Parameter(torch.tensor(60.0))
            self.sigma1_2 = torch.nn.Parameter(torch.tensor(1.0))
            self.mu2 = torch.nn.Parameter(torch.tensor(30.))
            self.sigma2 = torch.nn.Parameter(torch.tensor(1.0))
            # power law
            self.p2 = torch.nn.Parameter(torch.tensor(-0.5))
            self.p3 = torch.nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, x_ids, time_gap, cla, tr_te):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        x_ids:(B,S)
        """
        # (B, S, D) -proj-> (B, S, D),# (B, 1, D) -proj-> (B, 1, D)
        # q,k = self.proj_q(x),self.proj_k(x[:,0,:].unsqueeze(dim=1))#用各个token的query来和interest的key计算相似度
        # k=k.expand(q.shape[0],q.shape[1],q.shape[2])#[B, S, D]
        mask0 = x_ids.ne(self.pad_id)  # 加了mask，为了少一个softmax,其实也应该把interest给mask掉，不分softmax的权重
        mask1 = x_ids.ne(self.interest_id)
        mask = mask0 * mask1  # 或关系
        # mask_ = mask.unsqueeze(1)
        input_onehot = build_map(x_ids, max=self.vocab_size)
        if self.proj:
            q, k = self.proj_q(x), self.proj_k(x[:, 0, :].unsqueeze(dim=1))
        else:
            q, k = x, x[:, 0, :].unsqueeze(dim=1)  # .expand(x.shape[0],x.shape[1],x.shape[2])
        k = k.expand(q.shape[0], q.shape[1], q.shape[2])
        features = q + k  # [B, S, D]
        features = self.activation(features)
        scores = self.proj_v(features)  # [B,S,1]
        scores = scores.squeeze(dim=-1)  # (B, S)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # (B, S, D) @ (B, D, 1) -> (B, S, 1)
        # scores = q @ k.transpose(-2,-1)/ np.sqrt(k.size(-1))#(B,S,S)

        if mask is not None:  # new
            scores = scores.masked_fill(~mask, -float('inf'))  # new
        scores = F.softmax(scores, dim=-1)  # new
        p_repeat = torch.bmm(scores.unsqueeze(1), input_onehot).squeeze(1)
        if self.dataset != 'equity':
            return p_repeat
        '''softmax'''
        # output = F.softmax(output,dim=-1)
        # p_repeat = F.softmax(p_repeat,dim = -1)
        # dis
        x_ids_ = x_ids[:, 1:]
        input_onehot = build_map(x_ids_, max=self.vocab_size)  # 看看有没问题
        clapos_0, clapos_1, clapos_2, clapos_3 = cla.ne(0), cla.ne(1), cla.ne(2), cla.ne(3)
        # 假设只有0、1、2三个类别显著，则在将除0外的类别都掩码时，3、4类别在应被随机掩码，而1、2类别应被赋予大的值。
        if tr_te == 'train':
            # pad0,pad1,pad2,pad3 = random.randint(1,90),random.randint(1,90),random.randint(1,90),random.randint(1,90)
            pad0, pad1, pad2, pad3 = 180, 180, 180, 180
        elif tr_te == 'test':
            pad0, pad1, pad2, pad3 = 180, 180, 180, 180
        cla_0, cla_1, cla_2, cla_3 = time_gap.masked_fill(clapos_0, pad0), time_gap.masked_fill(clapos_1,
                                                                                                pad1), time_gap.masked_fill(
            clapos_2, pad2), time_gap.masked_fill(clapos_3, pad3)
        p_gap_0, p_gap_1, p_gap_2 = self.Distri(cla_0, 0).unsqueeze(1), self.Distri(cla_1, 1).unsqueeze(1), self.Distri(
            cla_2, 2).unsqueeze(1)
        p_gap_0, p_gap_1, p_gap_2 = torch.bmm(p_gap_0, input_onehot).squeeze(1), torch.bmm(p_gap_1,
                                                                                           input_onehot).squeeze(
            1), torch.bmm(p_gap_2, input_onehot).squeeze(1)
        # p_gap = self.Distri(time_gap).unsqueeze(1)#[B,S]
        # p_gap = torch.bmm(p_gap, input_onehot).squeeze(1)
        # pdb.set_trace()
        p_repeat_ = (1 - self.w_p_g) * p_repeat + self.w_p_g * (p_gap_0 + p_gap_1 + p_gap_2)  # [B,vocab]
        return p_repeat_  # output

    def Distri(self, x, mode):
        if mode == 0:  # 1G
            G1 = torch.distributions.Normal(loc=self.mu0, scale=self.sigma0)
            score = self.fs0 * G1.log_prob(x).exp()
        if mode == 1:  # 2G
            G1 = torch.distributions.Normal(loc=self.mu1_1, scale=self.sigma1_1)
            G2 = torch.distributions.Normal(loc=self.mu1_2, scale=self.sigma1_2)
            score = self.fs1_1 * G1.log_prob(x).exp() + self.fs1_2 * G2.log_prob(x).exp()
        if mode == 2:  # G+m
            G1 = torch.distributions.Normal(loc=self.mu2, scale=self.sigma2)
            P = torch.pow(x, self.p2)
            score = self.fs2_1 * G1.log_prob(x).exp() + self.fs2_2 * P
        if mode == 3:  # m
            P = torch.pow(x, self.p3)
            score = self.fs3 * P
        # score = self.fs1*G1.log_prob(x).exp()+self.fs2*G2.log_prob(x).exp()+self.fs3*P
        return score

    def get_par(self):
        # total
        par = [float(self.fs0), float(self.fs1_1), float(self.fs1_2), float(self.fs2_1), float(self.fs2_2),
               float(self.fs3),
               float(self.w_p_g), float(self.mu0), float(self.sigma0), float(self.mu1_1), float(self.sigma1_1),
               float(self.mu1_2),
               float(self.sigma1_2), float(self.mu2), float(self.sigma2), float(self.p2), float(self.p3)]
        return par


class Explore_decoder_add(nn.Module):  # 无需改
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg, device):
        super().__init__()
        self.vocab_size = cfg['vocab_size']
        self.proj_q = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_k = nn.Linear(cfg['dim'], cfg['dim'])
        self.proj_v = nn.Linear(cfg['dim'], 1)
        self.activation = nn.Tanh()
        self.W_ec = nn.Linear(2 * cfg['dim'], self.vocab_size)  # explore这个motion根据拼接后的向量，输出一个得分分布
        # self.drop = nn.Dropout(cfg['p_drop_attn'])
        self.proj = cfg['explore_proj']
        self.interest_id = cfg['interest_id']
        self.pad_id = cfg['pad_id']

    def forward(self, x, x_ids):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D),# (B, 1, D) -proj-> (B, 1, D)
        # q,k = self.proj_q(x),self.proj_k(x[:,0,:].unsqueeze(dim=1))#用各个token的query来和interest的key计算相似度
        # k=k.expand(q.shape[0],q.shape[1],q.shape[2])#[B, S, D]
        mask0 = x_ids.ne(self.pad_id)  # 加了mask，为了少一个softmax,其实也应该把interest给mask掉，不分softmax的权重
        mask1 = x_ids.ne(self.interest_id)
        mask = mask0 * mask1  # 或关系，[B,S]
        # mask = x_ids.ne(259)#全为true
        # mask_ = mask.unsqueeze(1)#不知需不需要专门扩
        input_onehot = build_map(x_ids, max=self.vocab_size)
        if self.proj:
            q, k = self.proj_q(x), self.proj_k(x[:, 0, :].unsqueeze(dim=1))
        else:
            q, k = x, x[:, 0, :].unsqueeze(dim=1)  # .expand(x.shape[0],x.shape[1],x.shape[2])
        k = k.expand(q.shape[0], q.shape[1], q.shape[2])
        features = q + k  # [B, S, D]
        features = self.activation(features)
        scores = self.proj_v(features)  # [B,S,1]
        scores = scores.squeeze(dim=-1)  # (B, S)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # (B, S, D) @ (B, D, 1) -> (B, S, 1)
        # scores = q @ k.transpose(-2,-1)/ np.sqrt(k.size(-1))

        '''softmax'''
        # if mask is not None:#new
        #    scores = scores.masked_fill(~mask, -float('inf'))#new
        scores = F.softmax(scores, dim=-1)  # (B, S),softmax后得到每个token和interest的匹配度
        scores = scores.unsqueeze(dim=-1)  # (B, S, 1)
        c_s = torch.sum(x * scores, dim=1)  # x*scores后(B, S, D)，对S维度求和后(B, D)
        h_t = x[:, 0, :].squeeze(dim=1)  # (B, D)
        c_s = torch.cat((h_t, c_s), -1)  # (B,D+D)
        p_explore = self.W_ec(c_s)  # (B, vocab)
        explore_mask = torch.bmm(mask.float().unsqueeze(1), input_onehot).squeeze(1)  # 代表非padding部分
        '''softmax'''
        p_explore = p_explore.masked_fill(explore_mask.bool(), float('-inf'))
        p_explore = F.softmax(p_explore, dim=-1)
        return p_explore