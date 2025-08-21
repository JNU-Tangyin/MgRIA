'''
得到用户购买商品数据dataset模块,不带嵌入表示的，只给input_idx
'''
#得到G_u序列
import pdb
import random
import numpy as np
from torch.utils.data import Dataset as tDataset

class ItemData(tDataset):
    def __init__(self, user_gp, user_gp_stamp, user_gp_stamp_t, user_gp_cla, vocabulary, cfg, tr_te):
        self.user_gp = list(user_gp)
        self.user_gp_stamp = list(user_gp_stamp)
        self.user_gp_stamp_t = list(user_gp_stamp_t)
        self.user_gp_cla = list(user_gp_cla) if user_gp_cla else None
        self.vocab = list(vocabulary)  # 词表
        
        self.pad_id = cfg['pad_id']
        self.mask_id = cfg['mask_id']
        self.interest_id = cfg['interest_id']
        if self.vocab:
            self.dic_vocab_idx = self.str2idx(self.vocab)
            self.dic_vocab_idx["<PAD>"] = self.pad_id#257
            self.dic_vocab_idx['[MASK]'] = self.mask_id#256
            self.dic_vocab_idx['[Interest]'] = self.interest_id#258
        self.max_pred = cfg['max_pred']  # max tokens of prediction，训练时鉴于样本短，所以只预测一位
        self.max_len = cfg['max_len']
        self.tr_te = tr_te
        # 为得到self.input_mask,self.masked_ids, self.masked_pos, self.masked_weights
        self.input_mask, self.masked_ids, self.masked_pos, self.masked_weights = [], [], [], []
        # 为得到self.x,即用emb表示的各个样本，[batch,seq,emb]
        self.x, self.stamp = [], []  # 要的是换成emb的user_gp
        self.stamp_t = []
        self.stamp_gap = []
        self.dataset = cfg['dataset']
        self.cla = []
        # for user in self.user_gp:#超长序列处理可变，目前是截断
        for i in range(len(user_gp)):
            user = self.user_gp[i]
            stamp = self.user_gp_stamp[i]
            stamp_t = self.user_gp_stamp_t[i]
            if self.user_gp_cla:
                cla = self.user_gp_cla[i]
                input_ids_insert, input_stamp_insert, input_mask_insert, masked_ids_insert, masked_pos_insert, masked_weight_insert, stamp_t_insert, stamp_gap_insert, cla_insert = self.prepare( \
                    list(user[-self.max_len:]), list(stamp[-self.max_len:]), list(stamp_t[-self.max_len:]),list(cla[-self.max_len:]), cfg)  # 加个list防止原user被修改
                self.cla.append(cla_insert)
            else:
                input_ids_insert, input_stamp_insert, input_mask_insert, masked_ids_insert, masked_pos_insert, masked_weight_insert, stamp_t_insert, stamp_gap_insert, cla_insert = self.prepare( \
                    list(user[-self.max_len:]), list(stamp[-self.max_len:]), list(stamp_t[-self.max_len:]),[], cfg)  # 加个list防止原user被修改
                self.cla.append(cla_insert)
            # if i in [1,2,3]:
            #    print(len(masked_ids_insert),len(masked_pos_insert),len(masked_weight_insert))
            # else:
            #    input_insert,input_mask_insert,masked_ids_insert,masked_pos_insert,masked_weight_insert = self.prepare(user[:self.max_len])
            self.x.append(input_ids_insert)
            self.stamp.append(input_stamp_insert)
            self.input_mask.append(input_mask_insert)
            self.masked_ids.append(masked_ids_insert)
            self.masked_pos.append(masked_pos_insert)
            self.masked_weights.append(masked_weight_insert)
            self.stamp_t.append(stamp_t_insert)
            self.stamp_gap.append(stamp_gap_insert)
            
        # print(self.gap[0])
        # for i in range(11):
        #    print(i,len(self.gap[0][i]))
        self.x, self.stamp = np.array(self.x), np.array(self.stamp)  # 得到一个个emb好的样本
        self.stamp_t = np.array(self.stamp_t)
        self.stamp_gap = np.array(self.stamp_gap)
        self.input_mask, self.masked_ids, self.masked_pos, self.masked_weights = np.array(self.input_mask), np.array(
            self.masked_ids), np.array(self.masked_pos), np.array(self.masked_weights)
        self.cla = np.array(self.cla)
        # print(self.x[0])
        # print(self.stamp[0])#[B, S, 3]
        # print(self.stamp_t[0])#[B, S+1, S+1, D]

    def prepare(self, tokens, stamp, stamp_t, cla, cfg):  # tokens是一个购买序列
        if self.dataset == 'equity':
            # masked_ids = self.convert_tokens_to_ids(self.dic_vocab_idx, masked_tokens)
            input_ids = self.convert_tokens_to_ids(self.dic_vocab_idx, tokens)
        else:
            # masked_ids = masked_tokens
            input_ids = tokens
        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = self.max_pred  # 假设只预测一个先
        # candidate positions of masked tokens
        '''
        要改mask位置改这个cand_pos
        '''
        if self.tr_te == 'train':
            cand_pos = np.random.randint(len(input_ids), size=n_pred)  # cand_pos = [i for i, token in enumerate(tokens)]
            for pos in cand_pos:
                masked_tokens.append(input_ids[pos])
                masked_pos.append(pos + 1)  # 第一个位置留给interest
                if cfg['multi_mask']:
                    if random.random() < 0.8:  # 80%
                        input_ids[pos] = self.mask_id#'[MASK]'
                    elif random.random() > 0.9:  # 10%
                        input_ids[pos] = random.randint(0, cfg['vocab_size'] - 4)#self.get_random_word(self.vocab)
                else:
                    input_ids[pos] = self.mask_id#'[MASK]'
        elif self.tr_te == 'test':
            cand_pos = [len(input_ids) - 1]
            for pos in cand_pos:
                masked_tokens.append(input_ids[pos])
                masked_pos.append(pos + 1)  # 第一个位置留给interest
                input_ids[pos] = self.mask_id#'[MASK]'

        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)
        # 生成time_gap,还未扩充，前面不用帮interest补充，因为onehot计算时没考虑它
        time_gap = self.computegap(stamp_t, cand_pos[0])
        # 先给token的最前面插入代表序列兴趣的token
        input_ids.insert(0, self.interest_id)#'[Interest]')  # 这里在开头多加了一位，而stamp则希望是编码为0，所以在embedding中以全零加入
        stamp_t.insert(0, stamp_t[0])  # 这里在开头多加了一位，姑且用第一个时间戳来作为interest的时间戳
        input_mask = [1] * len(input_ids)

        # Zero Padding
        n_pad = self.max_len + 1 - len(input_ids)  # 不能考虑interest的位置
        input_mask.extend([self.pad_id] * n_pad)
        input_ids.extend([self.pad_id] * n_pad)
        # stamp.extend([[12,31,7,24]]*n_pad)
        stamp.extend([[12, 31, 7]] * n_pad)
        stamp_t.extend([stamp_t[0]] * n_pad)  # 按原文，用第一个时间戳来补全pad的时间戳
        time_gap.extend([90] * n_pad)  # 即使不小心考虑，也会因为太远而获得很小的概率
        
        time_matrix = self.computeRePos(stamp_t, cfg['time_span'])
        # stamp.insert(0,[12,31,7])

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([self.pad_id] * n_pad)
            # masked_ids.extend([self.pad_id] * n_pad)
            masked_pos.extend([self.pad_id] * n_pad)
            masked_weights.extend([self.pad_id] * n_pad)
        if cla:
            cla.extend([4] * n_pad)
        else:
            cla.extend([4] * self.max_len)
        return input_ids, stamp, input_mask, masked_tokens, masked_pos, masked_weights, time_matrix, time_gap, cla  #
            
            

    def computegap(self, time_seq, pos):
        time_vec = np.zeros([len(time_seq)], dtype=np.int32)
        for i in range(len(time_seq)):
            val = int(round((abs(time_seq[pos] - time_seq[i]) / 86400)))
            if val == 0:
                # time_vec[i] = 90#0不算复购，会让幂律变为inf，所以换成一个影响小的
                if self.tr_te == 'train':
                    time_vec[i] = random.randint(1, 90)
                elif self.tr_te == 'test':
                    time_vec[i] = 1
            else:
                time_vec[i] = val
        return list(time_vec)

    def computeRePos(self, time_seq, time_span):  # 一个user序列构建一个基础的时间间隔矩阵
        # size = time_seq.shape[0]
        size = len(time_seq)
        time_matrix = np.zeros([size, size], dtype=np.int32)

        # 用每个序列的最小时间间隔作为time_scale
        time_diff = set()
        for i in range(len(time_seq) - 1):
            if time_seq[i + 1] - time_seq[i] < 0:
                break
            # elif (time_seq[i+1]-time_seq[i])/86400 != 0:#
            elif (time_seq[i + 1] - time_seq[i]) / 86400 > 1:  # 最小也得是1
                time_diff.add((time_seq[i + 1] - time_seq[i]) / 86400)
        ## 不同逻辑的语句块，起码用注释隔开
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        # time_scale = 1 if len(time_diff) ==0 else min(time_diff)
        # 根据r_u_ij的计算公式求
        # 更优雅的做法是：1 构造一个S*S的矩阵，元素为span 2 针对整个time_matrix做运算而非每个元素单独计算
        for i in range(size):
            for j in range(size):
                span = int(round((abs(time_seq[i] - time_seq[j]) / 86400) / time_scale))  # 将秒变成天，此处可以先不向下取整，而是用round取整
                if span > time_span:
                    time_matrix[i][j] = time_span
                else:
                    time_matrix[i][j] = span
                # time_matrix[i][j] = min(span, time_span)    # 拿本行替换以上if-else
        return time_matrix  # [S, S]，[S]array

    # def computeScale(time_matrix,time_min):
    # list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))，这是原代码用法
    # return time_matrix#[S, S]

    def get_random_word(self, vocab_words):
        i = random.randint(0, len(vocab_words) - 1)
        return vocab_words[i]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):  # pytorch的dataset生成batch时默认有限类型
        return self.x[index], self.stamp[index], self.input_mask[index], self.masked_ids[index], self.masked_pos[index], \
            self.masked_weights[index], self.stamp_t[index], self.stamp_gap[index], self.cla[index]


    def str2idx(self, vocab):  # 识别emb序列并输出商品序列，例如输出['item1','item3']
        dic_v_idx, count_idx = {}, 0
        for i in vocab:
            dic_v_idx[i] = count_idx
            count_idx += 1
        # dic_v_idx = {item:e for e,item in enumerate(vocab)}   # 拿本行可替代以上代码
        return dic_v_idx

    def convert_tokens_to_ids(self, vocab, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        # print("x遇到的新商品：")
        for token in tokens:
            if token in vocab:
                ids.append(vocab[token])
            else:
                print(token, end=",")
                vocab[token] = len(vocab) - 2
                ids.append(vocab[token])
        # ids = [vocab[token] for token in tokens if token in vocab else len(vocab) - 2]    # 本行可以替代上面所有
        return ids