import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from numpy import mean
import pandas as pd
import numpy as np
# customerized modules
from output import BertRepeatModel
from gu import ItemData
from metrics import * 
from sklearn.model_selection import train_test_split

data_dir = r"code/data/"    # 数据文件夹

#-------------------------测试用
equity = ['recall@3','recall@10','mrr@10','ndcg@10']
#-------------------------测试用

# 数据集的处理
# 训练数据集的处理，得到S_u,T_u,C_u序列
def convert_unique_idx(df, column_name,cla_dic=False):
    if cla_dic:
        column_dict = cla_dic
    else:
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    #assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

# 非常恶劣的代码风格，将绝对路径嵌在函数里面

def Get_user_gp(df,cfg):
    df = df.rename(columns={cfg['session_id']: 'session_id', cfg['item_id']: 'item_id', cfg['time_id']: 'time'})
    df_sort = df.sort_values(by ='time')
    if cfg['dataset'] == 'equity':
        df_sort,cla_mapping = convert_unique_idx(df_sort,'classify',cfg['cla_dic'])  
        print(cla_mapping)
        df_sort['time']= df_sort.time.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp())
    df_sort['createtime'] = pd.to_datetime(df_sort.time)#time stamp
    df_sort['month'] = df_sort.createtime.dt.month #apply(lambda row:row.month,1)
    df_sort['day'] = df_sort.createtime.dt.day #apply(lambda row:row.day,1)
    df_sort['weekday'] = df_sort.createtime.dt.weekday #.apply(lambda row:row.weekday(),1)
    df_sort['hour'] = df_sort.createtime.dt.hour #.apply(lambda row:row.hour,1)
    user_gp_item = df_sort.groupby(['session_id'])['item_id'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_t = df_sort.groupby(['session_id'])['time'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_month = df_sort.groupby(['session_id'])['month'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_day = df_sort.groupby(['session_id'])['day'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_weekday = df_sort.groupby(['session_id'])['weekday'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_hour = df_sort.groupby(['session_id'])['hour'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    if cfg['dataset'] == 'equity':
        user_gp_cla = df_sort.groupby(['session_id'])['classify'].agg({list}).reset_index()['list'].values.tolist()
    else:
        user_gp_cla = None
    return user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla#,user_gp_hour

def Get_train_test_dataset(train_size=0.2,cfg=None):#(file_dir = r"../../datasets/equity.csv",train_size = 0.2,cfg=None),cla_dic=False):
    df = pd.read_csv(cfg['data_path'],encoding='gbk')  
    df_train,df_test = train_test_split(df,test_size=train_size,random_state=123)# 划分训练集和测试集
    return Get_user_gp(df_train,cfg),Get_user_gp(df_test,cfg)

def get_loss(model, batch, device, cfg):  # make sure loss is tensor
    x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch  # masked_pos其实好像没啥用
    x = x.to(torch.int64).to(device)
    stamp = stamp.to(torch.int64).to(device)
    # gap = gap.to(torch.int64)
    masked_ids = masked_ids.to(torch.int64).to(device)
    masked_pos = masked_pos.to(torch.int64).to(device)
    masked_weights = masked_weights.to(torch.float64).to(device)
    time_matrix = time_matrix.to(torch.int64).to(device)
    time_gap = time_gap.to(torch.int64).to(device)
    cla = cla.to(torch.int64).to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    logits_lm = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, 'train')  # [batch,pre_num,vocab_num] forwawrd
    loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
    loss_lm = (loss_lm * masked_weights).mean()

    return logits_lm, loss_lm

def train(user_gp, user_gp_test, user_gp_stamp, user_gp_stamp_test,user_gp_stamp_t, user_gp_stamp_t_test, user_gp_cla,user_gp_cla_test, vocabulary, cfg):
    dataset = ItemData(user_gp, user_gp_stamp, user_gp_stamp_t, user_gp_cla, vocabulary, cfg, 'train')
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    dataset_test = ItemData(user_gp_test, user_gp_stamp_test, user_gp_stamp_t_test, user_gp_cla_test, vocabulary, cfg,'test')
    loader_test = DataLoader(dataset_test, batch_size=cfg['batch_size'] * 10, shuffle=True)
    torch.manual_seed(cfg['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertRepeatModel(cfg,device)
    if cfg['pre_train']:
        model.transformer = torch.load(cfg['pre_train']).transformer
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    
    
    #if token embed block needs L2 norm
    # model_else = []
    # for name, p in model.named_parameters():
    #     if name != 'transformer.embed.tok_embed.weight':
    #         model_else += [p] 
    # optim = torch.optim.Adam([{'params':model.transformer.embed.tok_embed.parameters(),'weight_decay':cfg['weight_decay']},{'params':model_else}],lr=cfg['lr'])
    
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])  # 也可以换其它的优化器
    loss_total = []
    loss_per_eps = []
    metric_total = {i:[] for i in equity}
    t1 = time.time()
    for i in range(cfg['iter_n']):
        model.train()
        count_iter = 0
        # t1 = time.time()
        for batch_idx, batch in enumerate(loader):
            optim.zero_grad()
            logits, loss = get_loss(model, batch, device, cfg)
            loss.backward()
            optim.step()
            loss_total.append(loss.detach().cpu().numpy())
            count_iter += 1
            if batch_idx % 50 == 0:
                print(
                    "Epoch: ", i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                )
                # t2 = time.time()-t1
                # t1 = time.time()
                # print('用时：',t2)
        loss_per_eps.append(sum(loss_total[-count_iter:]) / count_iter)
        print("| loss_per_eps: %.3f" % loss_per_eps[-1])
        if Config['test_while_train']:
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.cuda() if torch.cuda.is_available() else model.cpu()
            # if torch.cuda.is_available():
            #     print("GPU test avaliable")
            #     device = torch.device("cuda")
            #     model = model.cuda()
            # else:
            #     device = torch.device("cpu")
            #     model = model.cpu()
            average = np.zeros(len(equity))
            # correct_range可以改成其他参数了
            correct_range = cfg['correct_range']
            # new
            for ctt in range(cfg['test_times']):
                ans = test(loader_test, model, correct_range, device)
                average += ans
                # ans_test = test(loader_test, model, correct_range, device)
            print('----------------- Average Accuracy:', average / cfg['test_times'], '---------------------')
            
            # 每个batch的指标序列
            for idx,m in enumerate(equity):
                metric_total[m].append((average / cfg['test_times'])[idx])
            # if cfg['save']:
            #     torch.save(model, cfg['save_dir'] + 'repeat_res_int_dis' + str(i) + str(ans_test)[2:] + '.model')
            #     print('Model saved to', cfg['save_dir'] + 'repeat_res_int_dis' + str(i) + str(ans_test)[2:] + '.model')

    # print('use time:', time.time() - t1)

    metric_avg = {m:np.mean(lst) for m,lst in metric_total.items()}
    metric_total = pd.DataFrame(metric_total)
    metric_avg = pd.DataFrame(metric_avg,index = [0])
    # save metric
    metric_total.to_csv(r'results/metric_total.csv')
    metric_avg.to_csv(r'results/metric_avg.csv')
    print('metric_toal:',metric_total)
    print('metric_avg:',metric_avg)

    # 画loss图
    plt.plot(loss_total)
    plt.show()
    plt.plot(loss_per_eps)
    plt.show()
    # print('Mean Recall after 15 epochs:', mean(recall_total[15:]))
    # print(recall_total)

def test(loader_test, model, correct_range, device):
    res = []
    accum_matrix = pd.DataFrame(columns = ['hits_num','hits_mrr','hits_ndcg'],index = list(set([i.split('@')[1] for i in equity])))
    accum_matrix.loc[:] = 0
    test_num = 0  # new 

    for batch_idx, batch in enumerate(loader_test):
        # print('start batch', batch_idx, '----------------------')
        x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch
        x = x.to(torch.int64).to(device)
        stamp = stamp.to(torch.int64).to(device)
        time_matrix = time_matrix.to(torch.int64).to(device)
        time_gap = time_gap.to(torch.int64).to(device)
        cla = cla.to(torch.int64).to(device)
        masked_ids = masked_ids.to(torch.int64).to(device)
        masked_pos = masked_pos.to(torch.int64).to(device)
        input_mask = input_mask.to(torch.int64).to(device)
        masked_weights = masked_weights.to(torch.float32).to(device)
        # y_pre = model(x, stamp, gap, input_mask, masked_pos).tolist()  #

        y_pre = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, 'test')

        test_num += masked_ids.size(0)

        for i in equity:
            _, N = i.split('@')
            hits_num,hits_index = calculate_hits(y_pre,masked_ids,N)
            accum_matrix.loc[N,'hits_num'] += hits_num
            accum_matrix.loc[N,'hits_mrr'] += sum(1/(hits_index+1))
            accum_matrix.loc[N,'hits_ndcg'] += sum(1/np.log2(1+hits_index+1))

    for i in equity:
        metric,N = i.split('@')
        res.append(eval(metric)(accum_matrix,test_num,N))
    print(res)
    return np.array(res)

Config = {
    'n_layers': 1,  # Number of layers
    'n_heads': 5,  # Number of attention heads
    'p_drop_hidden': 0.1,  # Dropout probability for hidden layers
    'p_drop_attn': 0.1,  # Dropout probability for attention layers
    'iter_n': 2,  # Number of iterations
    'half_epoch': 1,  # Half of the epoch size
    'save': False,  # Flag indicating whether to save the model
    'max_pred': 1,  # Maximum number of predictions
    'lr': 0.002,  # Learning rate
    'weight_decay': 1e-4,  # Weight decay
    'seed': 2300,  # Random seed
    'test_while_train': True,  # Flag indicating whether to test while training
    'test_times': 1,  # Number of times to run the test
    # correct_range 对应计算指标的N
    'correct_range': 3,  # Range for correctness
    'model': 'Repeat',  # Model type
    'pro_add': 'add',  # Pro add type
    'time_span': 65,  # Time span
    'pre_train': False,  # Pre-training flag
    'save_dir': 'code/model/',  # Directory to save the model
    'repeat_proj': False,  # Repeat projection flag
    'explore_proj': False,  # Explore projection flag
    'res': True,  # Res flag
    'multi_mask': True  # Multi-mask flag
}

config_equity = {
    'dim': 35,  # Dimension
    'dim_ff': 35 * 4,  # Dimension of the feed-forward layer
    'session_id':'user', # user column name in dataset
    'item_id':'item', # item column name in dataset
    'time_id':'createtime', # time column name in dataset
    'vocab_size': 259,  # Size of the vocabulary
    'max_len': 11,#11,  # Maximum length of input sequences
    'batch_size': 128,  # Size of each training batch
    'mask_id': 256,  # ID for masking tokens
    'pad_id': 257,  # ID for padding tokens
    'interest_id': 258,  # ID for interest tokens
    'dataset':'equity',
    'cla_dic': {'视频': 1, '游戏': 3, '出行': 3, '电商': 0, '音频': 1, '阅读': 2, '美食': 2, '酒店': 4, '医疗': 4, '生活服务': 3, '通信': 4, '工具': 0, '教育': 4, '办公': 2, '快递': 4},  # Classification dictionary
    'data_path':r'datasets/equity.csv',
    'vocab_path':r'datasets/quanyi_train_vocab.npy'    
}
config_tafeng = {
    'dim': 35,  # Dimension
    'dim_ff': 35 * 4,  # Dimension of the feed-forward layer
    'session_id':'session_id',
    'item_id':'item_id',
    'time_id':'time',
    'vocab_size': 15786+3,#259,  # Size of the vocabulary
    'max_len': 40,#11,  # Maximum length of input sequences
    'batch_size': 256,#128,  # Size of each training batch
    'mask_id': 15787,#256,  # ID for masking tokens
    'pad_id': 15786,#257,  # ID for padding tokens
    'interest_id': 15788,#258,  # ID for interest tokens
    'dataset':'tafeng',
    'data_path':r'datasets/tafeng.csv'
}
config_taobao = {
    'dim': 50,  # Dimension
    'dim_ff': 50 * 4,  # Dimension of the feed-forward layer
    'session_id':'SessionID',
    'item_id':'ItemID',
    'time_id':'Time',
    'vocab_size': 287005+3,  # Size of the vocabulary
    'max_len': 19,#11,  # Maximum length of input sequences
    'batch_size': 256,#128,  # Size of each training batch
    'mask_id': 287006, # ID for masking tokens
    'pad_id': 287005, # ID for padding tokens
    'interest_id': 287007,# ID for interest tokens
    'dataset':'taobao',
    'data_path':r'datasets/taobao.csv'
}

Config.update(config_tafeng) # choose dataset's config to concate

if __name__ == "__main__":
    # 获取训练集
    train_gp,test_gp = Get_train_test_dataset(train_size = 0.2,cfg=Config)#Get_train_test_dataset(cla_dic=Config['cla_dic'])
    user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla = train_gp
    
    user_gp_stamp = []#把时间戳信息融合在一起
    user_gp_stamp_t = []
    for i in range(len(user_gp_item)):
        insert = []
        insert_t = []
        for j in range(len(user_gp_item[i])):
            #insert.append([user_gp_month[i][j],user_gp_day[i][j],user_gp_weekday[i][j],user_gp_hour[i][j]])
            insert.append([user_gp_month[i][j],user_gp_day[i][j],user_gp_weekday[i][j]])#user_gp_t是s，s/60/60/24向下取整为日期差
            insert_t.append(user_gp_t[i][j])
            #insert.append(user_gp_month[i][j])
        user_gp_stamp.append(insert)
        user_gp_stamp_t.append(insert_t)
    print('Timestamps loaded')
    
    if 'vocab_path' in Config:
        vocabulary=np.load(Config['vocab_path'])
        vocabulary=vocabulary.tolist()
    else:
        vocabulary=[]

    print(user_gp_item[:5])#,dic_item_emb.keys())
    print(user_gp_stamp[:5])
    print(user_gp_stamp_t[:5])
    print("vocab loaded.")

    # test部分
    # 获得训练集的用户购买商品序列
    user_gp_test,user_gp_t_test,user_gp_month_test,user_gp_day_test,user_gp_weekday_test,user_gp_cla_test = test_gp
    user_gp_stamp_test = []  # 把时间戳信息融合在一起
    user_gp_stamp_t_test = []
    print('#Sequences in test-set:', len(user_gp_test))

    # Narm,Stamp diginetica yoochoose(click) movielen rights FM book?

    for i in range(len(user_gp_test)):
        insert_test = []
        insert_t_test = []
        for j in range(len(user_gp_test[i])):
            # insert_test.append([user_gp_month_test[i][j],user_gp_day_test[i][j],user_gp_weekday_test[i][j],user_gp_hour_test[i][j]])
            insert_test.append([user_gp_month_test[i][j], user_gp_day_test[i][j], user_gp_weekday_test[i][j]])
            insert_t_test.append(user_gp_t_test[i][j])
            # insert_test.append(user_gp_month_test[i][j])
        user_gp_stamp_test.append(insert_test)
        user_gp_stamp_t_test.append(insert_t_test)

    # print('we have %s rebuy test' % len(user_gp_stamp_test_re))
    # print('we have %s other test' % len(user_gp_stamp_test_no))

    train(user_gp_item,
          user_gp_test,
          user_gp_stamp,
          user_gp_stamp_test,
          user_gp_stamp_t,
          user_gp_stamp_t_test,
          user_gp_cla,
          user_gp_cla_test,
          vocabulary,
          Config
        )