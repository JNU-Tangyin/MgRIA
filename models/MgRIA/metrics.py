import torch

def calculate_hits(y_pre,masked_ids,N):
    _, indices = torch.topk(y_pre, int(N), -1)
    indices = indices.view(indices.size()[0], -1)
    masked_ids = masked_ids.view(-1, 1).expand_as(indices)
    hits = (masked_ids == indices).nonzero()
    # hits : tensor[number of hits,2]
    hits_num = hits[:,0].shape[0]
    hits_index = hits[:,1].cpu().numpy()
    return hits_num,hits_index

def recall(accum_matrix,test_num,N):
    return accum_matrix.loc[N,'hits_num']/test_num

def mrr(accum_matrix,test_num,N):
    return accum_matrix.loc[N,'hits_mrr']/test_num

def ndcg(accum_matrix,test_num,N):
    return accum_matrix.loc[N,'hits_ndcg']/test_num

    
