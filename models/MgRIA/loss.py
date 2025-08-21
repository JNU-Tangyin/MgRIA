import torch
import torch.nn as nn


def get_loss(model, batch, device, cfg):  # make sure loss is tensor
    x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch  # masked_pos其实好像没啥用
    x = x.to(torch.int64).to(device)
    stamp = stamp.to(torch.int64).to(device)
    # gap = gap.to(torch.int64)
    masked_ids = masked_ids.to(torch.int64).to(device)
    masked_pos = masked_pos.to(torch.int64).to(device)
    time_matrix = time_matrix.to(torch.int64).to(device)
    time_gap = time_gap.to(torch.int64).to(device)
    cla = cla.to(torch.int64).to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    logits_lm = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla,
                      'train')  # [batch,pre_num,vocab_num]
    loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
    loss_lm = (loss_lm * masked_weights.float()).mean()

    return logits_lm, loss_lm