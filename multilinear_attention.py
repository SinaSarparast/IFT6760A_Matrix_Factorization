import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import torch.nn as nn
import numpy as np

# https://github.com/szhangtju/The-compression-of-Transformer

__author__ = "Xindian Ma"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SingleCoreAttention(nn.Module):
    ''' Single Core Attention (Single Block Attention)'''
    def __init__(self, temperature, d_v, n_head, atten_dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=0)
        # why they have hard coded two core tensors?
        core_1 = self.softmax(torch.randn(d_v))
        core_2 = self.softmax(torch.randn(d_v))
        self.vectors = torch.stack((core_1,core_2),dim=0)

    def forward(self, q, k, v, mask=None):
        mb_size, len_q, dimen = q.size()

        cores_1 = torch.zeros(dimen,dimen,len_q).to(device)
        cores_2 = torch.zeros(dimen,dimen,len_q).to(device)
        for i in range(int(min(dimen,len_q))):
            cores_1[i][i][i] = self.vectors[0][i].to(device)
            cores_2[i][i][i] = self.vectors[1][i].to(device)
        full_matrix_1 = torch.einsum('pqk, bip,bjq,bkr->bijr', [cores_1, q, k, v]).contiguous().to(device)
        full_matrix_2 = torch.einsum('pqk, bip,bjq,bkr->bijr', [cores_2, q, k, v]).contiguous().to(device)
        average_tensor = (torch.sum(full_matrix_1, dim=2)+torch.sum(full_matrix_2, dim=2)).mul_(0.5)
        average_tensor = average_tensor/self.temperature
        # output = torch.stack(average_tensor).to(device).float()
        output = self.dropout(average_tensor)
        attn = torch.bmm(q, k.transpose(1, 2)).to(device)
        del(cores_1)
        del(cores_2)
        return output, attn

class MultiLinearAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        # self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention = SingleCoreAttention(np.power(d_k,0.5), self.d_v, n_head, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, attn_mask=None, **kwargs):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        attn_mask = attn_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn= self.attention(q, k, v, mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn