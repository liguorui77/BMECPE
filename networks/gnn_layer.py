import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from config import DEVICE
import numpy as np
import math
import csv

class GraphAttentionLayer(nn.Module):
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn
        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)


    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        print(batch)
        assert in_dim == self.in_dim
        feat_in_ = feat_in.unsqueeze(1) #对输入的特征矩阵进行维度拓展，将其从(batch, N, in_dim)扩展为(batch, 1, N, in_dim)
        h = torch.matmul(feat_in_, self.W) #self.W维度(att_head, in_dim, out_dim)
        attn_src = torch.matmul(h, self.w_src)
        attn_dst = torch.matmul(h, self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)
        attn = torch.tanh(attn)
        if adj is not None:
            # 将adj转换为numpy数组，并将其转换为张量
            adj = np.array(adj)
            adj = torch.tensor(adj, dtype=torch.float32, device=DEVICE)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask.bool(), -999)
        attn = F.softmax(attn, dim=-1)
        feat_out = torch.matmul(attn, h) + self.b
        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)
        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in
        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)
        return feat_out

    # 将输入维度转换成输出维度*注意头数量
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'

   # 第一层图注意力层的输入维度 in_dim 为 self.feat_dim，即 768。
   # 第一层图注意力层的输出维度 out_dim 为 self.gnn_dims[0]，即 192。
   # 第一层图注意力层的注意头数量 att_head 为 self.att_heads[0]，即 4。
   # 最终 doc_sents_h 的形状将是 (batch_size, max_doc_len, out_dim * att_head)，即 (batch_size, max_doc_len, 192 * 4)，即 (batch_size, max_doc_len, 768)。这与输入特征的维度保持一致。

