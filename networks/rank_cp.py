from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE
from networks.gnn_layer import GraphAttentionLayer

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert_encoder = BertEncoder(configs)
        self.gnn = GraphNN(configs)
        self.pred_e = Pre_Predictions(configs)



    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len, adj, q_type , label ,flag):
        # shape: batch_size, max_doc_len, 1024
        # 其中doc_sents_h是文档中每个句子的BERT编码结果，query_h是查询的BERT编码结果
        doc_sents_h, query_h = self.bert_encoder(query, query_mask, query_seg, query_len, seq_len, doc_len)
        # 模型将doc_sents_h输入到self.gnn中的图神经网络，得到增强后的文档表示doc_sents_h,adj表示邻接矩阵
        doc_sents_h = self.gnn(doc_sents_h, doc_len, adj)
        # 模型将查询的BERT编码结果query_h与增强后的文档表示doc_sents_h进行拼接，得到最终用于预测的特征表示
        doc_sents_h = torch.cat((query_h, doc_sents_h), dim=-1)
        # 拼接后的结果将是 (batch_size, max_doc_len, 768 + 768)，即 (batch_size, max_doc_len, 1536)
        pred = self.pred_e(doc_sents_h)
        return pred # 返回处理后的概率值，不确定性和狄利克雷分布参数

    # 在损失函数中，模型首先将真实标签和预测结果进行掩码操作，然后使用BCELoss（二分类交叉熵损失）计算损失。
    def loss_pre(self, pred, true, mask):
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = torch.FloatTensor(pred.float()).to(DEVICE)
        true = torch.FloatTensor(true.float()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)


# 这个Pre_Predictions模型类实现了一个简单的预测模型，用于将文档的表示映射到句子级别的预测概率值
class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = 1536
        self.out_e = nn.Linear(self.feat_dim, 1)
        #out_e维度是(1,1536)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_e = torch.sigmoid(pred_e)
        return pred_e # shape: bs ,max_doc_len

class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(768, 1)
        self.fc_query = nn.Linear(768, 1)

    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len):
        hidden_states = self.bert(input_ids=query.to(DEVICE),
                                  attention_mask=query_mask.to(DEVICE),
                                  token_type_ids=query_seg.to(DEVICE))[0]
        hidden_states, mask_doc, query_state, mask_query = self.get_sentence_state(hidden_states, query_len, seq_len, doc_len)
        alpha = self.fc(hidden_states).squeeze(-1)  # bs, max_doc_len, max_seq_len
        mask_doc = 1 - mask_doc # bs, max_doc_len, max_seq_len
        alpha = alpha.to(torch.float32)
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # bs, max_doc_len, 768
        # print(hidden_states.shape)

        alpha_q = self.fc_query(query_state).squeeze(-1)  # bs, query_len
        mask_query = 1 - mask_query  # bs, max_query_len
        alpha_q = alpha_q.to(torch.float32)
        alpha_q.data.masked_fill_(mask_query.bool(), -9e5)
        alpha_q = F.softmax(alpha_q, dim=-1).unsqueeze(-1).repeat(1, 1, query_state.size(-1))
        query_state = torch.sum(alpha_q * query_state, dim=1)  # bs, 768
        query_state = query_state.unsqueeze(1).repeat(1, hidden_states.size(1), 1) # bs, max_doc_len, 768

        # doc_sents_h = torch.cat((query_state, hidden_states), dim=-1)
        # hidden_states: (batch_size, max_doc_len, max_seq_len, 768)
        # query_state: (batch_size, max_doc_len, 768)
        return hidden_states.to(DEVICE), query_state.to(DEVICE)


    def get_sentence_state(self, hidden_states, query_lens, seq_lens, doc_len):
        # 对问题的每个token做注意力，获得问题句子的向量表示；对文档的每个句子的token做注意力，得到每个句子的向量表示
        sentence_state_all = []
        query_state_all = []
        mask_all = []
        mask_query = []
        max_seq_len = 0
        for seq_len in seq_lens: # 找出最长的一句话包含多少token
            for l in seq_len:
                max_seq_len = max(max_seq_len, l)
        max_doc_len = max(doc_len) # 最长的文档包含多少句子
        max_query_len = max(query_lens)  # 最长的问句包含多少token
        for i in range(hidden_states.size(0)):  # 对每个batch
            # 对query
            query = hidden_states[i, 1: query_lens[i] + 1]
            assert query.size(0) == query_lens[i]
            if query_lens[i] < max_query_len:
                query = torch.cat([query, torch.zeros((max_query_len - query_lens[i], query.size(1))).to(DEVICE)], dim=0)
            query_state_all.append(query.unsqueeze(0))
            mask_query.append([1] * query_lens[i] + [0] * (max_query_len -query_lens[i]))
            # 对文档sentence
            mask = []
            begin = query_lens[i] + 2  # 2是[cls], [sep]
            sentence_state = []
            for seq_len in seq_lens[i]:
                sentence = hidden_states[i, begin: begin + seq_len]
                begin += seq_len
                if sentence.size(0) < max_seq_len:
                    sentence = torch.cat([sentence, torch.zeros((max_seq_len - seq_len, sentence.size(-1))).to(DEVICE)],
                                         dim=0)
                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * seq_len + [0] * (max_seq_len - seq_len))
            sentence_state = torch.cat(sentence_state, dim=0).to(DEVICE)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_seq_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(DEVICE)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        query_state_all = torch.cat(query_state_all, dim=0).to(DEVICE)
        mask_query = torch.tensor(mask_query).to(DEVICE)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(DEVICE)
        mask_all = torch.tensor(mask_all).to(DEVICE)
        return sentence_state_all, mask_all, query_state_all, mask_query



# 这个GraphNN类实现了一个基于图神经网络的模型，用于处理文档数据，并通过注意力机制来学习句子之间的关系。
class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]  # [1024, 192]

        self.gnn_layers = len(self.gnn_dims) - 1  #self.gnn_layers = 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')] # [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers): #只有一个注意力层
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp) #输入的是(4, 768, 192, dp)
            )

    # 它将图注意力层逐层应用于输入节点特征 (doc_sents_h)，并在所有层处理后返回最终的节点表示。
    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size() #这行代码提取了 doc_sents_h 张量的维度。其中 batch 表示批处理大小，max_doc_len 是文档中句子的最大数量，第三个维度（下划线 _）表示每个句子的特征维度。
        assert max(doc_len) == max_doc_len
        # 在每次循环中，当前的图注意力层 (gnn_layer) 与邻接矩阵 (adj) 一起应用于输入节点特征 (doc_sents_h)。gnn_layer 执行图注意力操作并更新节点表示。
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h


