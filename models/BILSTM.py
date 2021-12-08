import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import AverageMeter

class BiLstm(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,out_size,dropout=0.1):
        super(BiLstm,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(embedding_size,hidden_size,batch_first=True,bidirectional=True)
        #self.fc = nn.Linear(2*hidden_size,out_size)
        self.fc = nn.Linear(embedding_size,out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,inputs_ids,input_lens):
        # [b,l,emb_size ]
        emb = self.dropout(self.embedding(inputs_ids))
        # 这里要求输入按长度递减排好序，否则enforce_sorted设置为false,低版本方法有不同之处
        #emb = nn.utils.rnn.pack_padded_sequence(emb, input_lens, batch_first=True)
        #emb,_ = self.bilstm(emb)
        #emb,_ = nn.utils.rnn.pad_packed_sequence(emb,batch_first=True,padding_value=0.,total_length=inputs_ids.shape[1])
        scores = self.fc(emb)
        return scores

    def predict(self,input_ids, input_mask, input_tags, input_lens):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(input_ids, input_lens)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids

    def forward_loss(self, inputs_ids, mask, input_lens, targets):
        """计算损失
        参数:
            logits: [B, L, out_size]
            targets: [B, L]
            lengths: [B]
        """
        logits = self.forward(inputs_ids, input_lens)
        mask = (mask != 0)  # [B, L]
        # print(type(mask))
        targets = targets[mask]   # 拉平成了一个维度 B * L (去除了mask中为false的，实际长度要减去mask中为false值)
        # print(targets.shape)
        out_size = logits.size(2)
        # mask.unsqueeze(2) 【 B, L, 1 】
        # expand把第三维度复制成outsize
        logits = logits.masked_select(
            mask.unsqueeze(2).expand(-1, -1, out_size)
        ).contiguous().view(-1, out_size)
        # 最后输出维度为【B*L,outsize】
        # 第一维度其实减去了mask掉的

        assert logits.size(0) == targets.size(0)
        loss = F.cross_entropy(logits, targets)

        return loss