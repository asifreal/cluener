import torch
from torch.nn import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
from .torch_crf import CRF

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class BiLstmCRFModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1):
        super(BiLstmCRFModel, self).__init__()
        self.id2label = {i: label for i, label in enumerate(label2id)}
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                              batch_first=True,num_layers=2,dropout=drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2,len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output= self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags, input_group=None):
        features = self.forward(input_ids, input_mask)
        return self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
      

    def predict(self, input_ids, input_mask, input_tags, input_lens):
        features = self.forward(input_ids, input_mask)
        tags, _ = self.crf._obtain_labels(features, self.id2label, input_lens)
        return tags


class BiLstmCRFAttnModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1, pretrain=None):
        super(BiLstmCRFAttnModel, self).__init__()
        self.id2label = {i: label for i, label in enumerate(label2id)}
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if pretrain is not None:
            self.embedding.from_pretrained(pretrain)
        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                              batch_first=True,num_layers=2,dropout=drop_p,
                              bidirectional=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=8, batch_first=True)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2,len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        packed_out, _ = self.bilstm(embs)
        atten_out, _ = self.multihead_attn(query=packed_out, key=packed_out, value=packed_out, key_padding_mask=input_mask)
        #u = torch.tanh(torch.matmul(packed_out, self.w_omega))
        #att = torch.matmul(u, self.u_omega)
        #att_score = F.softmax(att, dim=1)
        #atten_out = packed_out * att_score

        seqence_output= self.layer_norm(atten_out)
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags, input_group=None):
        features = self.forward(input_ids, input_mask)
        return self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
      

    def predict(self, input_ids, input_mask, input_tags, input_lens):
        features = self.forward(input_ids, input_mask)
        tags, _ = self.crf._obtain_labels(features, self.id2label, input_lens)
        return tags