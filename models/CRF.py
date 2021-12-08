import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn_crfsuite import CRF
from .crf import CRF as CRFTorch
from torch import optim
from tqdm import tqdm
from .metrics import AverageMeter
import torch.nn.functional as F

class CRFModel(object):
    def __init__(self,algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=False):
        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self,sentences,tag_lists):
        # list of lists of dicts
        features = [sent2features(sent) for sent in sentences]
        self.model.fit(features,tag_lists)

    def test(self,sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


class CRFTorchModel(nn.Module):
    def __init__(self,  label2id, vocab_size, emb_size=100, device=None):
        super(CRFTorchModel, self).__init__()
        self.label2id = label2id
        self.id2label = {i: label for i, label in enumerate(label2id)}
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.fc = nn.Linear(emb_size, len(label2id))
        self.dropout = nn.Dropout(0.1)
        self.crf = CRFTorch(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self,inputs_ids,input_lens):
        # [b,l,emb_size ]
        emb = self.dropout(self.embedding(inputs_ids))
        scores = self.fc(emb)
        return scores

    def predict(self, input_ids, input_mask, input_tags, input_lens):
        logits = self.forward(input_ids, input_lens)  # [B, L, out_size]
        pred_tag_lists, _  = self.crf._obtain_labels(logits, self.id2label, input_lens)
        return pred_tag_lists

    def forward_loss(self, inputs_ids, mask, input_lens, input_tags):
        """计算损失
        参数:
            logits: [B, L, out_size]
            targets: [B, L]
            lengths: [B]
        """
        logits = self.forward(inputs_ids, input_lens)
        loss = self.crf.calculate_loss(logits, tag_list=input_tags, lengths=input_lens)
        return loss


def sent2features(sent):
    """抽取序列特征"""
    def word2features(sent,i):
        """抽取单个字的特征"""
        word = sent[i]
        prev_word = "<s>" if i == 0 else sent[i-1]
        next_word = "</s>" if i == (len(sent)-1) else sent[i+1]
        # 使用的特征：
        # 前一个词，当前词，后一个词，
        # 前一个词+当前词， 当前词+后一个词
        feature = {
            'w':word,
            'w-1':prev_word,
            'w+1':next_word,
            'w-1:w':prev_word+word,
            'w:w+1':word+next_word,
            'bias':1
        }
        return feature
    return [word2features(sent,i) for i in range(len(sent))]

