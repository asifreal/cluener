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


class CRFTorchModel(object):
    def __init__(self,  label2id, vocab_size, max_iterations=100, emb_size=100, device=None):
        self.max_iterations = max_iterations
        self.label2id = label2id
        self.id2label = {i: label for i, label in enumerate(label2id)}
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.fc = nn.Linear(emb_size, len(label2id))
        self.act = nn.Sigmoid()
        self.lossfn= nn.CrossEntropyLoss()
        self.model = CRFTorch(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def train(self, train_loader):
        parameters = [p for p in self.model.parameters() if p.requires_grad] 
        parameters += [p for p in self.embedding.parameters() if p.requires_grad]
        parameters += [p for p in self.fc.parameters() if p.requires_grad]
        optimizer = optim.Adam(parameters, lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                    verbose=1, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8)
        
        for epoch in range(1, self.max_iterations + 1):
            train_loss = AverageMeter()
            self.model.train()
            pbar = tqdm(total=len(train_loader))
            for step, batch in enumerate(train_loader):
                input_ids, input_mask, input_tags, input_lens = batch
                #features = tensor2features(input_ids, len(self.label2id)) #input_ids.unsqueeze(-1).repeat(1, 1, len(self.label2id))   
                features = self.act(self.fc(self.embedding(input_ids)))
                crf_loss = self.model.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
                logits = features.masked_select(
                    (input_mask!=0).unsqueeze(2).expand(-1, -1, len(self.label2id))
                ).contiguous().view(-1, len(self.label2id))
                target = input_tags[input_mask != 0].contiguous().view(-1)
                #print(input_mask.shape, input_tags.shape, features.shape, logits.shape, target.shape)
                label_loss = F.cross_entropy(logits, target)
                loss = label_loss * 100 + crf_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
                train_loss.update(loss.item(), n=1)

                pbar.update(1)
                pbar.set_description(desc='Epoch: {:>5d}'.format(epoch))
                pbar.set_postfix({'loss':'{:.>5f}'.format(loss.item()), 'crfloss':'{:.>5f}'.format(crf_loss.item()), 'clsloss':'{:.>5f}'.format(label_loss.item())})

            #print(f'Epoch: ({epoch} / {self.max_iterations}) - loss: { train_loss.avg}')

    def test(self, dev_loader):
        result = []
        for _, batch in enumerate(dev_loader):
            input_ids, input_mask, input_tags, input_lens = batch
            #features = tensor2features(input_ids, len(self.label2id)) #input_ids.unsqueeze(-1).repeat(1, 1, len(self.label2id))
            features = self.fc(self.embedding(input_ids))
            pred_tag_lists, _ = self.model._obtain_labels(features, self.id2label, input_lens)
            result.extend(pred_tag_lists)
        return result


def tensor2features(input_ids, length):
    m, n = input_ids.shape
    left = (length - 1) // 2
    right = length - 1 - left
    
    buffer = nn.functional.pad( input_ids.repeat(1, n).view(m, n, -1), (left, right) )
    r = torch.LongTensor(m, n, length)
    for i in range(m):
        for j in range(n):
            r[i, j, :] = buffer[i, j, j : j+length]
    return r


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

