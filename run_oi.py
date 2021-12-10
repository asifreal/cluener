import config
import argparse
import torch
from metrics import  SeqEntityMetrics
from data.classifier_processor import ClassifierProcessor
from models.HMM import HMM
from models.CRF import CRFModel, CRFTorchModel
from models.BILSTM_CRF import BiLstmCRFModel, BiLstmCRFAttnModel
from models.BILSTM import BiLstm, BiLstmAttention
from data.data_loader import DataLoader
from models.metrics import AverageMeter
from trainer import Trainer


print("读取数据中...")
processor = ClassifierProcessor(data_dir=config.data_dir)
processor.build_vocab()

train_data = processor.get_train_examples()
dev_data = processor.get_dev_examples()
word2id = processor.vocab.word2idx
tag2id = config.label2id['oi']
id2tag = {i: label for i, label in enumerate(tag2id)}


batch_size = 32
train_loader = DataLoader(data=train_data, batch_size=batch_size,
                             shuffle=False, seed=43, sort=False, key='exists',
                             vocab = processor.vocab, label2id = tag2id)

dev_loader = DataLoader(data=dev_data, batch_size=batch_size,
                             shuffle=False, seed=43, sort=False, key='exists',
                             vocab = processor.vocab, label2id = tag2id)

def bilstm_evaluate():
    model = BiLstm(vocab_size=len(processor.vocab), embedding_size=32,
                    hidden_size=100,out_size=len(tag2id))

    trainer = Trainer(model, id2tag, tag2id, device='gpu', name='oi-bilstm', markup='oi')
    trainer.train(train_loader, dev_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)

def bilstm_attention_evaluate():
    model = BiLstmAttention(vocab_size=len(processor.vocab), embedding_size=128,
                    hidden_size=128,out_size=len(tag2id))

    trainer = Trainer(model, id2tag, tag2id, device='gpu', name='oi-bilstm-attn', markup='oi')
    trainer.train(train_loader, dev_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


if __name__ == "__main__":
    #bilstm_evaluate()
    bilstm_attention_evaluate()
