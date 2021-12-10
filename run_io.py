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
tag2id = config.label2id['io']
id2tag = {i: label for i, label in enumerate(tag2id)}


train_word_lists = [ x['context'] for x in train_data ]
train_tag_lists = [ x['tag'] for x in train_data ]
dev_word_lists = [ x['context'] for x in dev_data ]
dev_tag_lists = [ x['tag'] for x in dev_data ]


batch_size = 32
train_loader = DataLoader(data=train_data, batch_size=batch_size,
                             shuffle=False, seed=43, sort=False,
                             vocab = processor.vocab, label2id = tag2id)

dev_loader = DataLoader(data=dev_data, batch_size=batch_size,
                             shuffle=False, seed=43, sort=False,
                             vocab = processor.vocab, label2id = tag2id)


def hmm_evaluate():
    # 读取数据
    #训练并评估hmm模型
    print("正在训练HMM模型")
    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists, train_tag_lists, word2id, tag2id)

    # 模型评估
    print("正在评估HMM模型")
    pred_tag_lists = hmm_model.test(dev_word_lists, word2id, tag2id)
    metrics = SeqEntityMetrics(id2tag, markup='io')
    metrics.update(dev_tag_lists, pred_tag_lists)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


def crf_evaluate():
    print("正在训练CRF模型")
    crf_model = CRFModel()
    crf_model.train(train_word_lists,train_tag_lists)

    # 模型评估
    print("正在评估CRF模型")
    pred_tag_lists = crf_model.test(dev_word_lists)
    metrics = SeqEntityMetrics(id2tag, markup='io')
    metrics.update(dev_tag_lists, pred_tag_lists)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


def my_crf_evaluate():
    crf_model = CRFTorchModel(tag2id, len(processor.vocab), device='cuda:0')

    trainer = Trainer(crf_model, id2tag, tag2id, device='gpu', name='mycrf', markup='io')
    trainer.train(train_loader, dev_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


def bilstm_evaluate():
    model = BiLstm(vocab_size=len(processor.vocab), embedding_size=32,
                    hidden_size=100,out_size=len(tag2id))

    trainer = Trainer(model, id2tag, tag2id, device='gpu', name='io-bilstm', markup='io')
    trainer.train(train_loader, dev_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)

def bilstm_attention_evaluate():
    model = BiLstmAttention(vocab_size=len(processor.vocab), embedding_size=32,
                    hidden_size=32,out_size=len(tag2id))

    trainer = Trainer(model, id2tag, tag2id, device='gpu', name='io-bilstm-attn', markup='io')
    trainer.train(train_loader, dev_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)

def bilstm_crf_evaluate():
    model = BiLstmCRFModel(vocab_size=len(processor.vocab), embedding_size=128,
                     hidden_size=384,device='cuda:0',label2id=tag2id)

    trainer = Trainer(model, id2tag, tag2id, device='gpu', name='io-bilstm-crf', markup='io')
    trainer.train(train_loader, dev_loader, epoches=10)
    
    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)

def bilstm_attn_crf_evaluate():
    model = BiLstmCRFAttnModel(vocab_size=len(processor.vocab), embedding_size=128,
                     hidden_size=128,device='cuda:0',label2id=tag2id, drop_p=0.5)

    trainer = Trainer(model, id2tag, tag2id, device='gpu', name='io-bilstm-attn-crf', markup='io')
    trainer.train(train_loader, dev_loader, epoches=50)
    
    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)

if __name__ == "__main__":
    #hmm_evaluate()
    #crf_evaluate()
    #my_crf_evaluate()
    #bilstm_evaluate()
    #bilstm_attention_evaluate()
    #bilstm_crf_evaluate()
    bilstm_attn_crf_evaluate()
