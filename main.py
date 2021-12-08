import config
import argparse
import torch
from metrics import  SeqEntityMetrics
from data.cluener_processor import CluenerProcessor
from models.HMM import HMM
from models.CRF import CRFModel, CRFTorchModel
from models.BILSTM_CRF import BiLstmCRFModel
from models.BILSTM import BiLstm
from data.data_loader import DataLoader
from models.metrics import AverageMeter
from trainer import Trainer


print("读取数据中...")
processor = CluenerProcessor(data_dir=config.data_dir)
processor.build_vocab()

train_data = processor.get_train_examples()
dev_data = processor.get_dev_examples()
word2id = processor.vocab.word2idx
tag2id = config.label2id['bios']
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
    metrics = SeqEntityMetrics(id2tag, markup='bios')
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
    metrics = SeqEntityMetrics(id2tag, markup='bios')
    metrics.update(dev_tag_lists, pred_tag_lists)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


def my_crf_evaluate():
    crf_model = CRFTorchModel(tag2id, len(processor.vocab), device='cuda:0')

    trainer = Trainer(crf_model, id2tag, tag2id, device='gpu', name='mycrf')
    trainer.train(train_loader, dev_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


def bilstm_evaluate():
    model = BiLstm(vocab_size=len(processor.vocab), embedding_size=32,
                    hidden_size=100,out_size=len(tag2id))

    trainer = Trainer(model, id2tag, tag2id, device='gpu')
    trainer.train(train_loader, epoches=50)

    metrics, pred_tag_ids = trainer.evaluate(dev_loader)
    overall, class_info = metrics.result()
    metrics.print(overall, class_info)


def bilstm_crf_evaluate():
    print("bilstm+crf模型的评估与训练...")

    train_loader = DataLoader(data=train_data, batch_size=64,
                                 shuffle=False, seed=42, sort=True,
                                 vocab = processor.vocab,label2id = tag2id)
    
    model = BiLstmCRFModel(vocab_size=len(processor.vocab), embedding_size=128,
                     hidden_size=384,device=None,label2id=tag2id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    for epoch in range(1, 1 + 10):
        print(f"Epoch {epoch}/{10}")
        model.train()
        assert model.training
        for step, batch in enumerate(train_loader):
            input_ids, input_mask, input_tags, input_lens = batch
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())
        

    # bilstm_operator = BiLSTM_operator(vocab_size,out_size,crf=True)

    # model_name = "bilstm_crf" if crf else "bilstm"
    # print("start to train the {} ...".format(model_name))
    # bilstm_operator.train(train_word_lists,train_tag_lists,dev_word_lists,dev_tag_lists,word2id,tag2id)

    # print("评估{}模型中...".format(model_name))
    # pred_tag_lists, test_tag_lists = bilstm_operator.test(
    #     test_word_lists, test_tag_lists, word2id, tag2id)

    # metrics = SeqEntityMetrics(id2tag, markup='bios')
    # metrics.update(dev_tag_lists, pred_tag_lists)
    # overall, class_info = metrics.result()
    # metrics.print(overall, class_info)

if __name__ == "__main__":
    #hmm_evaluate()
    #crf_evaluate()
    my_crf_evaluate()
    #bilstm_evaluate()
    #bilstm_crf_evaluate()
