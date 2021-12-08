import torch
from models.BILSTM import BiLstm, cal_loss
from models.metrics import AverageMeter
from metrics import SeqEntityMetrics

class Trainer():
    def __init__(self, id2tag, tag2id):
        self.id2tag = id2tag
        self.tag2id =tag2id
        pass

    def train(self, model, train_loader, dev_loader=None, epoches=20):
        print("bilstm模型的训练...")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                    verbose=1, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8)
        for epoch in range(1, 1 + epoches):
            print(f"Epoch {epoch}/{10}")
            train_loss = AverageMeter()
            model.train()
            assert model.training
            for step, batch in enumerate(train_loader):
                input_ids, input_mask, input_tags, input_lens = batch
                features = model(input_ids, input_lens)
                loss = cal_loss(features, input_tags, input_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                train_loss.update(loss.item(), n=1)
            print("loss: ", train_loss.avg)

    def evaluate(self, model, dev_loader):
        print("bilstm模型的评估...")
        pred_tag_ids = []
        metrics = SeqEntityMetrics(self.id2tag, markup='bios')
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(dev_loader):
                input_ids, input_mask, input_tags, input_lens = batch
                batch_tagids = model.predict(input_ids, input_lens)
                metrics.update(input_tags.numpy(), batch_tagids.numpy())
        overall, class_info = metrics.result()
        metrics.print(overall, class_info)