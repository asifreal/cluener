import torch
from models.BILSTM import BiLstm
from models.metrics import AverageMeter
from metrics import SeqEntityMetrics
import time, os
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar

class Trainer():
    def __init__(self, model, id2tag, tag2id, name='bilstm', device=None):
        self.id2tag = id2tag
        self.tag2id =tag2id
        self.name = name
        if device=='gpu':
            self.device = torch.device(f"cuda:0")
        else:
            self.device = 'cpu'
        model.to(self.device)
        self.model = model
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_idr = f'output/{self.name}'
        if not os.path.exists(log_idr):
            os.mkdir(log_idr)
        init_logger(log_file=f'output/{self.name}/{time_}.log')

    def train(self, train_loader, val_loader=None, epoches=20, lr=0.001):
        logger.info(f"{self.name}模型的训练...")
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                    verbose=1, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8)
        best_f1 = 0
        for epoch in range(1, 1 + epoches):
            pbar = ProgressBar(n_total=len(train_loader), desc=f'Training {epoch}/{epoches}')
            train_loss = AverageMeter()
            self.model.train()
            for step, batch in enumerate(train_loader):
                input_ids, input_mask, input_tags, input_lens = batch
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                input_tags = input_tags.to(self.device)
                loss = self.model.forward_loss(input_ids, input_mask, input_lens, input_tags)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                train_loss.update(loss.item(), n=1)
                pbar(step, {'loss': loss.item()})
            print(" ")
            logger.info(f"average loss:{train_loss.avg}")

            train_log = {'loss': train_loss.avg}
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            
            if val_loader is None: continue
            metrics, pred_tag_ids = self.evaluate(val_loader)
            overall, class_info = metrics.result()
            logs = dict(train_log, **overall)
            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            logger.info(show_info)
            scheduler.step(logs['macro-f1'])
            if logs['macro-f1'] > best_f1:
                logger.info(f"\nEpoch {epoch}: macro-f1 improved from {best_f1} to {logs['macro-f1']}")
                logger.info("Eval Entity Score: ")
                best_f1 = logs['macro-f1']
                self.save_model(epoch)
                for key, value in class_info.items():
                    info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                    logger.info(info)
            else:
                logger.info(f"\nEpoch {epoch}: macro-f1 decreased from {best_f1} to {logs['macro-f1']}")

    def evaluate(self, dev_loader, show=False):
        if show: logger.info(f"{self.name}模型的评估...")
        pred_tag_ids = []
        metrics = SeqEntityMetrics(self.id2tag, markup='bios')
        self.model.eval()
        with torch.no_grad():
            pbar = ProgressBar(n_total=len(dev_loader), desc="Evaluating")
            for step, batch in enumerate(dev_loader):
                input_ids, input_mask, input_tags, input_lens = batch
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                input_tags = input_tags.to(self.device)
                batch_tagids = self.model.predict(input_ids, input_mask, input_tags, input_lens)
                pred_batch_tagids = batch_tagids if isinstance(batch_tagids, list)  else batch_tagids.cpu().numpy() 
                metrics.update(input_tags.cpu().numpy(), pred_batch_tagids)
                pred_tag_ids.append(pred_batch_tagids)
                pbar(step)
            print(" ")
        return metrics, pred_tag_ids

    def save_model(self, epoch):
        if isinstance(self.model, torch.nn.DataParallel):
            model_stat_dict = self.model.module.state_dict()
        else:
            model_stat_dict = self.model.state_dict()
        state = {'epoch': epoch, 'arch': self.name, 'state_dict': model_stat_dict}
        model_path =  f'output/{self.name}/best-model.bin'
        logger.info(f"save model to disk {model_path}")
        torch.save(state, str(model_path))