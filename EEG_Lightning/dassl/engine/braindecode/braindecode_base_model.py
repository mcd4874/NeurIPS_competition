from dassl.engine import TRAINER_REGISTRY,TrainerXU,TrainerX
from dassl.data import DataManager
from dassl.utils import MetricMeter
from torch.utils.data import Dataset as TorchDataset
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.engine.trainer_tmp import SimpleNet
import numpy as np

from braindecode.models import Deep4Net

@TRAINER_REGISTRY.register()
class BrainDecode(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ce = nn.CrossEntropyLoss()
        self.valid_ce = nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("torch weight  : ",torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)
        self._best_epoch_val_loss = 10000

        self.ce = torch.nn.NLLLoss()
        self.valid_ce = torch.nn.NLLLoss()

    def build_model(self):
        cfg = self.cfg
        print('Building F')
        self.model = Deep4Net(**cfg.MODEL.BACKBONE.PARAMS)
        print("model shape : ",self.model)
        self.model.to(self.device)

        # self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim_F = build_optimizer(self.model, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.model, self.optim_F, self.sched_F)




    def forward_backward(self, batch_x, backprob=True):
        parsed = self.parse_batch_train(batch_x)
        input_x, label_x, = parsed
        log_softmax_logit = self.model(input_x)
        # if (input_x != input_x).any():
        #     print("nan problem in input")
        if backprob:
            loss_x = self.ce(log_softmax_logit, label_x)
            self.model_backward_and_update(loss_x, 'F')
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            loss_x = self.valid_ce(log_softmax_logit, label_x)
        loss_summary = {
            'loss': loss_x.item()
        }

        return loss_summary

    def model_inference(self, input,return_feature=False):
    # def model_inference(self, input):
        log_softmax_logit = self.model(input)
        preds = log_softmax_logit
        if return_feature:
            return preds,None
        return preds

    # def get_model_architecture(self):
        # model_architecture = {
        #     "backbone": self.F
        # }
        # return model_architecture

    @torch.no_grad()
    def validate(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        losses = MetricMeter()

        print('Do evaluation on {} set'.format('valid set'))
        data_loader = self.val_loader
        assert data_loader is not None
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            loss = self.forward_backward(batch, backprob=False)
            losses.update(loss)
            # total_loss += loss['loss']
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)

        return [total_loss,losses.dict_results(), results]

    def parse_batch_train(self, batch_x):
        """Overide parse
        """
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        input_x = input_x.to(self.device)
        input_x = input_x.permute(0, 2, 3, 1)
        label_x = label_x.to(self.device)
        return input_x, label_x

    def parse_batch_test(self, batch):
        input = batch['eeg_data']
        label = batch['label']
        # domain = batch['']
        input = input.permute(0, 2, 3, 1)

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

