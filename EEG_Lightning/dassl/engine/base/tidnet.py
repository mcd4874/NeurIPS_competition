from dassl.engine.base.base_model import BaseModel
from dassl.engine import TRAINER_REGISTRY,TrainerBase

@TRAINER_REGISTRY.register()
class TIDNET(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.ce = nn.CrossEntropyLoss()
        #
        # if cfg.DATASET.TOTAL_CLASS_WEIGHT:
        #     total_data_class_weight = self.dm.dataset.whole_class_weight
        #     if total_data_class_weight is not None:
        #         torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
        #         print("torch weight  : ",torch_weight)
        #         self.ce = nn.CrossEntropyLoss(weight=torch_weight)
        # self._best_epoch_val_loss = 10000

    # def build_model(self):
    #     cfg = self.cfg
    #     print('Building F')
    #     # print("Params key : ",cfg.MODEL.BACKBONE.PARAMS.keys())
    #     self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes,**cfg.MODEL.BACKBONE.PARAMS)
    #     self.F.to(self.device)
    #     print('# params: {:,}'.format(count_num_param(self.F)))
    #     self.optim_F = build_optimizer(self.F, cfg.OPTIM)
    #     self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
    #     self.register_model('F', self.F, self.optim_F, self.sched_F)
    #
    # def forward_backward(self, batch_x, backprob=True):
    #     parsed = self.parse_batch_train(batch_x)
    #     input_x, label_x = parsed
    #     logit_x = self.F(input_x)
    #     loss_x = self.ce(logit_x, label_x)
    #
    #     if backprob:
    #         self.model_backward_and_update(loss_x, 'F')
    #         if (self.batch_idx + 1) == self.num_batches:
    #             self.update_lr()
    #
    #     loss_summary = {
    #         'loss': loss_x.item()
    #     }
    #
    #     return loss_summary
    #
    # def model_inference(self, input):
    #     feat = self.F(input)
    #     preds = F.softmax(feat, 1)
    #     return preds
    #
    # @torch.no_grad()
    # def validate(self):
    #     """A generic testing pipeline."""
    #     self.set_model_mode('eval')
    #     self.evaluator.reset()
    #     losses = MetricMeter()
    #
    #     print('Do evaluation on {} set'.format('valid set'))
    #     data_loader = self.val_loader
    #     assert data_loader is not None
    #     for batch_idx, batch in enumerate(data_loader):
    #         input, label = self.parse_batch_test(batch)
    #         loss = self.forward_backward(batch, backprob=False)
    #         losses.update(loss)
    #         # total_loss += loss['loss']
    #         output = self.model_inference(input)
    #         self.evaluator.process(output, label)
    #
    #     results = self.evaluator.evaluate()
    #     total_loss = losses.meters['loss'].avg
    #
    #     for k, v in results.items():
    #         tag = '{}/{}'.format('validation', k)
    #         self.write_scalar(tag, v, self.epoch)
    #     return total_loss
    #
    # def after_epoch(self):
    #     """
    #     save the best model for given validation loss
    #     """
    #     epoch_total_loss = self.validate()
    #     if self._best_epoch_val_loss > epoch_total_loss:
    #         print("save best model at epoch %f , Improve loss from %4f -> %4f" % (
    #         self.epoch, self._best_epoch_val_loss, epoch_total_loss))
    #         self._best_epoch_val_loss = epoch_total_loss
    #         self.save_model(epoch=self.epoch, directory=self.output_dir, is_best=True)
    #     super().after_epoch()
    # def parse_batch_train(self, batch_x):
    #     input_x = batch_x['eeg_data']
    #     label_x = batch_x['label']
    #     input_x = input_x.to(self.device)
    #     label_x = label_x.to(self.device)
    #     return input_x, label_x