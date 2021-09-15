import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import check_isfile, count_num_param, open_specified_layers
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.modeling import build_head
from dassl.utils import MetricMeter

@TRAINER_REGISTRY.register()
class CustomADDA(TrainerXU):
    """Adversarial Discriminative Domain Adaptation.
    https://arxiv.org/abs/1702.05464.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.open_layers = ['backbone']
        if isinstance(self.model.head, nn.Module):
            self.open_layers.append('head')

        self.source_model = copy.deepcopy(self.model)
        self.source_model.eval()
        for param in self.source_model.parameters():
            param.requires_grad_(False)

        self.build_critic()

        self.bce = nn.BCEWithLogitsLoss()

        self._best_epoch_val_loss = 10000

    def check_cfg(self, cfg):
        assert check_isfile(
            cfg.MODEL.INIT_WEIGHTS
        ), 'The weights of source model must be provided'

    def build_critic(self):
        cfg = self.cfg

        print('Building critic network')
        fdim = self.model.fdim
        critic_body = build_head(
            'mlp',
            verbose=cfg.VERBOSE,
            in_features=fdim,
            hidden_layers=[fdim, fdim // 2],
            activation='leaky_relu'
        )
        self.critic = nn.Sequential(critic_body, nn.Linear(fdim // 2, 1))
        print('# params: {:,}'.format(count_num_param(self.critic)))
        self.critic.to(self.device)
        self.optim_c = build_optimizer(self.critic, cfg.OPTIM)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM)
        self.register_model('critic', self.critic, self.optim_c, self.sched_c)

    def forward_backward(self, batch_x, batch_u,backprob=False):
        open_specified_layers(self.model, self.open_layers)
        input_x, _,_, input_u = self.parse_batch_train(batch_x, batch_u)
        domain_x = torch.ones(input_x.shape[0], 1).to(self.device)
        domain_u = torch.zeros(input_u.shape[0], 1).to(self.device)

        _, feat_x = self.source_model(input_x, return_feature=True)
        _, feat_u = self.model(input_u, return_feature=True)

        logit_xd = self.critic(feat_x)
        logit_ud = self.critic(feat_u.detach())

        loss_critic = self.bce(logit_xd, domain_x)
        loss_critic += self.bce(logit_ud, domain_u)
        if backprob:
            self.model_backward_and_update(loss_critic, 'critic')

        logit_ud = self.critic(feat_u)
        loss_model = self.bce(logit_ud, 1 - domain_u)
        if backprob:
            self.model_backward_and_update(loss_model, 'model')

        loss_summary = {
            'loss_critic': loss_critic.item(),
            'loss_model': loss_model.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    # def validate(self,full_results = False):
    def validate(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        losses = MetricMeter()

        print('Do evaluation on {} set'.format('valid set'))
        data_loader = self.val_loader
        assert data_loader is not None

        self.num_batches = len(self.train_loader_x)
        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            input, label,domain,target = self.parse_batch_train(batch_x,batch_u)
            loss = self.forward_backward(batch_x,batch_u, backprob=False)
            losses.update(loss)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_model'].avg

        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        # if full_results:
        return [total_loss,losses.dict_results(),results]
        # return total_loss
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

    def model_inference(self, input):
        logits = self.model(input)
        preds = F.softmax(logits, 1)
        return preds