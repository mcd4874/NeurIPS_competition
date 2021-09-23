from dassl.engine import TRAINER_REGISTRY
import torch
from torch.nn import functional as F
from dassl.engine.da.heterogeneous.heterogeneous_adaptation_dann import HeterogeneousDANN
import numpy as np




@TRAINER_REGISTRY.register()
class HeterogeneousDANNDSBN(HeterogeneousDANN):
    """
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_dsbn_idx = len(self.dm.source_domains_label_size)
        print("current target dsbn idx : ", self.target_dsbn_idx)

    def build_temp_layer(self, cfg):
        embedding_layer_info = cfg.MODEL.LAYER
        layer_name = embedding_layer_info.NAME
        layer_params = embedding_layer_info.PARAMS
        total_domain = layer_params.total_domain
        check_total_domain = len(self.dm.source_domains_label_size) + 1
        if total_domain != check_total_domain:
            print("there is problem with the provided total domain : ", total_domain)
            layer_params.total_domain = check_total_domain
        print("total domain for DSBN : ", layer_params.total_domain)
        return [layer_name, layer_params]


    def forward_backward(self, batch_x, list_batch_u,backprob = True):
        parsed = self.parse_batch_train(batch_x, list_batch_u)
        input_x, label_x, domain_x, list_input_u,list_label_u,domain_u = parsed
        loss_u = 0
        temp_feat_u = []
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f, d)
            temp_feat_u.append(temp_layer)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)

        loss_u /= len(domain_u)
        f_target = self.TargetFeature(input_x)
        temp_layer_target = self.TemporalLayer(f_target, self.target_dsbn_idx)
        logits_target = self.TargetClassifier(temp_layer_target)
        if backprob:
            loss_x = self.ce(logits_target, label_x)
        else:
            loss_x = self.val_ce(logits_target, label_x)


        feat_u = torch.cat(temp_feat_u, 0)


        global_step = self.batch_idx + self.epoch * self.num_batches
        progress = global_step / (self.max_epoch * self.num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1
        lmda = lmda* self.lmda # modify the scale of lmda
        n_iter = self.epoch * self.num_batches + self.batch_idx
        self.write_scalar('train/lmda', lmda, n_iter)

        feat_x = self.revgrad(temp_layer_target, grad_scaling=lmda)
        feat_u = self.revgrad(feat_u, grad_scaling=lmda)

        # test to combine 2 vector and calculate loss
        loss_d = self.calculate_dann(target_feature=feat_x, source_feature=feat_u)


        total_loss = loss_x + loss_u + loss_d
        loss_summary = {
            'total_loss': total_loss.item(),
            'loss_x': loss_x.item(),
            'loss_u': loss_u.item(),
            'loss_d': loss_d.item(),
            'lmda_factor': lmda

        }

        if backprob:
            self.model_backward_and_update(total_loss)
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

        return loss_summary

    def model_inference(self, input,return_feature=False):
        f = self.TargetFeature(input)
        temp_layer = self.TemporalLayer(f, self.target_dsbn_idx)
        logits = self.TargetClassifier(temp_layer)
        result = F.softmax(logits, 1)
        if return_feature:
            return result,temp_layer
        return result
