import time
import os.path as osp
import os
import datetime
from collections import OrderedDict,defaultdict
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from dassl.data import DataManager,MultiDomainDataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights, generate_path_for_multi_sub_model
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
import pandas as pd

import optuna
from optuna.trial import Trial

import pytorch_lightning as pl


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            pretrained_path = model_cfg.BACKBONE.PRETRAINED_PATH,
            **kwargs
        )
        fdim = self.backbone.out_features

        self.head = None

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)
            if model_cfg.LAST_FC.NAME and  model_cfg.LAST_FC.max_norm > -1.0:
                print("use max norm {} constraint on last FC".format(model_cfg.LAST_FC.max_norm))
                self.classifier = LinearWithConstraint(fdim,num_classes,max_norm=model_cfg.LAST_FC.max_norm)



        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase(pl.LightningModule):
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self._best_epoch_val_loss = 10000
        self._history = defaultdict(list)


    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False,is_checkpoint=False):
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models[name].state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': self._optims[name].state_dict(),
                    'scheduler': self._scheds[name].state_dict(),
                    'best_val_checkpoint':self._best_epoch_val_loss
                },
                osp.join(directory, name),
                is_best=is_best,
                is_checkpoint=is_checkpoint
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print('No checkpoint found, train from scratch')
            return 0

        for name in names:
            path = osp.join(directory, name)
            start_epoch,self._best_epoch_val_loss = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )
        return start_epoch

    def resume_history_if_exist(self,directory):
        file = os.path.join(directory,'history.csv')
        if os.path.exists(file):
            self._history = pd.read_csv(file).to_dict('list')
            print("load available history from : {}".format(file))



    def load_model(self, directory, epoch=None):
        names = self.get_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            print(state_dict.keys())
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode='train', names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Loss is infinite or NaN!')

    def init_writer(self, log_dir):
        if self.__dict__.get('_writer') is None or self._writer is None:
            print(
                'Initializing summary writer for tensorboard '
                'with log_dir={}'.format(log_dir)
            )
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def param_tuning(self, trial: Trial):
        # .report(score, self._trainer.state.epoch)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            epoch_total_loss, validate_losses, validate_results = self.validate()
            trial.report(epoch_total_loss, self.epoch)
            if trial.should_prune():
                message = "Trial was pruned at {} epoch.".format(self.epoch)
                raise optuna.TrialPruned(message)
            if self._best_epoch_val_loss > epoch_total_loss:
                self._best_epoch_val_loss = epoch_total_loss
                self.best_models = self._models.copy()
        names = self.get_model_names()
        for name in names:
            self._models[name].load_state_dict(self.best_models[name].state_dict())
        best_epoch_total_loss, validate_losses, validate_results = self.validate()
        return best_epoch_total_loss

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def analyze_result(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self,batch, backprob=False):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            self._optims[name].zero_grad()

    def model_backward(self, loss,retain_graph = False,create_graph=False):
        self.detect_anomaly(loss)
        loss.backward(retain_graph=retain_graph,create_graph=create_graph)

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            self._optims[name].step()

    def model_backward_and_update(self, loss, names=None,retain_graph = False,create_graph=False):
        self.model_zero_grad(names)
        self.model_backward(loss,retain_graph)
        self.model_update(names)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)
        self.cfg = cfg

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        #generate folder to deal with train-k-folds
        self.TRAIN_K_FOLD = cfg.DATASET.TRAIN_K_FOLDS
        #generate test appropriate test set from the dataset
        self.TEST_K_FOLD = cfg.DATASET.TEST_K_FOLDS

        #folder to save history information:
        history_dir = cfg.TRAIN_EVAL_PROCEDURE.HISTORY_FOLDER
        self.history_dir = osp.join(self.output_dir, history_dir)

        self.INCREMENT_FOLD = (cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_NUM_TRAIN_SUGJECT) > 0 and (cfg.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.NUM_INCREMENT_FOLDS > 1)

        self.history_dir = generate_path_for_multi_sub_model(self.cfg,self.history_dir)
        self.output_dir = generate_path_for_multi_sub_model(self.cfg,self.output_dir)





        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.dm.lab2cname)

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        self.dm = DataManager(self.cfg)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes

    @property
    def data_manager(self):
        return self.dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print('Building model')
        if cfg.MODEL.BACKBONE.PARAMS:
            self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes, **cfg.MODEL.BACKBONE.PARAMS)
        else:
            self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    def create_classifier(self,fdim,num_classes):
        classifier = nn.Linear(fdim, num_classes)
        if self.cfg.MODEL.LAST_FC.NAME and self.cfg.MODEL.LAST_FC.max_norm > -1.0:
            print("use max norm constraint on last FC")
            max_norm=self.cfg.MODEL.LAST_FC.max_norm
            classifier = LinearWithConstraint(fdim, num_classes,max_norm=max_norm)
        return classifier

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    # def param_tuning(self,trial: Trial):
    #     # .report(score, self._trainer.state.epoch)
    #     for self.epoch in range(self.start_epoch, self.max_epoch):
    #         self.run_epoch()
    #         epoch_total_loss, validate_losses, validate_results = self.validate()
    #         trial.report(epoch_total_loss,self.epoch)
    #         if trial.should_prune():
    #             message = "Trial was pruned at {} epoch.".format(self.epoch)
    #             raise optuna.TrialPruned(message)

    def before_train(self):
        directory = self.output_dir
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        self.resume_history_if_exist(self.history_dir)

        # Initialize summary writer
        if self.cfg.DISPLAY_INFO.writer:
            self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print('Finished training')

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # # Save model
        # self.save_model(self.epoch, self.output_dir)

        # # #save history record
        # if self.cfg.TRAIN.SAVE_HISTORY_RECORD and self._history:
        #     print("Saved train history process")
        #     history = pd.DataFrame.from_dict(self._history)
        #     file_output = os.path.join(self.history_dir, 'history.csv')
        #     history.to_csv(file_output, index=False)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()

    def after_epoch(self):

        """Save the best model based on validation loss value"""
        # epoch_total_loss, validate_losses, validate_results = self.validate(full_results=True)
        epoch_total_loss, validate_losses, validate_results = self.validate()
        # save validate information for data analysis
        for val_metric, value in validate_results.items():
            new_val_metric = "val_" + val_metric
            self._history[new_val_metric].append(value)
        for loss_name, val in validate_losses.items():
            new_val_loss = "val_" + loss_name
            self._history[new_val_loss].append(val)

        if self._best_epoch_val_loss > epoch_total_loss:
            print("save best model at epoch %f , Improve loss from %4f -> %4f" % (
                self.epoch, self._best_epoch_val_loss, epoch_total_loss))
            self._best_epoch_val_loss = epoch_total_loss
            self.save_model(epoch=self.epoch, directory=self.output_dir, is_best=True)


        """Generate test results based on the current model """
        not_last_epoch = (self.epoch + 1) != self.max_epoch
        do_test = self.cfg.TEST.EVAL_FREQ > 0 and not self.cfg.TEST.NO_TEST
        meet_test_freq = (
            self.epoch + 1
        ) % self.cfg.TEST.EVAL_FREQ == 0 if do_test else False
        meet_checkpoint_freq = (
            self.epoch + 1
        ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        # if not_last_epoch and do_test and meet_test_freq:
        if  do_test and meet_test_freq:

            test_result = self.test()
            # save infomration to do analysis later
            for test_metric, value in test_result.items():
                new_test_metric = "test_" + test_metric
                self._history[new_test_metric].append(value)



        if not_last_epoch and meet_checkpoint_freq:
            print("save check point from freq ")
            self.save_model(self.epoch, self.output_dir)

        if not not_last_epoch and self.cfg.TRAIN.SAVE_LAST_EPOCH:
            print("save last epoch ")
            self.save_model(self.epoch, self.output_dir)

        #save the model checkpoint in order to resume the training process in case of interruption
        self.save_model(self.epoch,self.output_dir,is_checkpoint=True)

        # save history record every
        if self.cfg.TRAIN.SAVE_HISTORY_RECORD and self._history:
            print("Saved train history process")
            history = pd.DataFrame.from_dict(self._history)
            file_output = os.path.join(self.history_dir, 'history.csv')
            history.to_csv(file_output, index=False)



    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        # data_loader = self.val_loader if split == 'val' else self.test_loader
        data_loader = self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        print("results check : ",results)

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results

    @torch.no_grad()
    def get_EasyTL(self):
        self.set_model_mode('eval')
        self.evaluator.reset()
        import scipy
        import time
        from dassl.engine.da.heterogeneous.easyTL import EasyTL

        train_data = []
        train_label = []
        for batch_idx, batch in enumerate(self.train_loader_x):
            input_x, label_x = self.parse_batch_test(batch)
            _, feature_x = self.model_inference(input_x, return_feature=True)

            train_data.append(feature_x.cpu().numpy())
            train_label.append(label_x.cpu().numpy())
        train_data = np.concatenate(train_data)
        train_label = np.concatenate(train_label)

        test_data = []
        test_label = []
        Acc3=0
        n=0
        for batch_idx, batch in enumerate(self.test_loader):
            input_x, label_x = self.parse_batch_test(batch)
            # for trial_idx in range(label_x.shape[0]):
            #     current_input,current_label = input_x[trial_idx:trial_idx+1,],label_x[trial_idx:trial_idx+1,]
            #
            # # print("shape label x : ",label_x[0:1].shape)
            #     _, feature_x = self.model_inference(current_input, return_feature=True)
            #     acc, yt_prob_tmp = EasyTL(train_data, train_label, feature_x.cpu().numpy(), current_label.cpu().numpy())
            #
            _, feature_x = self.model_inference(input_x, return_feature=True)
            acc, yt_prob_tmp = EasyTL(train_data, train_label, feature_x.cpu().numpy(), label_x.cpu().numpy())
            # # self.evaluator.process(yt_prob_tmp, label_x.cpu().numpy())
            Acc3+=acc
            n+=1
            test_data.append(feature_x.cpu().numpy())
            test_label.append(label_x.cpu().numpy())
        Acc3 = Acc3/n
        test_data = np.concatenate(test_data)
        test_label = np.concatenate(test_label)
        Xs = train_data
        Ys = train_label
        Xt = test_data
        Yt = test_label
        # print("original test label uniue : ", np.unique(test_label))
        t0 = time.time()
        Acc1, _ = EasyTL(Xs, Ys, Xt, Yt, 'raw')
        t1 = time.time()
        print("Time Elapsed: {:.2f} sec".format(t1 - t0))
        # print("EasyTL(c) ACC : {:.1f} %".format(Acc1 * 100))
        Acc2, yt_prob = EasyTL(Xs, Ys, Xt, Yt)
        # self.evaluator.process(yt_prob, Yt)
        # results = self.evaluator.evaluate()
        # Acc3 = results['accuracy']
        t2 = time.time()
        print("Time Elapsed: {:.2f} sec".format(t2 - t1))

        print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1 * 100, Acc2 * 100))
        print("Easy TL 3 : {:.1f} %".format(Acc3*100))
        # print("EAsyTL acc results : ", results)
        return Acc1,Acc2,Acc3

    @torch.no_grad()
    def analyze_result(self):
        # Acc1,Acc2,Acc3 = self.get_EasyTL()
        result =  self.test()
        # result["TL_raw_acc"] = Acc1*100
        # result["TL_coral_acc"] = Acc2*100
        # result["TL_coral_b_acc"] = Acc3*100

        return result
        # return self.test()

    def get_test_subject_data_loader(self):
        return self.dm.list_subject_test_loader

    @torch.no_grad()
    def detail_test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        # print('Do evaluation on {} set'.format(split))
        # data_loader = self.val_loader if split == 'val' else self.test_loader
        list_subject_data_loader = self.dm.list_subject_test_loader
        assert list_subject_data_loader is not None

        # record information for each test subjects
        pick_test_subjects = self.dm.dataset.pick_test_subjects
        num_test_subjects = len(pick_test_subjects)
        test_subject_evaluators = [build_evaluator(self.cfg, lab2cname=self.dm.lab2cname) for i in
                                   range(num_test_subjects)]
        for subject_idx in range(num_test_subjects):
            data_loader = list_subject_data_loader[subject_idx]
            for batch_idx, batch in enumerate(data_loader):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                # print("test subject : ",pick_test_subjects)
                # print("label shaep : ",(label.size()))
                test_subject_evaluators[subject_idx].process(output, label)

        # results = [evaluator.evaluate() for evaluator in test_subject_evaluators]

        # for k, v in results.items():
        #     tag = '{}/{}'.format(split, k)
        #     self.write_scalar(tag, v, self.epoch)

        results_dict = dict()
        for idx in range(num_test_subjects):
            test_subject = pick_test_subjects[idx]
            test_subject_evaluator = test_subject_evaluators[idx]
            test_subject_result = test_subject_evaluator.evaluate()
            results_dict['subject_' + str(test_subject)] = test_subject_result

        print("results dict : ", results_dict)

        return results_dict



    def model_inference(self, input,return_feature=False):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch['eeg_data']
        label = batch['label']
        # domain = batch['']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == 'train_x':
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == 'train_u':
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == 'smaller_one':
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        elif self.cfg.TRAIN.COUNT_ITER == 'bigger_one':
            self.num_batches = max(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
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

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()


    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_u = batch_u['eeg_data']

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, domain_x, input_u



class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.epoch_losses = None

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()
            self.epoch_losses = losses

    def parse_batch_train(self, batch_x):
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        domain_x = domain_x.to(self.device)
        return input_x, label_x, domain_x

from dassl.utils.data_helper import ForeverDataIterator
import numpy as np
class TrainerMultiAdaptation(SimpleTrainer):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.epoch_losses = None

        #perform pre-train on source model prior to train target model
        self.pre_train_source_models = cfg.TRAINER.PARAMS.pretrain
        self.pre_train_epochs = cfg.TRAINER.PARAMS.pretrain_epochs

        n_source_domain = self.dm.dataset.source_num_domain
        print("n source domain : ", n_source_domain)
        n_source_batch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
        # self.split_batch = n_source_batch_size // n_source_domain
        self.n_source_domain = n_source_domain

        # create a cross entropy loss for target dataset
        self.ce = nn.CrossEntropyLoss()
        # self.ce_1 =  nn.CrossEntropyLoss()
        if cfg.DATASET.TOTAL_CLASS_WEIGHT:
            total_data_class_weight = self.dm.dataset.whole_class_weight
            if total_data_class_weight is not None:
                torch_weight = torch.from_numpy(np.array(total_data_class_weight)).float().to(self.device)
                print("target dataset has classes weight  : ", torch_weight)
                self.ce = nn.CrossEntropyLoss(weight=torch_weight)

        # create a cross entropy loss for each source domain dataset
        self.cce = [nn.CrossEntropyLoss() for _ in range(self.n_source_domain)]
        if cfg.DATASET.DOMAIN_CLASS_WEIGHT:
            domain_class_weight = self.dm.source_domains_class_weight
            for domain, weight in domain_class_weight.items():
                print("source domain {} dataset has class weight : {}".format(domain, weight))
                torch_weight = torch.from_numpy(np.array(weight)).float().to(self.device)
                self.cce[domain] = nn.CrossEntropyLoss(weight=torch_weight)

        self.val_ce = nn.CrossEntropyLoss()

        # self.pre_train_source_models = True
        # self.pre_train_epochs = 5

    def build_data_loader(self):
        """Create essential data-related attributes.
        We create multiple train_loader_u wh ere each loader_u is used for 1 source domain
        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        self.dm = MultiDomainDataManager(self.cfg)
        self.train_loader_x = self.dm.train_loader_x
        self.list_train_loader_u = self.dm.list_train_u_loader
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.target_num_classes = self.dm.num_classes
        self.list_source_num_classes = self.dm.source_domains_label_size

    def pre_forward(self,list_batch_u,backprob=True):
        parsed = self.parse_source_batches(list_batch_u)
        list_input_u, list_label_u, domain_u = parsed
        loss_u = 0
        temp_feat_u = []
        for u, y, d in zip(list_input_u, list_label_u, domain_u):
            f = self.SourceFeatures[d](u)
            temp_layer = self.TemporalLayer(f)
            temp_feat_u.append(temp_layer)
            logits = self.SourceClassifiers[d](temp_layer)
            loss_u += self.cce[d](logits, y)

        loss_u /= len(domain_u)
        loss_summary = {
            'loss_u': loss_u.item(),
        }
        if backprob:
            self.model_backward_and_update(loss_u)
        return loss_summary
    def pre_train_source(self):
        # for pretrain_e in range(self.pre_train_epochs):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_of_biggest_train_loader_u = max([len(train_loader_u) for train_loader_u in self.list_train_loader_u])
        list_train_loader_u_iter = [iter(train_loader_u) for train_loader_u in self.list_train_loader_u]
        num_batches = len_of_biggest_train_loader_u
        end = time.time()
        for batch_idx in range(num_batches):
            list_batch_u = list()
            for train_loader_u_iter_idx in range(len(list_train_loader_u_iter)):
                train_loader_u_iter = list_train_loader_u_iter[train_loader_u_iter_idx]
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.list_train_loader_u[train_loader_u_iter_idx])
                    batch_u = next(train_loader_u_iter)
                    list_train_loader_u_iter[train_loader_u_iter_idx] = train_loader_u_iter
                list_batch_u.append(batch_u)
            # print("batch x : ",batch_x.size())
            # print("current train batch x : ",batch_x.size()[0])
            # print("list train batch u ",[batch_u.size()[0] for batch_u in list_batch_u])

            data_time.update(time.time() - end)
            loss_summary = self.pre_forward(list_batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = num_batches - (batch_idx + 1)
                nb_future_epochs = (
                                           self.pre_train_epochs - (self.epoch + 1)
                                   ) * num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'pre-train epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.pre_train_epochs,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * num_batches + batch_idx
            end = time.time()

    # def before_train(self):
    #     super().before_train()
    #     if self.pre_train_source_models and self.pre_train_epochs>0:
    #         self.pre_train_source()

    def after_epoch(self):
        if self.pre_train_source_models and self.pre_train_epochs>0 and self.pre_train_epochs>self.epoch:
            print("save pretrain models")
            self.save_model(self.epoch,self.output_dir,is_checkpoint=True)
        else:
            super().after_epoch()

    def run_epoch(self):
        if self.pre_train_source_models and self.pre_train_epochs>0 and self.pre_train_epochs>self.epoch:
            self.pre_train_source()
        else:
            self.set_model_mode('train')
            losses = MetricMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            # Decide to iterate over target or source dataset
            len_train_loader_x = len(self.train_loader_x)
            len_of_biggest_train_loader_u = max([len(train_loader_u) for train_loader_u in self.list_train_loader_u])
            len_of_smallest_train_loader_u = min([len(train_loader_u) for train_loader_u in self.list_train_loader_u])

            print("loader x batch_size: ",len_train_loader_x)
            print("loader u biggest size : ",len_of_biggest_train_loader_u)
            print("loader u smallest size : ",len_of_smallest_train_loader_u)
            if self.cfg.TRAIN.COUNT_ITER == 'train_x':
                self.num_batches = len_train_loader_x
            elif self.cfg.TRAIN.COUNT_ITER == 'train_u':
                self.num_batches = len_of_biggest_train_loader_u
            elif self.cfg.TRAIN.COUNT_ITER == 'smaller_one':
                self.num_batches = min(len_train_loader_x, len_of_smallest_train_loader_u)
            elif self.cfg.TRAIN.COUNT_ITER == 'bigger_one':
                self.num_batches = max(len_train_loader_x, len_of_biggest_train_loader_u)
            else:
                raise ValueError

            # train_loader_x_iter = iter(self.train_loader_x)
            # list_train_loader_u_iter = [iter(train_loader_u) for train_loader_u in self.list_train_loader_u]

            train_loader_x_iter = ForeverDataIterator(self.train_loader_x)
            list_train_loader_u_iter = [ForeverDataIterator(train_loader_u) for train_loader_u in self.list_train_loader_u]

            # print("initial check on size of each ")
            end = time.time()
            for self.batch_idx in range(self.num_batches):
                # try:
                #     # print("length of iter : ",len(train_loader_x_iter))
                #     batch_x = next(train_loader_x_iter)
                # except StopIteration:
                #     train_loader_x_iter = iter(self.train_loader_x)
                #     batch_x = next(train_loader_x_iter)
                batch_x = next(train_loader_x_iter)
                list_batch_u = list()

                for train_loader_u_iter_idx in range(len(list_train_loader_u_iter)):
                    train_loader_u_iter = list_train_loader_u_iter[train_loader_u_iter_idx]
                    batch_u = next(train_loader_u_iter)
                    # try:
                    #     batch_u = next(train_loader_u_iter)
                    # except StopIteration:
                    #     train_loader_u_iter = iter(self.list_train_loader_u[train_loader_u_iter_idx])
                    #     batch_u = next(train_loader_u_iter)
                    #     list_train_loader_u_iter[train_loader_u_iter_idx] = train_loader_u_iter
                    list_batch_u.append(batch_u)
                # print("batch x : ",batch_x.size())
                # print("current train batch x : ",batch_x.size()[0])
                # print("list train batch u ",[batch_u.size()[0] for batch_u in list_batch_u])

                data_time.update(time.time() - end)
                loss_summary = self.forward_backward(batch_x, list_batch_u)
                batch_time.update(time.time() - end)
                losses.update(loss_summary)

                if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                    nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                    nb_future_epochs = (
                                               self.max_epoch - (self.epoch + 1)
                                       ) * self.num_batches
                    eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(
                        'epoch [{0}/{1}][{2}/{3}]\t'
                        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'eta {eta}\t'
                        '{losses}\t'
                        'lr {lr}'.format(
                            self.epoch + 1,
                            self.max_epoch,
                            self.batch_idx + 1,
                            self.num_batches,
                            batch_time=batch_time,
                            data_time=data_time,
                            eta=eta,
                            losses=losses,
                            lr=self.get_current_lr()
                        )
                    )

                n_iter = self.epoch * self.num_batches + self.batch_idx
                for name, meter in losses.meters.items():
                    self.write_scalar('train/' + name, meter.avg, n_iter)
                self.write_scalar('train/lr', self.get_current_lr(), n_iter)

                end = time.time()

    @torch.no_grad()
    def validate(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        losses = MetricMeter()

        print('Do evaluation on {} set'.format('valid set'))
        data_loader = self.val_loader
        assert data_loader is not None

        num_batches = len(data_loader)
        valid_loader_x_iter = iter(data_loader)

        list_train_loader_u_iter = [iter(train_loader_u) for train_loader_u in self.list_train_loader_u]
        for self.batch_idx in range(num_batches):
            try:
                batch_x = next(valid_loader_x_iter)
            except StopIteration:
                valid_loader_x_iter = iter(data_loader)
                batch_x = next(valid_loader_x_iter)

            list_batch_u = list()
            for train_loader_u_iter_idx in range(len(list_train_loader_u_iter)):
                train_loader_u_iter = list_train_loader_u_iter[train_loader_u_iter_idx]
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.list_train_loader_u[train_loader_u_iter_idx])
                    list_train_loader_u_iter[train_loader_u_iter_idx] = train_loader_u_iter
                    batch_u = next(train_loader_u_iter)
                list_batch_u.append(batch_u)
            input, label, _, _, _, _ = self.parse_batch_train(batch_x, list_batch_u)
            loss = self.forward_backward(batch_x, list_batch_u, backprob=False)
            losses.update(loss)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()
        total_loss = losses.meters['loss_x'].avg
        val_losses = losses.dict_results()
        for k, v in results.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        for k, v in val_losses.items():
            tag = '{}/{}'.format('validation', k)
            self.write_scalar(tag, v, self.epoch)
        return [total_loss, losses.dict_results(), results]

    def parse_target_batch(self,batch_x):
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        domain_x = batch_x['domain']
        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        return input_x, label_x, domain_x

    def parse_source_batches(self,list_batch_u):
        list_input_u = list()
        list_label_u = list()
        for batch_u in list_batch_u:
            input_u = batch_u['eeg_data']
            label_u = batch_u['label']
            input_u = input_u.to(self.device)
            label_u = label_u.to(self.device)
            list_input_u.append(input_u)
            list_label_u.append(label_u)
        domain_u = [d for d in range(len(list_batch_u))]
        return list_input_u,list_label_u,domain_u

    def parse_batch_train(self, batch_x, list_batch_u):
        input_x, label_x, domain_x = self.parse_target_batch(batch_x)
        list_input_u, list_label_u, domain_u = self.parse_source_batches(list_batch_u)
        return input_x, label_x, domain_x, list_input_u,list_label_u,domain_u






