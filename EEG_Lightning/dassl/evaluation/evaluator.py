import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn import metrics
from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0

        self._y_true = []
        self._y_pred = []
        self._y_probs = []

        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self,mo,gt):
        if isinstance(mo,(np.ndarray)) and isinstance(gt,(np.ndarray)):
            self.numpy_process(mo,gt)
        elif torch.is_tensor(mo) and torch.is_tensor(gt):
            self.torch_process(mo,gt)
        else:
            raise TypeError

    def numpy_process(self,mo,gt):
        pred = np.argmax(mo, axis=1)
        # np.equal()
        matched = np.equal(pred,gt.flatten())
        # self._correct += np.sum(pred == gt.flatten())
        self._correct += np.sum(matched)

        self._total += mo.shape[0]

        self._y_true.extend(gt.flatten())
        self._y_pred.extend(pred)

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i])
                self._per_class_res[label].append(matched_i)

    def torch_process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)
        pred = pred[1]
        matched = pred.eq(gt).float()
        self._correct += int(matched.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i].item())
                self._per_class_res[label].append(matched_i)

    def evaluate(self,info = None):
        results = OrderedDict()
        acc = 100. * self._correct / self._total
        err = 100. - acc
        results['accuracy'] = acc
        results['error_rate'] = err

        if info is not None:
            print(info)

        print(
            '=> result\n'
            '* total: {:,}\n'
            '* correct: {:,}\n'
            '* accuracy: {:.2f}%\n'
            '* error: {:.2f}%'.format(self._total, self._correct, acc, err)
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print('=> per-class result')
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100. * correct / total
                accs.append(acc)
                print(
                    '* class: {} ({})\t'
                    'total: {:,}\t'
                    'correct: {:,}\t'
                    'acc: {:.2f}%'.format(
                        label, classname, total, correct, acc
                    )
                )
                label_name = 'class_{}_acc'.format(label)
                results[label_name] = acc
            print('* average: {:.2f}%'.format(np.mean(accs)))

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize='true'
            )
            save_path = osp.join(self.output_dir, 'cmat.pt')
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results

@EVALUATOR_REGISTRY.register()
class BinaryClassification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._y_probs = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_probs = []

        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)


    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, 2]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matched = pred.eq(gt).float()
        self._correct += int(matched.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._y_probs.extend(mo.data.cpu().numpy()[:,1].tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matched_i = int(matched[i].item())
                self._per_class_res[label].append(matched_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100. * self._correct / self._total
        err = 100. - acc

        fpr, tpr, thresholds = metrics.roc_curve(self._y_true, self._y_probs, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        # auc =  roc_auc_score(self._y_true, self._y_probs)

        results['accuracy'] = acc
        results['error_rate'] = err
        results['auc'] = auc

        print(
            '=> result\n'
            '* total: {:,}\n'
            '* correct: {:,}\n'
            '* accuracy: {:.2f}%\n'
            '* auc: {:.2f}%\n'
            '* error: {:.2f}%'.format(self._total, self._correct, acc, auc,err)
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print('=> per-class result')
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100. * correct / total
                accs.append(acc)
                print(
                    '* class: {} ({})\t'
                    'total: {:,}\t'
                    'correct: {:,}\t'
                    'acc: {:.2f}%'.format(
                        label, classname, total, correct, acc
                    )
                )
                label_name = 'class_{}_acc'.format(label)
                results[label_name] = acc
            print('* average: {:.2f}%'.format(np.mean(accs)))

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize='true'
            )
            save_path = osp.join(self.output_dir, 'cmat.pt')
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
