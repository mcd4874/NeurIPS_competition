from dassl.engine import TRAINER_REGISTRY,TrainerBase
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
from dassl.engine.base.base_model import BaseModel

import copy
import scipy.signal as signal
# class filterBank(object):
#     """
#     filter the given signal in the specific bands using cheby2 iir filtering.
#     If only one filter is specified then it acts as a simple filter and returns 2d matrix
#     Else, the output will be 3d with the filtered signals appended in the third dimension.
#     axis is the time dimension along which the filtering will be applied
#     """
#
#     def __init__(self, filtBank, fs, filtAllowance=2, axis=-1, filtType='filter'):
#         self.filtBank = filtBank
#         self.fs = fs
#         self.filtAllowance = filtAllowance
#         self.axis = axis
#         self.filtType = filtType
#
#     def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=-1, filtType='filter'):
#         """
#          Filter a signal using cheby2 iir filtering.
#         Parameters
#         ----------
#         data: 2d/ 3d np array
#             trial x channels x time
#         bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
#             if any value is specified as None then only one sided filtering will be performed
#         fs: sampling frequency
#         filtAllowance: transition bandwidth in hertz
#         filtType: string, available options are 'filtfilt' and 'filter'
#         Returns
#         -------
#         dataOut: 2d/ 3d np array after filtering
#             Data after applying bandpass filter.
#         """
#         aStop = 30  # stopband attenuation
#         aPass = 3  # passband attenuation
#         nFreq = fs / 2  # Nyquist frequency
#
#         if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
#                 bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
#             # no filter
#             print("Not doing any filtering. Invalid cut-off specifications")
#             return data
#
#         elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
#             # low-pass filter
#             print("Using lowpass filter since low cut hz is 0 or None")
#             fPass = bandFiltCutF[1] / nFreq
#             fStop = (bandFiltCutF[1] + filtAllowance) / nFreq
#             # find the order
#             [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
#             b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
#
#         elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
#             # high-pass filter
#             print("Using highpass filter since high cut hz is None or nyquist freq")
#             fPass = bandFiltCutF[0] / nFreq
#             fStop = (bandFiltCutF[0] - filtAllowance) / nFreq
#             # find the order
#             [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
#             b, a = signal.cheby2(N, aStop, fStop, 'highpass')
#
#         else:
#             # band-pass filter
#             # print("Using bandpass filter")
#             fPass = (np.array(bandFiltCutF) / nFreq).tolist()
#             fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
#             # find the order
#             [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
#             b, a = signal.cheby2(N, aStop, fStop, 'bandpass')
#
#         if filtType == 'filtfilt':
#             dataOut = signal.filtfilt(b, a, data, axis=axis)
#         else:
#             dataOut = signal.lfilter(b, a, data, axis=axis)
#         return dataOut
#
#     def __call__(self, data1):
#
#         data = copy.deepcopy(data1)
#         d = data
#         # d = data['data']
#
#         # initialize output
#         out = np.zeros([*d.shape, len(self.filtBank)])
#         # print("out shape : ",out.shape)
#         # repetitively filter the data.
#         for i, filtBand in enumerate(self.filtBank):
#             filter = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
#                                                self.axis, self.filtType)
#             # print("filter shape : ",filter.shape)
#             out[:,:, :, i] =filter
#
#
#         # remove any redundant 3rd dimension
#         if len(self.filtBank) <= 1:
#             out = np.squeeze(out, axis=2)
#
#         # data['data'] = torch.from_numpy(out).float()
#         return out
#         # return data
#
#
# class filterBankFIR(object):
#     """
#     filter the given trial in the specific bands.
#     If only one filter is specified then it acts as a simple filter and returns 2d matrix
#     Else, the output will be 3d with the filtered signals appended in the third dimension.
#     axis is the time dimension along which the filtering will be applied
#     """
#
#     def __init__(self, filtBank, fs, filtOrder=10, axis=-1, filtType='filtfilt'):
#         self.filtBank = filtBank
#         self.fs = fs
#         self.filtOrder = filtOrder
#         self.axis = axis
#         self.filtType = filtType
#
#     def bandpassFilter(self, data, bandFiltCutF, fs, filtOrder=50, axis=-1, filtType='filter'):
#         """
#          Bandpass signal applying FIR filter of given order.
#         Parameters
#         ----------
#         data: 2d/ 3d np array
#             trial x channels x time
#         bandFiltCutF: two element list containing the low and high cut off frequency.
#             if any value is specified as None then only one sided filtering will be performed
#         fs: sampling frequency
#         filtOrder: order of the filter
#         filtType: string, available options are 'filtfilt' and 'filter'
#         Returns
#         -------
#         dataOut: 2d/ 3d np array after filtering
#             Data after applying bandpass filter.
#         """
#         # being FIR filter the a will be [1]
#         a = [1]
#
#         if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
#                 bandFiltCutF[1] == None or bandFiltCutF[1] == fs / 2.0):
#             # no filter
#             print("Not doing any filtering. Invalid cut-off specifications")
#             return data
#         elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
#             # low-pass filter
#             print("Using lowpass filter since low cut hz is 0 or None")
#             h = signal.firwin(numtaps=filtOrder + 1,
#                               cutoff=bandFiltCutF[1], pass_zero="lowpass", fs=fs)
#         elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
#             # high-pass filter
#             print("Using highpass filter since high cut hz is None or nyquist freq")
#             h = signal.firwin(numtaps=filtOrder + 1,
#                               cutoff=bandFiltCutF[0], pass_zero="highpass", fs=fs)
#         else:
#             h = signal.firwin(numtaps=filtOrder + 1,
#                               cutoff=bandFiltCutF, pass_zero="bandpass", fs=fs)
#
#         if filtType == 'filtfilt':
#             dataOut = signal.filtfilt(h, a, data, axis=axis)
#         else:
#             dataOut = signal.lfilter(h, a, data, axis=axis)
#         return dataOut
#
#     def __call__(self, data1):
#
#         data = copy.deepcopy(data1)
#         # d = data['data']
#         d = data
#
#         # initialize output
#         out = np.zeros([*d.shape, len(self.filtBank)])
#
#         # check if the filter order is less than number of samples in
#         # the data else set the order to nsample
#         if self.filtOrder > d.shape[self.axis]:
#             self.filtOrder = self.axis
#
#         # repetitively filter the data.
#         for i, filtBand in enumerate(self.filtBank):
#             out[:, :, i] = self.bandpassFilter(d, filtBand, self.fs, self.filtOrder,
#                                                self.axis, self.filtType)
#
#         # remove any redundant 3rd dimension
#         if len(self.filtBank) <= 1:
#             out = np.squeeze(out, axis=2)
#         return out
#
#         # data['data'] = torch.from_numpy(out).float()
#         # return data
#


@TRAINER_REGISTRY.register()
class FBCNet(TrainerBase):
    """
    Base model that use 1 classifier +backbone.
    Default choice is vanilla EEGNet
    """
    def __init__(self, cfg,require_parameter=None):
        super().__init__(cfg,require_parameter)
# class FBCNet(BaseModel):

    def build_model(self):
        cfg = self.cfg
        print('Building F')
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes,**cfg.MODEL.BACKBONE.PARAMS)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

        # diff = 4
        # filter_bands = []
        # # axis = 2
        # for i in range(1,9):
        #     filter_bands.append([i*diff,(i+1)*diff])
        # print("build filter band : ",filter_bands)
        # self.filter = filterBank(
        #     filtBank= filter_bands,
        #     fs=128
        #     # axis=axis
        # )
    def convert_to_cuda(self,data):
        # data = torch.from_numpy(data)

        data = data.to(self.device)

        return data


    def forward_backward(self, batch_x, backprob=True):
        parsed = self.parse_batch_train(batch_x)
        input_x, label_x, = parsed
        # print("input_x is tensor {}".format(torch.is_tensor(input_x)))
        # print("label_x is tensor {}".format(torch.is_tensor(label_x)))

        if len(input_x.shape) == 4:
            input_x = np.squeeze(input_x,axis=1)
            # print("update dim x : ",input_x.shape)

        #gneerate multi-view with multiple bandpass filter
        # if torch.tensor(input_x):
        input_x = input_x.numpy()
        filter_data = self.filter(input_x)
        filter_data = torch.from_numpy(filter_data).float()
        filter_data = self.convert_to_cuda(filter_data)
        # label_x = self.convert_to_cuda(label_x)

        # print("filter data shape : ",filter_data)
        # print("filter data size: ",filter_data.shape)

        logit_x = self.F(filter_data)

        # if (input_x != input_x).any():
        #     print("nan problem in input")
        if backprob:
            loss_x = self.ce(logit_x, label_x)
            self.model_backward_and_update(loss_x, 'F')
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            loss_x = self.valid_ce(logit_x, label_x)
        loss_summary = {
            'loss': loss_x.item()
        }

        return loss_summary

    def model_inference(self, input,return_feature=False):
    # def model_inference(self, input):

        if len(input.shape) == 4:
            input = np.squeeze(input, axis=1)
            # print("update dim x : ", input.shape)

        input = input.numpy()
        # gneerate multi-view with multiple bandpass filter
        filter_data = self.filter(input)
        filter_data = torch.from_numpy(filter_data).float()
        filter_data = self.convert_to_cuda(filter_data)

        feat = self.F(filter_data)
        preds = F.softmax(feat, 1)
        if return_feature:
            return preds,feat
        return preds


    def get_model_architecture(self):
        model_architecture = {
            "backbone": self.F
        }
        return model_architecture

    def parse_batch_test(self, batch):
        input = batch['eeg_data']
        label = batch['label']
        # domain = batch['']
        # input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_train(self, batch_x):
        """Overide parse
        """
        input_x = batch_x['eeg_data']
        label_x = batch_x['label']
        # input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        return input_x, label_x

