import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import mne
from tools import util
from mne.defaults import HEAD_SIZE_DEFAULT


def generate_trainer_cfg(dataset_root,output_dir,trainer,dataset_config_file_path,train_config_file_path):
    cfg = get_cfg_default()
    # dataset_root = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\da_dataset\\MULTI_DATASET_ADAPTATION_V1\\norm_none"
    # output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_none_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\transfer_adaptation"
    # trainer = "ShareLabelModelAdaptation"
    # dataset_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_none_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\dataset_config\\transfer_adaptation.yaml"
    # train_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_none_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\train_config\\transfer_adaptation.yaml"
    cfg.TRAINER.NAME = trainer
    cfg.DATASET.TRAIN_K_FOLDS = True
    cfg.DATASET.ROOT = dataset_root
    cfg.OUTPUT_DIR = output_dir
    cfg.set_new_allowed(True)
    if dataset_config_file_path:
        cfg.merge_from_file(dataset_config_file_path)
    if train_config_file_path:
        cfg.merge_from_file(train_config_file_path)
    cfg.freeze()
    return cfg

def load_specific_model_trainer(config_file,output_dir,test_fold=1,valid_fold=1,increment_fold=0):
    # output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_none_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\transfer_adaptation"
    # valid_fold = 1
    # test_fold = 1
    # increment_fold = 4
    config_file.DATASET['VALID_FOLD_TEST'] = test_fold
    config_file.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL['CURRENT_FOLD'] = increment_fold
    config_file.DATASET['VALID_FOLD'] = valid_fold
    test_fold_prefix = config_file.TRAIN_EVAL_PROCEDURE.TRAIN.TEST_FOLD_PREFIX
    increment_fold_prefix = config_file.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.INCREMENT_FOLD_PREFIX
    model_dir = os.path.join(output_dir, test_fold_prefix + "_" + str(test_fold))
    model_dir = os.path.join(model_dir, increment_fold_prefix + "_" + str(increment_fold))
    fold_model_dir = os.path.join(model_dir, str(valid_fold))
    model_trainer = build_trainer(config_file)
    model_trainer.load_model(fold_model_dir)
    return model_trainer

def get_process_filter_data(model_architecture):
    temporal_filter_weight = model_architecture['backbone'].backbone.c1.weight
    spatial_filter_weight = model_architecture['backbone'].backbone.c2.weight

    temporal_filter_data = temporal_filter_weight
    temporal_filter_data = temporal_filter_data.cpu().detach().numpy()
    temporal_filter_data = temporal_filter_data.squeeze()

    spatial_filter_data = spatial_filter_weight.cpu().detach().numpy()
    spatial_filter_data = spatial_filter_data.squeeze()
    spatial_filter_data = np.swapaxes(spatial_filter_data, 0, 1)

    return [temporal_filter_data,spatial_filter_data]
def plot_model_temporal_filter(filter_data,sampling_freq=128,montage = None):
    """
    assum the filter data has format (num_filter,kernel_size). We use kernel_size = sampling_freq//2 in EEGNet
    """

    # times = [i for i in range(16)]
    # plt.plot(filter_data[0,])
    # plt.show()
    print("temp filter shape : ",filter_data.shape)
    num_filters = filter_data.shape[0]
    num_rows = int(num_filters // 4)
    # plot the spatial filter
    print("num rows : ",num_rows)
    fig, ax = plt.subplots(ncols=4, nrows=num_rows)
    for i in range(num_rows):
        for j in range(4):
            t = 4*i + j
            print("current filter order is {}".format(t))
            # mne.viz.plot_topomap(evoked.data[:, t], fake_evoked.info, axes=ax[j,i],show=False)
            # ax[j,i].set_title('filter_{}'.format(str(t+1)), fontweight='bold')

            # mne.viz.plot_topomap(evoked.data[:, t], evoked.info, axes=ax[j, i],
            #                      show=False, sphere=(x, y, z, radius))
            print("current data : ",filter_data[t,:])
            ax[i,j].plot(filter_data[t,:])
            ax[i, j].set_title('temporal_filter_{}'.format(str(t + 1)), fontweight='bold')
    # plt.savefig("model_adapt.png")
    plt.show()

def plot_model_spatial_filter(filter_data,sampling_freq=128,montage = None):
    """
    assume filter data has format (channels,num_filter)
    assum that num_filter %4 == 0
    """
    num_spatial_filter = filter_data.shape[1]
    print("there are {} spatial filter in the model".format(num_spatial_filter))

    #make eeg data
    if not montage:
        montage = mne.channels.make_standard_montage('biosemi64')
    ch_names = montage.ch_names
    info = mne.create_info(ch_names, sfreq=sampling_freq, ch_types='eeg')
    evoked = mne.EvokedArray(filter_data, info)
    evoked.set_montage(montage)


    #format the spatial filter to be similar to EEGLAB format
    # first we obtain the 3d positions of selected channels
    # check_ch = ['Oz', 'Fpz', 'T7', 'T8']
    # ch_idx = [evoked.ch_names.index(ch) for ch in check_ch]
    # pos = np.stack([evoked.info['chs'][idx]['loc'][:3] for idx in ch_idx])
    # # now we calculate the radius from T7 and T8 x position
    # # (we could use Oz and Fpz y positions as well)
    # radius = np.abs(pos[[2, 3], 0]).mean()
    # # then we obtain the x, y, z sphere center this way:
    # # x: x position of the Oz channel (should be very close to 0)
    # # y: y position of the T8 channel (should be very close to 0 too)
    # # z: average z position of Oz, Fpz, T7 and T8 (their z position should be the
    # #    the same, so we could also use just one of these channels), it should be
    # #    positive and somewhere around `0.03` (3 cm)
    # x = pos[0, 0]
    # y = pos[-1, 1]
    # z = pos[:, -1].mean()
    # # lets print the values we got:
    # print([f'{v:0.5f}' for v in [x, y, z, radius]])

    num_rows = num_spatial_filter//4
    #plot the spatial filter
    fig, ax = plt.subplots(ncols=4, nrows=num_rows)
    for i in range(num_rows):
        for j in range(4):
            t = 4*i + j

            # mne.viz.plot_topomap(evoked.data[:, t], fake_evoked.info, axes=ax[j,i],show=False)
            # ax[j,i].set_title('filter_{}'.format(str(t+1)), fontweight='bold')
            #
            # mne.viz.plot_topomap(evoked.data[:, t], evoked.info, axes=ax[i,j],
            #                      show=False, sphere=(x, y, z, radius))

            mne.viz.plot_topomap(evoked.data[:, t], evoked.info, axes=ax[i, j],
                                 show=False)

            ax[i,j].set_title('spatial_filter_{}'.format(str(t+1)), fontweight='bold')
    # plt.savefig("model_adapt.png")
    plt.show()

def data_parser(input_data,convert_tensor=True):
    """
    parse data base on the EEGDatum format

    """
    data = input_data.eeg_data.astype(np.float32)
    label = input_data.label
    class_name = input_data.classname

    if convert_tensor:
        data = torch.tensor(np.expand_dims(data, axis=0)).cuda()
        label = torch.tensor(np.expand_dims(label, axis=0)).cuda()

    return [data,label,class_name]


def evaluate(list_models,dataset = None,parser = None):
    """
    assume both model share same test set
    """



    trainer_model_adapt =list_models["adapt"]
    # trainer_model_adapt_v1 = list_models["adapt_v1"]
    # trainer_model_vanilla = list_models["vanilla"]

    # trainer_model_adapt.set_model_mode('eval')
    # trainer_model_adapt_v1.set_model_mode('eval')
    # trainer_model_vanilla.set_model_mode('eval')


    if parser is None:
        parser = data_parser
    if dataset is None:
        data_manager = trainer_model_adapt.data_manager
        dataset = data_manager.dataset
        dataset = dataset.get_raw_test_data()

    test_data = dataset["raw_test_data"]
    test_label = dataset["raw_test_label"]
    test_subject_ids = dataset["raw_subject_ids"]

    def check_threshold(pred_probs,matched,threshold = 0.8):
        pass_probs = pred_probs >= threshold
        combine_match = np.logical_and(pass_probs,matched)
        # print("pred probs : ",pred_probs[:10])
        # print("thres hold probs",pred_probs[:10] >= threshold )
        # print("match : ",matched[:10])
        # print("combine match : ",combine_match[:10])
        #
        return combine_match



    # list_subject_test_loader = data_manager.list_subject_test_loader

    for idx in range(len(test_subject_ids)):
        threshold = 0.7

        data = test_data[idx]
        label = test_label[idx]

        all_correct = np.ones(len(data), dtype=bool)
        threshold_all_correct = np.ones(len(data), dtype=bool)
        all_incorrect = 0
        current_subject_id = test_subject_ids[idx]
        print("current test subject id : ",current_subject_id)
        data = np.array(data).astype(np.float32)
        label = np.squeeze(np.array(label)).astype(int)
        data = torch.tensor(data).cuda()
        # print("data shape : ",data.shape)
        # label = torch.tensor(label).cuda()

        for model_name,model in list_models.items():
            # print("current model is ")
            model.set_model_mode('eval')
            output = model.model_inference(data)
            pred = output.max(1)
            pred_val = pred[0].detach().cpu().numpy()
            class_type = pred[1].detach().cpu().numpy()
            matched = np.equal(class_type,label)

            # print(matched)
            threshold_matched = check_threshold(pred_val,matched,threshold=threshold)

            #located class 0 :
            pred_class_0_loc = np.where(class_type == 0,True,False)
            pred_class_1_loc = np.where(class_type == 1,True,False)

            correct_pred_class_0 = np.logical_and(matched,pred_class_0_loc)
            correct_pred_class_1 = np.logical_and(matched,pred_class_1_loc)

            print("{}_model results for subject {}".format(model_name,current_subject_id))
            print("total trials : ",len(data))
            print("total correct match : ",np.sum(matched.astype(float)))
            print("total correct match for class 0 : ",np.sum(correct_pred_class_0.astype(float)))
            print("total correct match for class 1 : ",np.sum(correct_pred_class_1.astype(float)))

            print("total correct match with threshold {} : {}".format(threshold,np.sum(threshold_matched.astype(float))))
            all_correct = np.logical_and(all_correct,matched)
            threshold_all_correct = np.logical_and(threshold_all_correct,threshold_matched)
        print("total trials that are predicted correct from all three models : {}".format(np.sum(all_correct.astype(float))))
        print("total trials that are predicted correct from all three models with threshold {} : {}".format(threshold,np.sum(threshold_all_correct.astype(float))))

        # print("pred probs : ",pred_val)
        # print("pred class : ",class_type_1)
        # print("label : ",label)
        # matched_1 = class_type_1.eq(label).float()
        # print("matched preds : ",matched_1)
        # matched_1 = matched_1.sum().item()
        # adapt_model_acc = matched_1
        #
        # print("adapt model correct trials : {}".format(adapt_model_acc))

    # for subject_test_loader in list_subject_test_loader:
    #     adapt_model_acc = 0
    #     for batch_idx, batch in enumerate(subject_test_loader):
    #         input, label = trainer_model_adapt.parse_batch_test(batch)
    #         output_1 = trainer_model_adapt.model_inference(input)
    #         pred = output_1.max(1)
    #         pred_val = pred[0]
    #         class_type_1 = pred[1]
    #         print(class_type_1)
    #         matched_1 = class_type_1.eq(label).float().sum().item()
    #         adapt_model_acc +=matched_1
    #     print("adapt model correct trials : {}".format(adapt_model_acc))

    #         print("label : ",label)
    #         print("output  :",output)
    # print("adapt model detail test : ",trainer_model_adapt.detail_test())

    # for subject_test_data in list_test_subject:
    #     all_correct = 0
    #     all_incorrect = 0
    #
    #     num_class_0 = 0
    #     num_class_1 = 0
    #
    #     threshold_class_0 = 0.9
    #     threshold_class_1 = 0.9
    #
    #     num_test_trials = len(subject_test_data)
    #     adapt_model_acc = 0
        # for data in subject_test_data:
        #     # print("current case ")
        #     # print("adapt model")
        #     data,label,class_name = parser(data)
        #     # print("data shape : ",data.shape)
        #     # print("label : ",label)
        #     # print("class name : ",class_name)
        #
        #
        #     output_1 = trainer_model_adapt.model_inference(data)
        #     pred = output_1.max(1)
        #     pred_val = pred[0]
        #     class_type_1 = pred[1]
        #     matched_1 = class_type_1.eq(label).float().item()
        #
        #     # print("current pred : ",pred)
        #     # print("pred[1] : ",pred[1])
        #     # matched = pred.eq(tensor_label)
        #     # print("match : ",matched.cpu().numpy())
        #     # print("output : ",output_1)
        #
        #     # print("adapt_v1 model")
        #     output_2 = trainer_model_adapt_v1.model_inference(data)
        #     pred = output_2.max(1)
        #     pred_val = pred[0]
        #     class_type_2 = pred[1]
        #     matched_2 = class_type_2.eq(label).float().item()
        #     # matched = pred.eq(tensor_label)
        #     # print("match : ", matched.cpu().numpy())
        #     # print(output)
        #     #
        #     # print("vanilla model")
        #     output_3 = trainer_model_vanilla.model_inference(data)
        #     pred = output_3.max(1)
        #     pred_val = pred[0]
        #     class_type_3 = pred[1]
        #     matched_3 = class_type_3.eq(label).float().item()
        #     # print("match : ", matched.cpu().numpy())
        #     # print(output)
        #
        #     # sum = matched_1+matched_2+matched_3
        #     # if sum==0:
        #     #     all_incorrect+=1
        #     # elif sum==3:
        #     #     all_correct+=1
        #
        #     if matched_1 == 1:
        #         adapt_model_acc+=1
        # print("total trials are {}".format(num_test_trials))
        # # print("all corrects are {}".format(all_correct))
        # # print("all incorrect are {}".format(all_incorrect))
        # print("adapt model correct trials : {}".format(adapt_model_acc))



    # full_data = dataset.get_raw_test_data()
    # test_data = full_data["raw_test_data"]
    # test_label = full_data["raw_test_label"]
    # test_subject_ids = full_data["raw_subject_ids"]
    #
    # for idx in range(len(test_subject_ids)):
    #     current_data = test_data[idx]
    #     current_label = test_label[idx]
    #     current_subject = test_subject_ids[idx]
    #     print("test subject id : ", test_subject_ids[idx])
    #     print("data information - test data shape {} , test label shape {} . ".format(test_data[idx].shape, test_label[idx].shape))

        all_correct = 0
        all_incorrect = 0

        num_class_0 = 0
        num_class_1 = 0

        threshold_class_0 = 0.9
        threshold_class_1 = 0.9

        # num_test_trials = len(current_data)
        # adapt_model_acc = 0
        #
        # print("ratio of class 0 : ",len(np.where(current_label==0)[0]))
        # print("ratio of class 1 : ",len(np.where(current_label==1)[0]))


        # data = torch.tensor(current_data).cuda()
        # label = torch.tensor(current_label).cuda()
        # output_1 = trainer_model_adapt.model_inference(data)
        # pred = output_1.max(1)
        # pred_val = pred[0]
        # class_type_1 = pred[1]
        # matched_1 = class_type_1.eq(label).float().sum().item()
        # adapt_model_acc = matched_1

        #
        #
        # for data_idx in range(num_test_trials):
        #     # print("current case ")
        #     # print("adapt model")
        #     data = current_data[data_idx,]
        #     label = current_label[data_idx,]
        #     data = torch.tensor(np.expand_dims(data, axis=0)).cuda()
        #     label = torch.tensor(np.expand_dims(label, axis=0)).cuda()
        #     # print("data shape : ",data.shape)
        #     # data,label,class_name = parser(data)
        #     # print("data shape : ",data.shape)
        #     # print("label : ",label)
        #     # print("class name : ",class_name)
        #
        #
        #     output_1 = trainer_model_adapt.model_inference(data)
        #     pred = output_1.max(1)
        #     pred_val = pred[0]
        #     class_type_1 = pred[1]
        #     matched_1 = class_type_1.eq(label).float().item()
        #
        #     # print("current pred : ",pred)
        #     # print("pred[1] : ",pred[1])
        #     # matched = pred.eq(tensor_label)
        #     # print("match : ",matched.cpu().numpy())
        #     # print("output : ",output_1)
        #
        #     # print("adapt_v1 model")
        #     output_2 = trainer_model_adapt_v1.model_inference(data)
        #     pred = output_2.max(1)
        #     pred_val = pred[0]
        #     class_type_2 = pred[1]
        #     matched_2 = class_type_2.eq(label).float().item()
        #     # matched = pred.eq(tensor_label)
        #     # print("match : ", matched.cpu().numpy())
        #     # print(output)
        #     #
        #     # print("vanilla model")
        #     output_3 = trainer_model_vanilla.model_inference(data)
        #     pred = output_3.max(1)
        #     pred_val = pred[0]
        #     class_type_3 = pred[1]
        #     matched_3 = class_type_3.eq(label).float().item()
        #     # print("match : ", matched.cpu().numpy())
        #     # print(output)
        #
        #     # sum = matched_1+matched_2+matched_3
        #     # if sum==0:
        #     #     all_incorrect+=1
        #     # elif sum==3:
        #     #     all_correct+=1
        #
        #     if matched_1 == 1:
        #         adapt_model_acc+=1
        # print("total trials are {}".format(num_test_trials))
        # print("all corrects are {}".format(all_correct))
        # print("all incorrect are {}".format(all_incorrect))
        # print("adapt model correct trials : {}".format(adapt_model_acc))


    # print(len(list_test_subject))
    # test_subject_data_loaders = trainer_model.get_test_subject_data_loader()
    # num_test_subject =len(test_subject_data_loaders)
    # print("There are {} test subjects data loader ".format(num_test_subject))
    # for test_subject in range(1):
    #     test_data_loader = test_subject_data_loaders[test_subject]
    #     print("current data loader : ",test_data_loader)
    #     for batch_idx, batch in enumerate(test_data_loader):
    #         input, label = trainer_model.parse_batch_test(batch)
    #         output = trainer_model.model_inference(input)
    #         print("label : ",label)
    #         print("output  :",output)

    # detail_result = trainer_model.detail_test()
    # print("detail result : ",detail_result)




#load adaptation model
dataset_root = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\da_dataset\\MULTI_DATASET_ADAPTATION_V1\\norm_zscore"
output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\transfer_adaptation"
trainer = "ShareLabelModelAdaptation"
dataset_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\dataset_config\\transfer_adaptation.yaml"
train_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\train_config\\transfer_adaptation.yaml"

valid_fold = 1
test_fold = 1
increment_fold = 1

config_file = generate_trainer_cfg(dataset_root,output_dir,trainer,dataset_config_file_path,train_config_file_path)
model_trainer_adapt = load_specific_model_trainer(config_file,output_dir,test_fold=test_fold,valid_fold=valid_fold,increment_fold=increment_fold)


# model_architecture = model_trainer_adapt.get_model_architecture()
#
# temporal_filter_data,spatial_filter_data = get_process_filter_data(model_architecture)
# plot_model_temporal_filter(temporal_filter_data)
# plot_model_spatial_filter(spatial_filter_data)
#
# Biosemi_montage = util._read_theta_phi_in_degrees(fname="biosemi64.txt",head_size=HEAD_SIZE_DEFAULT,
#                                      fid_names=['Nz', 'LPA', 'RPA'],
#                                      add_fiducials=False)
#
# plot_model_spatial_filter(spatial_filter_data,montage=Biosemi_montage)
#

#load adaptation_v1 model
dataset_root = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\da_dataset\\MULTI_DATASET_ADAPTATION_V1\\norm_zscore"
output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation_v1\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\transfer_adaptation"
trainer = "ShareLabelModelAdaptationV1"
dataset_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation_v1\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\dataset_config\\transfer_adaptation.yaml"
train_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation_v1\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION\\within_subject\\train_config\\transfer_adaptation.yaml"


config_file = generate_trainer_cfg(dataset_root,output_dir,trainer,dataset_config_file_path,train_config_file_path)
model_trainer_adapt_v1 = load_specific_model_trainer(config_file,output_dir,test_fold=test_fold,valid_fold=valid_fold,increment_fold=increment_fold)

# model_architecture = model_trainer_adapt_v1.get_model_architecture()
#
# weight = model_architecture['backbone'].backbone.c2.weight
# plot_model_filter(weight)
# plt.savefig("model_adapt_v1.png")


#load vanilla model
dataset_root = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\da_dataset\\MULTI_DATASET_ADAPTATION_V1\\norm_zscore"
output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\vanilla\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION_vanilla\\within_subject\\transfer_adaptation_vanilla"
trainer = "BaseModel"
dataset_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\vanilla\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION_vanilla\\within_subject\dataset_config\\transfer_adaptation.yaml"
train_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\vanilla\\norm_zscore_v1\\Biosemi_TRANSFER_ADAPTATION_vanilla\\within_subject\\train_config\\transfer_adaptation.yaml"

# valid_fold = 1
# test_fold = 1
# increment_fold = 4
#
config_file = generate_trainer_cfg(dataset_root,output_dir,trainer,dataset_config_file_path,train_config_file_path)
model_trainer_vanilla = load_specific_model_trainer(config_file,output_dir,test_fold=test_fold,valid_fold=valid_fold,increment_fold=increment_fold)

# model_architecture = model_trainer_vanilla.get_model_architecture()
#
# temporal_filter_data,spatial_filter_data = get_process_filter_data(model_architecture)
# plot_model_spatial_filter(spatial_filter_data)

# plt.savefig("model_vanilla.png")


list_models = {
    "adapt":model_trainer_adapt,
    "adapt_v1":model_trainer_adapt_v1,
    "vanilla":model_trainer_vanilla
}
# evaluate(list_models)

if __name__ == '__main__':
    evaluate(list_models)

# model_architecture = model_trainer.get_model_architecture()


# weight = model_architecture['backbone'].backbone.c2.weight
# plot_model_filter(weight)


# model_architecture = model_trainer.get_model_architecture()
#
# weight = model_architecture['backbone'].backbone.c2.weight
# plot_model_filter(weight)


#load adaptation_v1 model


# weight = model_architecture['backbone'].backbone.c2.weight
# plot_model_filter(weight)

# print("shape : ",weight.shape)
# print(weight[0,0].shape)

# def convert_weight_to_evoke_data(model_weight):
#     biosemi_montage = mne.channels.make_standard_montage('biosemi64')
#     ch_names = biosemi_montage.ch_names
# # print(ch_names)
#     sampling_freq = 128
#     info = mne.create_info(ch_names, sfreq=sampling_freq,ch_types='eeg')
# data = weight[3,0].cpu().detach().numpy()
# fake_evoked = mne.EvokedArray(data, info)
# fake_evoked.set_montage(biosemi_montage)
# weight_EEG_setup = mne.io.RawArray(weight[0,0].cpu().detach().numpy(),info=info).set_montage("biosemi64")
# weight_EEG_setup.plot_sensors(ch_type='eeg')

# create a two-panel figure with some space for the titles at the top
# fig, ax = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw=dict(top=0.9),
#                        sharex=True, sharey=True)

# we plot the channel positions with default sphere - the mne way
# fake_evoked.plot_sensors(axes=ax[0], show=False)

# in the second panel we plot the positions using the EEGLAB reference sphere
# fake_evoked.plot_sensors(sphere=(x, y, z, radius), axes=ax[1], show=False)

# add titles
# ax[0].set_title('MNE channel projection', fontweight='bold')
# plt.show()
# ax[1].set_title('EEGLAB channel projection', fontweight='bold')



# fig, ax = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw=dict(top=0.9),
#                        sharex=True, sharey=True)

# mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax[0],
#                      show=False)
# mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax[1],
#                      show=False, sphere=(x, y, z, radius))
#
# # add titles
# ax[0].set_title('MNE', fontweight='bold')
# ax[1].set_title('EEGLAB', fontweight='bold')
# plt.show()




# ax[0].set_title('MNE', fontweight='bold')
# ax[1].set_title('EEGLAB', fontweight='bold')

#load ABM dataset
#load adaptation model
# dataset_root = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\da_dataset\\MULTI_DATASET_ADAPTATION_V1\\norm_zscore"
# output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\ABM_TRANSFER_ADAPTATION\\within_subject\\transfer_adaptation"
# trainer = "ShareLabelModelAdaptation"
# dataset_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\ABM_TRANSFER_ADAPTATION\\within_subject\dataset_config\\transfer_adaptation.yaml"
# train_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\ABM_TRANSFER_ADAPTATION\\within_subject\\train_config\\transfer_adaptation.yaml"
#
# valid_fold = 1
# test_fold = 1
# increment_fold = 1
#
# config_file = generate_trainer_cfg(dataset_root,output_dir,trainer,dataset_config_file_path,train_config_file_path)
# model_trainer_adapt = load_specific_model_trainer(config_file,output_dir,test_fold=test_fold,valid_fold=valid_fold,increment_fold=increment_fold)
#
#
# model_architecture = model_trainer_adapt.get_model_architecture()
#
# temporal_filter_data,spatial_filter_data = get_process_filter_data(model_architecture)
# plot_model_temporal_filter(temporal_filter_data)
# ABM_montage = util._read_theta_phi_in_degrees(fname="ABMx10.txt",head_size=HEAD_SIZE_DEFAULT,
#                                      fid_names=['Nz', 'LPA', 'RPA'],
#                                      add_fiducials=False)
# plot_model_spatial_filter(spatial_filter_data,montage=ABM_montage)
#
# #load Emotiv dataset
# #load adaptation model
# dataset_root = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\da_dataset\\MULTI_DATASET_ADAPTATION_V1\\norm_zscore"
# output_dir = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\Emotiv_TRANSFER_ADAPTATION\\within_subject\\transfer_adaptation"
# trainer = "ShareLabelModelAdaptation"
# dataset_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\Emotiv_TRANSFER_ADAPTATION\\within_subject\dataset_config\\transfer_adaptation.yaml"
# train_config_file_path = "C:\\wduong_folder\\Dassl.pytorch-master\\Dassl.pytorch-master\\adaptation_experiment\\adaptation\\norm_zscore_v1\\Emotiv_TRANSFER_ADAPTATION\\within_subject\\train_config\\transfer_adaptation.yaml"
#
# valid_fold = 1
# test_fold = 1
# increment_fold = 1
#
# config_file = generate_trainer_cfg(dataset_root,output_dir,trainer,dataset_config_file_path,train_config_file_path)
# model_trainer_adapt = load_specific_model_trainer(config_file,output_dir,test_fold=test_fold,valid_fold=valid_fold,increment_fold=increment_fold)
#
#
# model_architecture = model_trainer_adapt.get_model_architecture()
#
# temporal_filter_data,spatial_filter_data = get_process_filter_data(model_architecture)
# plot_model_temporal_filter(temporal_filter_data)
# ABM_montage = util._read_theta_phi_in_degrees(fname="emotivEPOCH.txt",head_size=HEAD_SIZE_DEFAULT,
#                                      fid_names=['Nz', 'LPA', 'RPA'],
#                                      add_fiducials=False)
# plot_model_spatial_filter(spatial_filter_data,montage=ABM_montage)