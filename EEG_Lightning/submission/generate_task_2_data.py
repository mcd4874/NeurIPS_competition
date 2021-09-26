from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np
from numpy.random import RandomState
import pickle
import time
import torch
import os
import pandas as pd
import mne

from NeurIPS_2.util.support import (
    expand_data_dim,generate_common_chan_test_data,load_Cho2017,load_Physionet,load_BCI_IV,
    correct_EEG_data_order,relabel,process_target_data,relabel_target,load_dataset_A,load_dataset_B,modify_data,
    generate_data_file,print_dataset_info,print_info,get_dataset_A_ch,get_dataset_B_ch,shuffle_data,EuclideanAlignment,reduce_dataset,LabelAlignment,
    generate_common_target_chans,create_epoch_array,reformat,load_source_data,load_target_data,combine
)

from NeurIPS_2.util.setup_dataset import (
    setup_datasets,setup_specific_subject_dataset_EA
)

def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    elif l == 'feet': return 2
    else: return 3
def relabel_target(l):
    if l == 0: return 0
    elif l == 1: return 1
    elif l ==2 : return 2
    else: return 3

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)


# data_path = "C:\\Users\\wduong\\mne_data\\finalMI"

data_path = "../da_dataset/finalMI"


#get common channel between dataset A and dataset B
dataset_A_channels = get_dataset_A_ch()
dataset_B_channels = get_dataset_B_ch()
X_src1,_,_ = load_Cho2017(subjects=[1])
X_src2, _, _ = load_Physionet(subjects=[1])
X_src3, _, _ = load_BCI_IV(subjects=[1])

import matplotlib
matplotlib.use('TkAgg')

Cho2017_channels = X_src1.ch_names
Physionet_channels = X_src2.ch_names
BCI_IV_channels = X_src3.ch_names

common_channel_A_B = generate_common_target_chans(target_chan=BCI_IV_channels,source_chans=[dataset_A_channels,dataset_B_channels,Cho2017_channels,Physionet_channels])
montage = None
# common_channel_A_B = generate_common_target_chans(target_chan=dataset_A_channels,source_chans=[dataset_B_channels,Cho2017_channels,Physionet_channels])
print("common chan A_B size : ",len(common_channel_A_B))
print("common chan A_B : ",common_channel_A_B)

# subject_ids = None
subject_ids = [1,2,3]
X_src1,y_src1,m_src1 = load_source_data(target_channels=common_channel_A_B,relabel_func=relabel,dataset_name="cho2017",subject_ids=subject_ids)
X_src2,y_src2,m_src2 = load_source_data(target_channels=common_channel_A_B,relabel_func=relabel,dataset_name="physionet",subject_ids=subject_ids)
X_src3,y_src3,m_src3 = load_source_data(target_channels=common_channel_A_B,relabel_func=relabel,dataset_name="BCI_IV",montage=montage,subject_ids=subject_ids)
# #

print("before update meta data : ",m_src2)
X_src2,y_src2,m_src2 = reformat(X_src2,y_src2,m_src2)
X_src2,y_src2,m_src2 = reduce_dataset(X_src2,y_src2,m_src2)
#



source_datasets = [
    (X_src1,y_src1,m_src1,"cho2017"),
    (X_src2,y_src2,m_src2, "physionet"),
    (X_src3,y_src3,m_src3, "BCI_IV")

]


#test 0
## shuffle data
target_dataset_A_name = "dataset_A"
save_folder_A = "../da_dataset/task_2/final_MI_A_1"
test_folder_A = "../da_dataset/task_2/final_MI_test_A_1"

target_dataset_B_name = "dataset_B"
save_folder_B = "../da_dataset/task_2/final_MI_B_1"
test_folder_B = "../da_dataset/task_2/final_MI_test_B_1"


generate_data=True
convert_EA = True

start_id=1
end_id=4
setup_specific_subject_dataset_EA(source_datasets,target_dataset_A_name,common_channel_A_B,path=data_path,save_folder=save_folder_A,test_folder=test_folder_A,generate_folder_data=generate_data,start_id=start_id,end_id=end_id,convert_EA=convert_EA)
start_id=4
end_id=6
setup_specific_subject_dataset_EA(source_datasets,target_dataset_B_name,common_channel_A_B,path=data_path,save_folder=save_folder_B,test_folder=test_folder_B,generate_folder_data=generate_data,start_id=start_id,end_id=end_id,convert_EA=convert_EA)


