
import numpy as np
from numpy.random import RandomState

import torch
import os
from NeurIPS_competition.util.support import (
    expand_data_dim,normalization,generate_common_chan_test_data,load_Cho2017,load_Physionet,load_BCI_IV,
    correct_EEG_data_order,relabel,process_target_data,relabel_target,load_dataset_A,load_dataset_B,modify_data,
    generate_data_file,print_dataset_info,print_info,get_dataset_A_ch,get_dataset_B_ch,shuffle_data,EuclideanAlignment,reduce_dataset,LabelAlignment,
    generate_common_target_chans,create_epoch_array,reformat,load_source_data,load_target_data,combine
)

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

def print_label_info(subjects_label):
    for subject_idx in range(len(subjects_label)):
        subject_label = subjects_label[subject_idx]
        subject_info = [0]*4
        for label in subject_label:
            subject_info[int(label)] = subject_info[int(label)]+1
        print("subject {} has label info {}".format(subject_idx+1,subject_info))

def setup_datasets(source_datasets, target_dataset_name, common_channels, save_folder="case",generate_folder_data=True):

    train_data, train_label, train_meta, test_data, test_label, test_meta = load_target_data(
        target_channels=common_channels, dataset_name=target_dataset_name)

    temp_train_data, temp_train_label, _ = reformat(train_data, train_label, train_meta)
    print_label_info(temp_train_label)
    temp_test_data, _, _ = reformat(test_data, test_label, test_meta)

    temp_X = np.concatenate([temp_train_data, temp_test_data], axis=1)

    EA = EuclideanAlignment()
    dataset_r_op = EA.generate_list_r_op(temp_X)



    LA = LabelAlignment(target_dataset=(train_data, train_label))
    idx = 1
    for source_dataset in source_datasets:
        X_src, y_src, m_src, dataset_name = source_dataset
        # conduct label alignment
        tmp_X_src, tmp_y_src, tmp_m_src = reformat(X_src, y_src, m_src)
        update_X_src, update_y_src = LA.convert_source_data_with_LA(tmp_X_src, tmp_y_src)
        LA_X_src, LA_y_src, LA_m_src = combine(update_X_src, update_y_src, tmp_m_src)

        m_src = {name: col.values for name, col in m_src.items()}
        LA_m_src = {name: col.values for name, col in LA_m_src.items()}

        print_info(tmp_X_src,dataset_name=dataset_name)
        print_info(update_X_src,dataset_name="LA_"+dataset_name)

        dataset = {
            'data': X_src,
            'label': y_src,
            'meta_data': m_src,
            'dataset_name': dataset_name,
            'chans':common_channels
        }

        LA_dataset = {
            'data': LA_X_src,
            'label': LA_y_src,
            'meta_data': LA_m_src,
            'dataset_name': dataset_name,
            'chans': common_channels

        }
        file_name = 'dataset_{}'.format(idx)
        idx += 1
        if generate_folder_data:
            generate_data_file([dataset], folder_name=save_folder, file_name=file_name)
            generate_data_file([LA_dataset], folder_name=save_folder + '/LA', file_name=file_name)

    target_r_op = {
        'dataset_name': target_dataset_name,
        'r_op_list': dataset_r_op,
        'chans': common_channels
    }

    train_meta = {name: col.values for name, col in train_meta.items()}
    test_meta = {name: col.values for name, col in test_meta.items()}

    target_dataset = {
        'data': train_data,
        'label': train_label,
        'meta_data': train_meta,
        'dataset_name': target_dataset_name,
        'chans': common_channels

    }
    test_dataset = {
        'data': test_data,
        'label': test_label,
        'meta_data': test_meta,
        'dataset_name': target_dataset_name,
        'chans': common_channels

    }
    if generate_folder_data:
        generate_data_file([target_dataset], folder_name=save_folder, file_name='NeurIPS_TL')
        generate_data_file([target_r_op], folder_name=save_folder, file_name=target_dataset_name + '_r_op')

    return test_dataset
def setup_specific_subject_dataset(source_datasets, target_dataset_name, common_channels, save_folder="case",test_folder="test_case",generate_folder_data=True):

    train_data, train_label, train_meta, test_data, test_label, test_meta = load_target_data(
        target_channels=common_channels, dataset_name=target_dataset_name)

    subjects_train_data,subjects_train_label,subjects_train_meta = reformat(train_data, train_label, train_meta)
    subjects_test_data,subjects_test_label,subjects_test_meta = reformat(test_data, test_label, test_meta)


    # temp_train_data, temp_train_label, _ = reformat(train_data, train_label, train_meta)
    # print_label_info(temp_train_label)
    # temp_test_data, _, _ = reformat(test_data, test_label, test_meta)

    temp_X = np.concatenate([subjects_train_data, subjects_test_data], axis=1)

    EA = EuclideanAlignment()
    dataset_r_op = EA.generate_list_r_op(temp_X)



    # LA = LabelAlignment(target_dataset=(train_data, train_label))
    idx = 1
    for source_dataset in source_datasets:
        X_src, y_src, m_src, dataset_name = source_dataset
        # conduct label alignment
        tmp_X_src, tmp_y_src, tmp_m_src = reformat(X_src, y_src, m_src)
        # update_X_src, update_y_src = LA.convert_source_data_with_LA(tmp_X_src, tmp_y_src)
        # LA_X_src, LA_y_src, LA_m_src = combine(update_X_src, update_y_src, tmp_m_src)

        m_src = {name: col.values for name, col in m_src.items()}
        # LA_m_src = {name: col.values for name, col in LA_m_src.items()}

        print_info(tmp_X_src,dataset_name=dataset_name)
        # print_info(update_X_src,dataset_name="LA_"+dataset_name)

        dataset = {
            'data': X_src,
            'label': y_src,
            'meta_data': m_src,
            'dataset_name': dataset_name,
            'chans':common_channels
        }

        # LA_dataset = {
        #     'data': LA_X_src,
        #     'label': LA_y_src,
        #     'meta_data': LA_m_src,
        #     'dataset_name': dataset_name,
        #     'chans': common_channels
        #
        # }
        file_name = 'dataset_{}'.format(idx)
        idx += 1
        if generate_folder_data:
            generate_data_file([dataset], folder_name=save_folder, file_name=file_name)
            # generate_data_file([LA_dataset], folder_name=save_folder + '/LA', file_name=file_name)

    for subject_id in range(len(subjects_train_data)):
        print("current subject id : ",subject_id)
        subject_train_data, subject_train_label, subject_train_meta = subjects_train_data[subject_id],subjects_train_label[subject_id],subjects_train_meta[subject_id]
        subject_test_data, subject_test_label, subject_test_meta = subjects_test_data[subject_id],subjects_test_label[subject_id],subjects_test_meta[subject_id]

        subject_r_op = dataset_r_op[subject_id]

        subject_dataset_name = "{}_{}".format(target_dataset_name,str(subject_id))

        target_r_op = {
            'dataset_name': subject_dataset_name,
            'r_op_list': [subject_r_op],
            'chans': common_channels
        }

        subject_train_meta = {name: col.values for name, col in subject_train_meta.items()}
        subject_test_meta = {name: col.values for name, col in subject_test_meta.items()}

        target_dataset = {
            'data': subject_train_data,
            'label': subject_train_label,
            'meta_data': subject_train_meta,
            'dataset_name': subject_dataset_name,
            'chans': common_channels

        }
        test_dataset = {
            'data': subject_test_data,
            'label': subject_test_label,
            'meta_data': subject_test_meta,
            'dataset_name': subject_dataset_name,
            'chans': common_channels

        }

        subject_save_folder=os.path.join(save_folder,subject_dataset_name)
        subject_test_folder = os.path.join(test_folder,subject_dataset_name)
        if generate_folder_data:
            generate_data_file([target_dataset], folder_name=subject_save_folder, file_name='NeurIPS_TL')
            generate_data_file([target_r_op], folder_name=subject_save_folder, file_name=subject_dataset_name + '_r_op')
            generate_data_file([test_dataset], folder_name=subject_test_folder, file_name='NeurIPS_TL')
            generate_data_file([target_r_op], folder_name=subject_test_folder, file_name=subject_dataset_name + '_r_op')


    # return test_dataset
