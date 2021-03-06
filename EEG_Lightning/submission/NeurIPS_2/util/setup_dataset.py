
import numpy as np
from numpy.random import RandomState
import pandas as pd
import torch
import os
from NeurIPS_2.util.support import (
    expand_data_dim,normalization_channels,normalization_time,generate_common_chan_test_data,load_Cho2017,load_Physionet,load_BCI_IV,
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

def setup_datasets(source_datasets, target_dataset_name, common_channels, save_folder="case",generate_folder_data=True,path=None,start_id=1,end_id=3):

    train_data, train_label, train_meta, test_data, test_label, test_meta = load_target_data(path=path,
        target_channels=common_channels, dataset_name=target_dataset_name,start_id=start_id,end_id=end_id)

    temp_train_data, temp_train_label, _ = reformat(train_data, train_label, train_meta)
    print_label_info(temp_train_label)
    print_info(temp_train_data,"target_MI_"+target_dataset_name)
    temp_test_data, _, _ = reformat(test_data, test_label, test_meta)
    print_info(temp_test_data,"test_MI_"+target_dataset_name)

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

def combine_phase_1_phase_2_MI(target_dataset_name, common_channels,path_1=None,path_2=None,start_id_1=1,end_id_1=3,start_id_2=1,end_id_2=4):
    train_data_1, train_label_1, train_meta_1, _, _, _ = load_target_data(path=path_1,
                                                                                             target_channels=common_channels,
                                                                                             dataset_name=target_dataset_name,
                                                                                             start_id=start_id_1,
                                                                                             end_id=end_id_1)
    train_data_2, train_label_2, train_meta_2, test_data, test_label, test_meta = load_target_data(path=path_2,
                                                                                             target_channels=common_channels,
                                                                                             dataset_name=target_dataset_name,
                                                                                             start_id=start_id_2,
                                                                                             end_id=end_id_2)

    train_data_1, train_label_1, _ = reformat(train_data_1, train_label_1, train_meta_1)
    train_data_2, train_label_2, _ = reformat(train_data_2, train_label_2, train_meta_2)
    update_train_data = train_data_1+train_data_2
    update_train_label = train_label_1+train_label_2
    update_subject_ids =[[subject]*len(update_train_label[subject]) for subject in range(len(update_train_label))]

    update_train_data = np.concatenate(update_train_data)
    update_train_label = np.concatenate(update_train_label)
    update_subject_ids = np.concatenate(update_subject_ids)
    assert len(update_train_label) == len(update_subject_ids)
    update_meta = pd.DataFrame({"subject":update_subject_ids,"session":["session_0"]*len(update_subject_ids),"run":["run_0"]*len(update_subject_ids)})

    return update_train_data,update_train_label,update_meta,test_data, test_label, test_meta

def setup_phase_1_2_dataset(source_datasets,target_dataset_name, common_channels, save_folder="case",generate_folder_data=True,path_1=None,path_2=None,start_id_1=1,end_id_1=3,start_id_2=1,end_id_2=4):
    train_data, train_label, train_meta, test_data, test_label, test_meta = combine_phase_1_phase_2_MI(target_dataset_name, common_channels, path_1=path_1, path_2=path_2, start_id_1=start_id_1, end_id_1=end_id_1,
                               start_id_2=start_id_2, end_id_2=end_id_2)

    temp_train_data, temp_train_label, _ = reformat(train_data, train_label, train_meta)
    print_label_info(temp_train_label)
    print_info(temp_train_data, "target_MI_" + target_dataset_name)

    generate_source_datasets_file(source_datasets, common_channels, save_folder, generate_folder_data)
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
    return test_dataset

def generate_source_datasets_file(source_datasets,common_channels,save_folder,generate_folder_data,convert_EA=False):
    # idx = 1
    # for source_dataset in source_datasets:
    #     X_src, y_src, m_src, dataset_name = source_dataset
    #     # conduct label alignment
    #     tmp_X_src, tmp_y_src, tmp_m_src = reformat(X_src, y_src, m_src)
    #     m_src = {name: col.values for name, col in m_src.items()}
    #     print_info(tmp_X_src,dataset_name=dataset_name)
    #     dataset = {
    #         'data': X_src,
    #         'label': y_src,
    #         'meta_data': m_src,
    #         'dataset_name': dataset_name,
    #         'chans':common_channels
    #     }
    #     file_name = 'dataset_{}'.format(idx)
    #     idx += 1
    #     if generate_folder_data:
    #         generate_data_file([dataset], folder_name=save_folder, file_name=file_name)

    idx = 1
    for source_dataset in source_datasets:
        X_src, y_src, m_src, dataset_name = source_dataset
        # conduct label alignment
        tmp_X_src, tmp_y_src, tmp_m_src = reformat(X_src, y_src, m_src)

        if convert_EA:
            EA = EuclideanAlignment()
            tmp_X_src = EA.convert_subjects_data_with_EA(tmp_X_src)

        X_src, y_src, m_src = combine(tmp_X_src,tmp_y_src, tmp_m_src)
        print("{} dtype {}".format(dataset_name,X_src.dtype))


        m_src = {name: col.values for name, col in m_src.items()}
        print_info(tmp_X_src,dataset_name=dataset_name)
        dataset = {
            'data': X_src,
            'label': y_src,
            'meta_data': m_src,
            'dataset_name': dataset_name,
            'chans':common_channels
        }
        file_name = 'dataset_{}'.format(idx)
        idx += 1
        if generate_folder_data:
            generate_data_file([dataset], folder_name=save_folder, file_name=file_name)

def generate_LA_source_dataset_file(LA,source_datasets,common_channels,save_folder,generate_folder_data,convert_EA=False):
    idx = 1
    for source_dataset in source_datasets:
        X_src, y_src, m_src, dataset_name = source_dataset
        # conduct label alignment
        tmp_X_src, tmp_y_src, tmp_m_src = reformat(X_src, y_src, m_src)
        update_X_src, update_y_src = LA.convert_source_data_with_LA(tmp_X_src, tmp_y_src)

        if convert_EA:
            EA = EuclideanAlignment()
            update_X_src = EA.convert_subjects_data_with_EA(update_X_src)

        LA_X_src, LA_y_src, LA_m_src = combine(update_X_src, update_y_src, tmp_m_src)
        print("LA {} dtype {}".format(dataset_name,LA_X_src.dtype))

        LA_m_src = {name: col.values for name, col in LA_m_src.items()}
        print_info(update_X_src,dataset_name="LA_"+dataset_name)
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
            generate_data_file([LA_dataset], folder_name=save_folder + '/LA', file_name=file_name)


def setup_specific_subject_dataset(source_datasets, target_dataset_name, common_channels, save_folder="case",test_folder="test_case",generate_folder_data=True,start_id=1,end_id=3,path=None):

    train_data, train_label, train_meta, test_data, test_label, test_meta = load_target_data(path=path,
        target_channels=common_channels, dataset_name=target_dataset_name,start_id=start_id,end_id=end_id)

    subjects_train_data,subjects_train_label,subjects_train_meta = reformat(train_data, train_label, train_meta)
    subjects_test_data,subjects_test_label,subjects_test_meta = reformat(test_data, test_label, test_meta)


    # temp_train_data, temp_train_label, _ = reformat(train_data, train_label, train_meta)
    # print_label_info(temp_train_label)
    # temp_test_data, _, _ = reformat(test_data, test_label, test_meta)

    temp_X = np.concatenate([subjects_train_data, subjects_test_data], axis=1)

    EA = EuclideanAlignment()
    dataset_r_op = EA.generate_list_r_op(temp_X)
    # EA = EuclideanAlignment(list_r_op=dataset_r_op)
    # subjects_train_data = EA.convert_subjects_data_with_EA(subjects_train_data)
    # subjects_test_label = EA.convert_subjects_data_with_EA(subjects_test_label)

    generate_source_datasets_file(source_datasets, common_channels, save_folder, generate_folder_data)

    for subject_id in range(len(subjects_train_data)):
        print("current subject id : ",subject_id)
        subject_train_data, subject_train_label, subject_train_meta = subjects_train_data[subject_id],subjects_train_label[subject_id],subjects_train_meta[subject_id]
        subject_test_data, subject_test_label, subject_test_meta = subjects_test_data[subject_id],subjects_test_label[subject_id],subjects_test_meta[subject_id]


        subject_r_op = dataset_r_op[subject_id]

        subject_dataset_name = "{}_{}".format(target_dataset_name,str(subject_id))
        subject_save_folder=os.path.join(save_folder,subject_dataset_name)
        subject_test_folder = os.path.join(test_folder,subject_dataset_name)


        ##generate subject specific LA for source data
        LA = LabelAlignment(target_dataset=(subject_train_data, subject_train_label))
        generate_LA_source_dataset_file(LA,source_datasets, common_channels, subject_save_folder, generate_folder_data)


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


        if generate_folder_data:
            generate_data_file([target_dataset], folder_name=subject_save_folder, file_name='NeurIPS_TL')
            generate_data_file([target_r_op], folder_name=subject_save_folder, file_name=subject_dataset_name + '_r_op')
            generate_data_file([test_dataset], folder_name=subject_test_folder, file_name='NeurIPS_TL')
            generate_data_file([target_r_op], folder_name=subject_test_folder, file_name=subject_dataset_name + '_r_op')


def setup_specific_subject_dataset_EA(source_datasets, target_dataset_name, common_channels, save_folder="case",test_folder="test_case",generate_folder_data=True,start_id=1,end_id=3,path=None,convert_EA=False):

    train_data, train_label, train_meta, test_data, test_label, test_meta = load_target_data(path=path,
        target_channels=common_channels, dataset_name=target_dataset_name,start_id=start_id,end_id=end_id)

    subjects_train_data,subjects_train_label,subjects_train_meta = reformat(train_data, train_label, train_meta)
    subjects_test_data,subjects_test_label,subjects_test_meta = reformat(test_data, test_label, test_meta)


    # temp_train_data, temp_train_label, _ = reformat(train_data, train_label, train_meta)
    # print_label_info(temp_train_label)
    # temp_test_data, _, _ = reformat(test_data, test_label, test_meta)

    temp_X = np.concatenate([subjects_train_data, subjects_test_data], axis=1)

    EA = EuclideanAlignment()
    dataset_r_op = EA.generate_list_r_op(temp_X)
    if convert_EA:
        EA = EuclideanAlignment(list_r_op=dataset_r_op)
        subjects_train_data_EA = EA.convert_subjects_data_with_EA(subjects_train_data)
        subjects_test_data_EA = EA.convert_subjects_data_with_EA(subjects_test_data)


    generate_source_datasets_file(source_datasets, common_channels, save_folder, generate_folder_data,convert_EA)

    for subject_id in range(len(subjects_train_data)):
        print("current subject id : ",subject_id)
        subject_train_data, subject_train_label, subject_train_meta = subjects_train_data[subject_id],subjects_train_label[subject_id],subjects_train_meta[subject_id]
        subject_test_data, subject_test_label, subject_test_meta = subjects_test_data[subject_id],subjects_test_label[subject_id],subjects_test_meta[subject_id]


        # subject_r_op = dataset_r_op[subject_id]

        subject_dataset_name = "{}_{}".format(target_dataset_name,str(subject_id))
        subject_save_folder=os.path.join(save_folder,subject_dataset_name)
        subject_test_folder = os.path.join(test_folder,subject_dataset_name)


        ##generate subject specific LA for source data
        LA = LabelAlignment(target_dataset=(subject_train_data, subject_train_label))
        generate_LA_source_dataset_file(LA,source_datasets, common_channels, subject_save_folder, generate_folder_data,convert_EA)

        if convert_EA:
            subject_train_data = subjects_train_data_EA[subject_id]
            subject_test_data = subjects_test_data_EA[subject_id]


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


        if generate_folder_data:
            generate_data_file([target_dataset], folder_name=subject_save_folder, file_name='NeurIPS_TL')
            generate_data_file([test_dataset], folder_name=subject_test_folder, file_name='NeurIPS_TL')

def generate_LA_dataset(LA,source_datasets,convert_EA=False):
    list_LA_datasets=list()
    for source_dataset in source_datasets:
        X_src, y_src, m_src, dataset_name = source_dataset
        # conduct label alignment
        tmp_X_src, tmp_y_src, tmp_m_src = reformat(X_src, y_src, m_src)
        update_X_src, update_y_src = LA.convert_source_data_with_LA(tmp_X_src, tmp_y_src)


        if convert_EA:
            EA = EuclideanAlignment()
            update_X_src = EA.convert_subjects_data_with_EA(update_X_src)

        LA_X_src, LA_y_src, LA_m_src = combine(update_X_src, update_y_src, tmp_m_src)
        # LA_m_src = {name: col.values for name, col in LA_m_src.items()}

        LA_source = (LA_X_src, LA_y_src, LA_m_src,dataset_name)
        list_LA_datasets.append(LA_source)
    return list_LA_datasets
def setup_specific_subject_filterbank_dataset(source_datasets, target_dataset_name, common_channels, save_folder="case",test_folder="test_case",generate_folder_data=True,start_id=1,end_id=3,path=None,fmin=4,fmax=36,seed=42,convert_EA=False):

    np.random.seed(seed)
    # print("filter data between")
    train_data, train_label, train_meta, test_data, test_label, test_meta = load_target_data(path=path,
        target_channels=common_channels, dataset_name=target_dataset_name,start_id=start_id,end_id=end_id,fmin=fmin,fmax=fmax)


    subjects_train_data,subjects_train_label,subjects_train_meta = reformat(train_data, train_label, train_meta)
    subjects_test_data,subjects_test_label,subjects_test_meta = reformat(test_data, test_label, test_meta)






    subject_id = start_id
    assert end_id-start_id == 1 and len(subjects_train_data) == 1

    # for subject_id in range(len(subjects_train_data)):
    print("current subject id : ",subject_id)


    ##generate subject specific LA for source data
    assert len(subjects_train_data) == 1
    LA = LabelAlignment(target_dataset=(subjects_train_data[0],subjects_train_label[0]))

    temp_X = np.concatenate([subjects_train_data, subjects_test_data], axis=1)
    EA = EuclideanAlignment()
    dataset_r_op = EA.generate_list_r_op(temp_X)
    if convert_EA:
        EA = EuclideanAlignment(list_r_op=dataset_r_op)
        subjects_train_data_EA = EA.convert_subjects_data_with_EA(subjects_train_data)
        subjects_test_data_EA = EA.convert_subjects_data_with_EA(subjects_test_data)
        subjects_train_data = subjects_train_data_EA
        subjects_test_data = subjects_test_data_EA

    list_LA_datasets = generate_LA_dataset(LA,source_datasets,convert_EA=convert_EA)
    subjects_train_data, subjects_train_label, subjects_train_meta = combine(subjects_train_data,subjects_train_label,subjects_train_meta)
    subjects_test_data, subjects_test_label, subjects_test_meta = combine(subjects_test_data,subjects_test_label,subjects_test_meta )

    target_dataset = (subjects_train_data,subjects_train_label,subjects_train_meta,target_dataset_name)
    test_dataset = (subjects_test_data, subjects_test_label, subjects_test_meta,target_dataset_name)

    return target_dataset,test_dataset,list_LA_datasets,

