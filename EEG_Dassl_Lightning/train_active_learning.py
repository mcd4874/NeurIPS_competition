import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from dassl.data.datasets.build import build_dataset
from NeurIPS_competition.util.support import (
    expand_data_dim, normalization, generate_common_chan_test_data, load_Cho2017, load_Physionet, load_BCI_IV,
    correct_EEG_data_order, relabel, process_target_data, relabel_target, load_dataset_A, load_dataset_B, modify_data,reformat
)

from train_util import (
    setup_cfg,print_args,reset_cfg,convert_to_dict,CustomModelCheckPoint,CustomeCSVLogger,CustomExperimentWriter,generate_excel_report,
    generate_model_info_config,trainer_setup,generate_setup,train_full_experiment
)

import pytorch_lightning as pl
from util_AL import load_test_data_from_file,load_prediction

# def generate_bag_experiment_MI_label(experiment_test_fold_preds, experiment_test_fold_probs, output_dir, predict_folder="predict_folder",only_count_best=False):
#     final_pred = np.zeros(experiment_test_fold_preds[0][0][0][0].shape)
#     final_prob = np.zeros(experiment_test_fold_probs[0][0][0][0].shape)
#     total_sub_exp = 0
#     for experiment_idx in range(len(experiment_test_fold_preds)):
#         model_preds = experiment_test_fold_preds[experiment_idx]
#         model_probs = experiment_test_fold_probs[experiment_idx]
#         for model_idx in range(len(model_preds)):
#             test_fold_preds=model_preds[model_idx]
#             test_fold_probs=model_probs[model_idx]
#             for test_fold in range(len(test_fold_preds)):
#                 current_fold_preds = test_fold_preds[test_fold]
#                 current_fold_probs = test_fold_probs[test_fold]
#                 for valid_fold in range(len(current_fold_preds)):
#                     current_valid_pred = current_fold_preds[valid_fold]
#                     current_valid_prob = current_fold_probs[valid_fold]
#                     final_pred = final_pred + current_valid_pred
#                     final_prob = final_prob + current_valid_prob
#                     total_sub_exp+=1
#     count= 0
#     pred_output = list()
#     subject_id = 0
#     subject_trials = 200
#     total_best_trial = 0
#
#     subject_info = defaultdict()
#
#     for trial_idx in range(len(final_pred)):
#         if trial_idx % subject_trials == 0:
#             subject_id += 1
#             preds_list_count = [0]*(total_sub_exp+1)
#             subject_info[subject_id] = (preds_list_count)
#             total_best_trial = 0
#
#         trial_pred = final_pred[trial_idx]
#         trial_prob = final_prob[trial_idx]
#         best_idx = np.argmax(trial_pred)
#         best_pred = trial_pred[best_idx]
#         best_prob = trial_prob[best_idx]
#         if best_pred == total_sub_exp:
#             print("subject {} has trial {} has max predict {}".format(subject_id,trial_idx,total_sub_exp))
#             count+=1
#             total_best_trial+=1
#         for idx in range(len(trial_pred)):
#             if idx !=best_idx:
#                 pred = trial_pred[idx]
#                 prob = trial_prob[idx]
#                 if pred > best_pred:
#                     best_pred = pred
#                     best_idx = idx
#                     best_prob = prob
#                 elif pred == best_pred:
#                     print("1 issue")
#                     if prob > best_prob:
#                         best_idx = idx
#                         best_prob = prob
#                     print("trial {} has pred {}, with probs {}, final pick {}".format(trial_idx,trial_pred,trial_prob,best_idx))
#         temp = subject_info[subject_id]
#         temp[int(best_pred)] = temp[int(best_pred)]+1
#         subject_info[subject_id] = temp
#         if only_count_best:
#             if best_pred !=total_sub_exp:
#                 pred_output.append(-1)
#             else:
#                 pred_output.append(best_idx)
#         else:
#             pred_output.append(best_idx)
#     pred_output = np.array(pred_output)
#     combine_folder = os.path.join(output_dir, predict_folder)
#     print("pred output : ",pred_output)
#     print("total count max prediction : ",count)
#     print("subject info : ",subject_info)
#     # np.savetxt(os.path.join(combine_folder, "pred_MI_label.txt"), pred_output, delimiter=',', fmt="%d")
#     return pred_output
#

def generate_bag_experiment_MI_label(fold_predict_results,only_count_best=False,confidence_level=5):

    probs = fold_predict_results[0]["probs"]
    preds = fold_predict_results[0]["preds"]

    final_pred = np.zeros(probs.shape)
    final_prob = np.zeros(preds.shape)
    total_sub_exp = len(fold_predict_results)
    if confidence_level > total_sub_exp:
        confidence_level = total_sub_exp

    for predict_result in fold_predict_results:
        current_prob = predict_result["probs"]
        current_pred = predict_result["preds"]

        final_pred = final_pred + current_pred
        final_prob = final_prob + current_prob

    pred_output = list()
    count_best = 0
    for trial_idx in range(len(final_pred)):
        trial_pred = final_pred[trial_idx]
        trial_prob = final_prob[trial_idx]
        best_idx = np.argmax(trial_pred)
        best_pred = trial_pred[best_idx]
        best_prob = trial_prob[best_idx]
        for idx in range(len(trial_pred)):
            pred = trial_pred[idx]
            prob = trial_prob[idx]
            if pred > best_pred:
                best_pred = pred
                best_idx = idx
                best_prob = prob
            elif pred == best_pred:
                if prob > best_prob:
                    best_idx = idx
                    best_prob = prob
        if only_count_best:
            # if best_pred !=total_sub_exp:
            if best_pred < confidence_level:
                pred_output.append(-1)
            else:
                pred_output.append(best_idx)
                count_best+=1
        else:
            pred_output.append(best_idx)
    pred_output = np.array(pred_output)
    print("There are {} predictions match level {} confidence".format(count_best,confidence_level))
    return pred_output

def load_prediction(experiments_setup,model_update_dir,confidence_level=5):
    fold_predict_results = list()
    for experiment in experiments_setup:
        sub_exp_path = experiment["generate_sub_exp_path"]
        combine_prefix = experiment["sub_exp_prefix"]
        predict_folder = "predict_folder"
        combine_predict_folder = os.path.join(model_update_dir, predict_folder, sub_exp_path)
        if not os.path.exists(combine_predict_folder):
            print("can not find prediction result from {}".format(combine_predict_folder))
        pred = np.loadtxt(os.path.join(combine_predict_folder, 'pred.txt'), delimiter=',')
        probs = np.loadtxt(os.path.join(combine_predict_folder, 'prob.txt'), delimiter=',')
        predict_info = {
            "probs": probs,
            "preds": pred
        }
        predict_info.update(combine_prefix)
        fold_predict_results.append(predict_info)
    pred = generate_bag_experiment_MI_label(fold_predict_results,only_count_best=True,confidence_level=confidence_level)
    # count(pred,name=dataset_type)
    return pred

#
# def load_prediction(dataset_type,common_path,experiment_type,model_list_prefix,augmentation_list_prefix,norm_list_prefix,target_dataset_list_prefix):
#     # test_folds=["test_fold_1"]
#     # increment_folds=["increment_fold_1"]
#     # valid_folds=["valid_fold_1","valid_fold_2","valid_fold_3","valid_fold_4","valid_fold_5"]
#     # predict_folder = "predict_folder"
#     # print("common path before load : ",common_path)
#     experiment_preds = list()
#     experiment_probs = list()
#     for experiment in experiment_type:
#         experiment = [experiment]
#         print("experiment : ",experiment)
#         model_preds = list()
#         model_probs = list()
#         for model_prefix in model_list_prefix:
#             model_prefix = [model_prefix]
#             prefix_list = [experiment,augmentation_list_prefix, norm_list_prefix, model_prefix, target_dataset_list_prefix]
#             list_full_path = generate_data_paths(common_path, prefix_list, [])
#             print("full path : ",list_full_path)
#             path = list_full_path[0]
#
#             test_fold_preds=list()
#             test_fold_probs=list()
#             for test_fold in test_folds:
#                 for increment_fold in increment_folds:
#                     valid_fold_preds=list()
#                     valid_fold_probs=list()
#                     for valid_fold in valid_folds:
#                         generate_path = os.path.join(test_fold,increment_fold,valid_fold)
#                         combine_folder = os.path.join(path, predict_folder, generate_path)
#                         pred = np.loadtxt(os.path.join(combine_folder, 'pred.txt'), delimiter=',')
#                         probs = np.loadtxt(os.path.join(combine_folder, 'prob.txt'), delimiter=',')
#                         valid_fold_preds.append(pred)
#                         valid_fold_probs.append(probs)
#                     test_fold_preds.append(valid_fold_preds)
#                     test_fold_probs.append(valid_fold_probs)
#             model_preds.append(test_fold_preds)
#             model_probs.append(test_fold_probs)
#         experiment_preds.append(model_preds)
#         experiment_probs.append(model_probs)
#         # print(test_fold_preds)
#     # generate_pred_MI_label(test_fold_preds, test_fold_probs, "", predict_folder="predict_folder")
#     pred = generate_bag_experiment_MI_label(experiment_preds,experiment_probs, "", predict_folder="predict_folder",only_count_best=True)
#
#     print(pred)
#     count(pred,name=dataset_type)
#     return pred

def generate_update_dataset_func(test_file_path,model_update_dir,confidence_level=5):

    def update_dataset_with_active_learning(cfg,experiments_setup,exclude_subject_test_trial=None):
        dataset_type = cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME
        test_data, test_label, test_meta = load_test_data_from_file(test_file_path, dataset_type)
        query_pred_label = load_prediction(experiments_setup,model_update_dir,confidence_level=confidence_level)
        # remove_trials = np.where(query_pred_label > -1)[0]
        # if exclude_test_trial:
        #     # query_pred_label = query_pred_label[exclude_test_trial]
        #     # test_data = test_data[exclude_test_trial]
        #     # test_meta = test_meta[exclude_test_trial]
        #     remove_trials = np.intersect1d(remove_trials, exclude_test_trial)

        test_data, query_pred_label, test_meta = reformat(test_data, query_pred_label, test_meta)

        dataset = build_dataset(cfg)

        train_x_data, train_x_label = dataset.train_x
        update_x_data = list()
        update_x_label = list()
        new_test_data = list()
        for subject_idx in range(len(train_x_data)):
            subject_train_x_data, subject_train_x_label = train_x_data[subject_idx], train_x_label[subject_idx]
            subject_test_data, subject_query_pred_label = test_data[subject_idx], query_pred_label[subject_idx]
            # print("subject query label : ",subject_query_pred_label)
            pred_indices = np.where(subject_query_pred_label > -1)[0]
            remain_indices = np.where(subject_query_pred_label == -1)[0]

            update_test_data = subject_test_data[pred_indices]
            update_query_label = subject_query_pred_label[pred_indices]

            remain_test_data = subject_test_data[remain_indices]
            new_test_data.append(remain_test_data)
            # update_test_data, update_query_label = list(), list()
            # for trial in range(len(subject_test_data)):
            #     trial_test_data = subject_test_data[trial]
            #     trial_test_label = subject_query_pred_label[trial]
            #     if trial_test_label > -1:
            #         update_test_data.append(trial_test_data)
            #         update_query_label.append(trial_test_label)
            # update_test_data = np.array(update_test_data)
            # update_query_label = np.array(update_query_label)

            # print("update query label : ",update_query_label)
            # print("shape test data {}, shape test label {}".format(update_test_data.shape,update_query_label.shape))
            # print("shape train data {}, shape train label {}".format(subject_train_x_data.shape,subject_train_x_label.shape))

            combine_subject_data = np.concatenate([subject_train_x_data, update_test_data])
            combine_subject_label = np.concatenate([subject_train_x_label, update_query_label])
            update_x_data.append(combine_subject_data)
            update_x_label.append(combine_subject_label)
            print("new train x data : ",combine_subject_data.shape)
            print("new test data : ",remain_test_data.shape)

        # return update_x_data, update_x_label,new_test_data,new_test_label
        # new_test_data = list()
        # new_test_label = list()

        dataset.set_train_x((update_x_data, update_x_label))
        dataset.set_train_u(new_test_data)
        return dataset
    return update_dataset_with_active_learning
    # data_manager.update_dataset(dataset=dataset)
# def update_datamanager_with_active_learning(cfg, test_file_path, data_manager):
#     dataset_type = cfg.DATAMANAGER.DATASET.SETUP.TARGET_DATASET_NAME
#     test_file_path = args.test_data if args.test_data != '' else None
#
#     # model_list_prefix = [
#     #     # 'adaptationV1',
#     #     # 'dannV1',
#     #     'mcdV1',
#     #     # 'addaV1',
#     #     # 'SRDA'
#     # ]
#     # target_dataset_list_prefix = [
#     #     dataset_type
#     # ]
#     # augmentation_list_prefix = [
#     #     'no_aug',
#     # ]
#     # norm_list_prefix = [
#     #     'no_norm',
#     # ]
#     # experiment_type = ["final_result_14_3_1"]
#     # common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\sub\\{}\\{}\\{}\\{}\\model"
#     # common_path = "C:\\wduong_folder\\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\{}\\sub\\{}\\{}\\{}\\{}\\model"
#     # common_path = "/data1/wduong_experiment_data/EEG_Dassl_Lightning/NeurIPS_competition/{}/sub/{}/{}/{}/{}/model"
#
#     test_data, test_label, test_meta = load_test_data_from_file(test_file_path, dataset_type)
#     query_pred_label = load_prediction(dataset_type, common_path, experiment_type, model_list_prefix,
#                                        augmentation_list_prefix, norm_list_prefix, target_dataset_list_prefix)
#     test_data, query_pred_label, test_meta = reformat(test_data, query_pred_label, test_meta)
#
#     dataset = build_dataset(cfg)
#
#     train_x_data, train_x_label = dataset.train_x
#     update_x_data = list()
#     update_x_label = list()
#     for subject_idx in range(len(train_x_data)):
#         subject_train_x_data, subject_train_x_label = train_x_data[subject_idx], train_x_label[subject_idx]
#         subject_test_data, subject_query_pred_label = test_data[subject_idx], query_pred_label[subject_idx]
#         # print("subject query label : ",subject_query_pred_label)
#         update_test_data, update_query_label = list(), list()
#         for trial in range(len(subject_test_data)):
#             trial_test_data = subject_test_data[trial]
#             trial_test_label = subject_query_pred_label[trial]
#             if trial_test_label > -1:
#                 update_test_data.append(trial_test_data)
#                 update_query_label.append(trial_test_label)
#         update_test_data = np.array(update_test_data)
#         update_query_label = np.array(update_query_label)
#
#         # print("update query label : ",update_query_label)
#         # print("shape test data {}, shape test label {}".format(update_test_data.shape,update_query_label.shape))
#         # print("shape train data {}, shape train label {}".format(subject_train_x_data.shape,subject_train_x_label.shape))
#
#         combine_subject_data = np.concatenate([subject_train_x_data, update_test_data])
#         combine_subject_label = np.concatenate([subject_train_x_label, update_query_label])
#         update_x_data.append(combine_subject_data)
#         update_x_label.append(combine_subject_label)
#     dataset.set_train_x((update_x_data, update_x_label))
#     data_manager.update_dataset(dataset=dataset)

# def main(args):
#     benchmark = False
#     deterministic = False  # this can help to reproduce the result
#     cfg = setup_cfg(args)
#     if cfg.SEED >= 0:
#         print('Setting fixed seed: {}'.format(cfg.SEED))
#         set_random_seed(cfg.SEED)
#     setup_logger(cfg.OUTPUT_DIR)
#
#     if torch.cuda.is_available() and cfg.USE_CUDA:
#         print("use determinstic ")
#         benchmark = False
#         deterministic = True #this can help to reproduce the result
#
#     print('Collecting env info ...')
#     print('** System info **\n{}\n'.format(collect_env_info()))
#
#     print("Experiment setup ...")
#     # cross test fold setup
#     N_TEST_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.N_TEST_FOLDS
#     START_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.START_TEST_FOLD
#     END_TEST_FOLD = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.END_TEST_FOLD
#     TEST_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD.TEST_FOLD_PREFIX
#
#     #shuffle fold
#     SHUFFLE_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.SHUFFLE_FOLD_PREFIX
#     N_SHUFFLE_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS
#     START_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.START_SHUFFLE_FOLD
#     END_SHUFFLE_FOLD = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.END_SHUFFLE_FOLD
#     USE_SHUFFLE = cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD.N_SHUFFLE_FOLDS > 1
#
#
#     #increment fold setup
#     # conduct incremental subject experiments
#     N_INCREMENT_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.N_INCREMENT_FOLDS
#     INCREMENT_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.INCREMENT_FOLD_PREFIX
#     START_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_INCREMENT_FOLD
#     END_INCREMENT_FOLD = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.END_INCREMENT_FOLD
#     USE_INCREMENT = cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD.START_NUM_TRAIN_SUGJECT > 0
#
#     #valid fold setup
#     N_VALID_FOLDS = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.N_VALID_FOLDS
#     START_VALID_FOLD = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.START_VALID_FOLD
#     END_VALID_FOLD = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.END_VALID_FOLD
#     VALID_FOLD_PREFIX = cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD.VALID_FOLD_PREFIX
#
#     data_manager_type = cfg.DATAMANAGER.MANAGER_TYPE
#
#     if args.train_k_folds:
#         generate_detail = True
#         test_folds_results = []
#         test_fold_detail_results = []
#         combine_prefix = dict()
#
#         for current_shuffle_fold in range(START_SHUFFLE_FOLD, END_SHUFFLE_FOLD + 1):
#             cfg.DATAMANAGER.DATASET.SETUP.SHUFFLE_TRAIN_VALID_FOLD['CURRENT_SHUFFLE_FOLD'] = current_shuffle_fold
#             shuffle_fold_prefix = ""
#             if USE_SHUFFLE:
#                 shuffle_fold_prefix = SHUFFLE_FOLD_PREFIX + "_" + str(current_shuffle_fold)
#                 combine_prefix[SHUFFLE_FOLD_PREFIX] = shuffle_fold_prefix
#             for current_increment_fold in range(START_INCREMENT_FOLD, END_INCREMENT_FOLD + 1):
#                 cfg.DATAMANAGER.DATASET.SETUP.INCREMENT_FOLD['CURRENT_INCREMENT_FOLD'] = current_increment_fold
#                 increment_fold_prefix = ""
#                 if USE_INCREMENT:
#                     increment_fold_prefix = INCREMENT_FOLD_PREFIX + "_" + str(current_increment_fold)
#                     combine_prefix[INCREMENT_FOLD_PREFIX] = increment_fold_prefix
#
#                 for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
#                     cfg.DATAMANAGER.DATASET.SETUP.TEST_FOLD['CURRENT_TEST_FOLD'] = current_test_fold
#                     combine_prefix[TEST_FOLD_PREFIX] = TEST_FOLD_PREFIX + "_" + str(current_test_fold)
#                     for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
#                         combine_prefix[VALID_FOLD_PREFIX] = VALID_FOLD_PREFIX+"_"+str(current_valid_fold)
#                         cfg.DATAMANAGER.DATASET.SETUP.VALID_FOLD['CURRENT_VALID_FOLD'] = current_valid_fold
#
#                         output_dir = cfg.OUTPUT_DIR
#                         generate_path = generate_path_for_multi_sub_model(cfg,
#                                                                           test_fold_prefix=combine_prefix[TEST_FOLD_PREFIX],
#                                                                           shuffle_fold_prefix=shuffle_fold_prefix,
#                                                                           increment_fold_prefix=increment_fold_prefix,
#                                                                           valid_fold_prefix=combine_prefix[VALID_FOLD_PREFIX])
#                         output_dir = os.path.join(output_dir, generate_path)
#                         if not os.path.isdir(output_dir):
#                             os.makedirs(output_dir)
#                         cfg.merge_from_list( ["output_dir",output_dir])
#                         print("current output dir : ",output_dir)
#                         pl.seed_everything(42)
#                         cfg_dict = convert_to_dict(cfg,[])
#                         print("cfg dict : ",cfg_dict)
#
#
#
#
#                         if data_manager_type == "single_dataset":
#                             data_manager = DataManagerV1(cfg)
#                         elif data_manager_type == "multi_datasetV2":
#                             print("use data manager for domain adaptation")
#                             data_manager = MultiDomainDataManagerV2(cfg)
#                         else:
#                             print("check multi process")
#                             data_manager = MultiDomainDataManagerV1(cfg)
#
#
#                         #load up
#                         test_file_path = args.test_data if args.test_data != '' else None
#
#                         update_datamanager_with_active_learning(cfg,test_file_path,data_manager)
#
#
#
#
#
#                         data_manager.prepare_data()
#                         data_manager.setup()
#                         require_parameter = data_manager.get_require_parameter()
#                         trainer_model = build_trainer(cfg,require_parameter=require_parameter)
#                         monitor = 'val_loss'
#                         checkpoint_callback = CustomModelCheckPoint(
#                         # checkpoint_callback = ModelCheckpoint(
#                             verbose=True,
#                             monitor = monitor,
#                             dirpath = output_dir,
#                             filename = 'checkpoint',
#                             save_top_k=1,
#                             save_last=True,
#                             every_n_val_epochs=1,
#                             auto_insert_metric_name = False)
#
#                         early_stopping = EarlyStopping(monitor='val_loss',patience=10)
#
#                         csv_logger = CustomeCSVLogger(
#                             save_dir=output_dir,
#                             version=0,
#                             experiment_writer=CustomExperimentWriter,
#                             # step_key='epoch'
#                         )
#                         tensorboard_logger = TensorBoardLogger(
#                             save_dir=output_dir,
#                             version=1
#                         )
#
#
#                         resume_dir = os.path.join(output_dir,'last.ckpt')
#                         # resume_dir = os.path.join(output_dir,'best.ckpt')
#
#                         if os.path.exists(resume_dir):
#                             resume = resume_dir
#                         else:
#                             resume = None
#                         trainer_lightning = Trainer(
#                             gpus=1,
#                             default_root_dir=output_dir,
#                             benchmark=benchmark,
#                             deterministic=deterministic,
#                             max_epochs=cfg.OPTIM.MAX_EPOCH,
#                             resume_from_checkpoint=resume,
#                             multiple_trainloader_mode=cfg.LIGHTNING_TRAINER.multiple_trainloader_mode,
#                             # callbacks=[early_stopping,checkpoint_callback],
#                             callbacks=[checkpoint_callback],
#                             logger=[csv_logger,tensorboard_logger],
#                             progress_bar_refresh_rate=cfg.LIGHTNING_TRAINER.progress_bar_refresh_rate,
#                             profiler=cfg.LIGHTNING_TRAINER.profiler,
#                             num_sanity_val_steps=cfg.LIGHTNING_TRAINER.num_sanity_val_steps,
#                             stochastic_weight_avg=cfg.LIGHTNING_TRAINER.stochastic_weight_avg
#
#                         )
#
#
#                         if not args.eval_only:
#                             trainer_lightning.fit(trainer_model,datamodule=data_manager)
#
#                         else:
#                             # trainer_lightning.checkpoin
#                             model = torch.load(os.path.join(output_dir,'checkpoint.ckpt'),map_location='cuda:0')
#                             # model = torch.load(os.path.join(output_dir,'last.ckpt'))
#
#                             print("save checkpoint keys : ",model.keys())
#                             trainer_model.load_state_dict(model['state_dict'])
#                             # model = trainer_model.load_from_checkpoint(checkpoint_path=os.path.join(output_dir,'best.ckpt'))
#                             # print("load model ",model)
#                             # model = pl.load_from_checkpoint(checkpoint_path=os.path.join(output_dir,'best.ckpt'))
#                             test_result = trainer_lightning.test(trainer_model,datamodule=data_manager)[0]
#                             print("test result : ",test_result                                                                                                                                                                     )
#                             test_result.update(combine_prefix)
#                             test_folds_results.append(test_result)
#             if args.eval_only:
#                 generate_excel_report(test_folds_results, args.output_dir, result_folder='result_folder')
#                 generate_model_info_config(cfg, args.output_dir,result_folder='result_folder')

# def train_full_experiment(cfg,experiments_setup,dataset=None,benchmark=False,deterministic=True,eval=False,use_best_model_pretrain=True,pretrain_dir="",seed=42):
#     fold_results = list()
#     for experiment in experiments_setup:
#         sub_exp_path = experiment["generate_sub_exp_path"]
#         output_dir = experiment["output_dir"]
#         combine_prefix = experiment["sub_exp_prefix"]
#         cfg = experiment["cfg"]
#         trainer_model,trainer_lightning,data_manager = trainer_setup(output_dir,cfg,benchmark,deterministic,seed=seed,provided_dataset=dataset)
#         if eval:
#             model_state = torch.load(os.path.join(output_dir, 'checkpoint.ckpt'), map_location='cuda:0')
#             print("save checkpoint keys : ", model_state.keys())
#             # print("state dict : ", model['state_dict'])
#             trainer_model.load_state_dict(model_state['state_dict'])
#             test_result = trainer_lightning.test(trainer_model, datamodule=data_manager)[0]
#             if len(test_result) > 1 and isinstance(test_result, list):
#                 test_result = test_result[0]
#             print("test result : ", test_result)
#             test_result.update(combine_prefix)
#             fold_results.append(test_result)
#         else:
#             if pretrain_dir != "":
#                 pretrain_dir = os.path.join(args.pretrain_dir, sub_exp_path)
#                 if os.path.exists(pretrain_dir):
#                     if use_best_model_pretrain:
#                         pretrain_dir = os.path.join(pretrain_dir, 'checkpoint.ckpt')
#                     else:
#                         pretrain_dir = os.path.join(pretrain_dir, 'last.ckpt')
#                     pretrain_model_state = torch.load(pretrain_dir, map_location='cuda:0')
#                     trainer_model.load_state_dict(pretrain_model_state['state_dict'])
#                     # require_parameter = data_manager.get_require_parameter()
#                     # trainer_model.load_from_checkpoint(pretrain_dir,cfg=cfg,require_parameter=require_parameter)
#                     print("load pretrain model from {}".format(pretrain_dir))
#             trainer_lightning.fit(trainer_model, datamodule=data_manager)
#
#     if eval:
#         generate_excel_report(fold_results, args.output_dir, result_folder='result_folder')
#         generate_model_info_config(cfg, args.output_dir, result_folder='result_folder')

def main(args):
    benchmark = False
    deterministic = False  # this can help to reproduce the result
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    eval = args.eval_only
    if cfg.LIGHTNING_MODEL.PRETRAIN.DIR != "":
        pretrain_dir = cfg.LIGHTNING_MODEL.PRETRAIN.DIR
        use_best_model_pretrain = cfg.LIGHTNING_MODEL.PRETRAIN.USE_BEST
    else:
        pretrain_dir = args.pretrain_dir
        use_best_model_pretrain = args.use_pretrain_best
        cfg.merge_from_list(["LIGHTNING_MODEL.PRETRAIN.DIR", pretrain_dir])
        cfg.merge_from_list(["LIGHTNING_MODEL.PRETRAIN.USE_BEST", use_best_model_pretrain])

    if torch.cuda.is_available() and cfg.USE_CUDA:
        print("use determinstic ")
        benchmark = False
        deterministic = True #this can help to reproduce the result

    seed = 42
    pl.seed_everything(seed)

    #we use this model to generate prediction to modify the provide dataset setup.
    model_update_dir = args.model_update_dir
    if model_update_dir != "":
        cfg.LIGHTNING_MODEL.ACTIVE_LEARNING.USE_MODEL_UPDATE_DIR = model_update_dir
    elif cfg.LIGHTNING_MODEL.ACTIVE_LEARNING.USE_MODEL_UPDATE_DIR != "":
        model_update_dir = cfg.LIGHTNING_MODEL.ACTIVE_LEARNING.USE_MODEL_UPDATE_DIR

    experiments_setup = generate_setup(cfg)
    test_file_path = args.test_data
    if test_file_path =="" and model_update_dir=="":
        update_dataset_func = None
    else:
        confidence_level = cfg.LIGHTNING_MODEL.ACTIVE_LEARNING.ensemble_confidence_level
        update_dataset_func= generate_update_dataset_func(test_file_path,model_update_dir)
            # update_dataset_with_active_learning(cfg, test_file_path,model_update_dir, experiments_setup)
    train_full_experiment(cfg, experiments_setup,update_dataset_func=update_dataset_func,benchmark=benchmark ,deterministic=deterministic,eval=eval, use_best_model_pretrain=use_best_model_pretrain, pretrain_dir=pretrain_dir,seed=seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument('--test-data', type=str, default='', help='path to test data')
    parser.add_argument('--model-update-dir', type=str, default='', help='path to test data')

    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--main-config-file',
        type=str,
        default='',
        help='path to main config file for full setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )

    parser.add_argument(
        '--pretrain-dir',
        type=str,
        default='',
        help='load pre-train model from this directory'
    )
    parser.add_argument(
        '--use_pretrain_best', action='store_true', help='use best record checkpoint from pre-train path'
    )


    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        '--train-k-folds', action='store_true', help='call trainer.train()'
    )
    parser.add_argument(
        '--tune-models', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        '--param-tuning-file',
        type=str,
        default='',
        help='path to main config file for full setup'
    )
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='gpu '
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    args = parser.parse_args()
    main(args)
