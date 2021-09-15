import numpy as np
import pandas as pd
import os



# #task_1_exp_2 just test_fold 1 ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_2\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_1"


# #task_1_exp_1 just test_fold 1 ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_1\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_2"


# #task_1_exp_2 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_2\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_3"

# #task_1_exp_1 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_1\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_4"

# #task_1_exp_2 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_2\\no_aug\\time_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_5"

# # #task_1_exp_3 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_3\\no_aug\\no_norm\\adaptationV1\\full_dataset\\model\\predict_folder"
# case="task_1_model_6"

# #task_1_exp_3 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_3\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_7"

# #task_1_exp_4 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_4\\tune_kern\\1\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_8"

# #task_1_exp_4 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_4\\tune_n_temp\\2\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_9"

# #task_1_exp_2 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_2\\tune_both_kern\\5\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_10"

# #task_1_exp_4 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_4\\temp_pool_kern\\2\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_11"


# #task_1_exp_5 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\test_sampler\\2\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_12"

# #task_1_exp_5 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\tune_params\\1\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_13"

# # #task_1_exp_3 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_3\\no_aug\\no_norm\\dann\\full_dataset\\model\\predict_folder"
# case="task_1_model_14"

# #task_1_exp_5 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\tune_params\\3\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_16"

# #task_1_exp_5 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_17"

# #task_1_exp_5 use full ensemble
# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\al_pretrain\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_18"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\al_al_pretrain\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_19"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\al_al_pretrain_1\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_20"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\al_al_pretrain_2\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# case="task_1_model_21"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\no_aug\\no_norm\\deepsleep_share_adaptV1\\full_dataset\\model\\predict_folder"
# case="task_1_model_22"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\quick_ver\\no_aug\\no_norm\\deepsleep_share_mcd\\full_dataset\\model\\predict_folder"

# case="task_1_model_23"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_exp_5\\al_pretrain\\no_aug\\no_norm\\deepsleep_share_adaptV1\\full_dataset\\model\\predict_folder"
#
# case="task_1_model_24"



# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_1\\quick_ver\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"

# case="task_1_final_1"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_1\\quick_ver\\no_aug\\no_norm\\vanilla\\full_dataset\\model\\predict_folder"

# case="task_1_final_2"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_1\\no_aug\\time_norm\\share_adaptV1\\full_dataset\\model\\predict_folder"
#
# case="task_1_final_3"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_2\\quick_ver\\no_aug\\time_norm\\deepsleep_share_adaptV1\\full_dataset\\model\\predict_folder"

# case="task_1_final_4"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_2\\quick_ver\\no_aug\\no_norm\\deepsleep_share_adaptV1\\full_dataset\\model\\predict_folder"
#
# case="task_1_final_5"

# result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_2\\quick_ver_0\\no_aug\\no_norm\\deepsleep_share_adaptV1\\full_dataset\\model\\predict_folder"

# case="task_1_final_6"

result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_2\\quick_ver_1_1\\no_aug\\no_norm\\deepsleep_share_adaptV1\\full_dataset\\model\\predict_folder"

case="task_1_final_7"
file = "pred_MI_label.txt"


pred = np.loadtxt(os.path.join(result_path, file), delimiter=',')
path = os.path.join("util",case,"answer.txt")
np.savetxt(path,pred,delimiter=',',fmt="%d")

def count(label,name=""):
    count_0 = len(np.where(label == 0)[0])
    count_1 = len(np.where(label == 1)[0])
    count_2 = len(np.where(label == 2)[0])
    count_3 = len(np.where(label == 3)[0])
    count_4 = len(np.where(label == 4)[0])
    count_5 = len(np.where(label == 5)[0])

    print("dataset {},  has {} label 0, {} label 1, and {} label 2, {} label 3, {} label 4, {} label 5".format(name,count_0,count_1,count_2,count_3,count_4,count_5))

    percent_0 = round(count_0/len(label) *100,2)
    percent_1 = round(count_1/len(label) *100,2)
    percent_2 = round(count_2/len(label) *100,2)
    percent_3 = round(count_3/len(label) *100,2)
    percent_4 = round(count_4/len(label) *100,2)
    percent_5 = round(count_5/len(label) *100,2)

    print("dataset {},  has {}% label 0, {}% label 1, and {}% label 2, {}% label 3, {}% label 4, {}% label 5".format(name,percent_0,percent_1,percent_2,percent_3,percent_4,percent_5))


count(pred,"current")

