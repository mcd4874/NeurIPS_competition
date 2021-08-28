import numpy as np
import pandas as pd
import os

#case 1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_3\\temp_aug\chan_norm\\adaptation\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_3\\temp_aug\chan_norm\\adaptation\dataset_B\model\\predict_folder"

# # case 2
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case2"

# # case 2-1 still need to submit this. use aug for dataset A and no aug for dataset B
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\no_aug\chan_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case2_1"

# case 2-2
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\FBCNET_adaptV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\FBCNET_adaptV1\dataset_B\model\\predict_folder"
# case="case2_2"

# # case 3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_6\\no_aug\chan_norm\\adaptation\\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_6\\no_aug\chan_norm\\adaptation\\dataset_B\model\\predict_folder"
# case="case3"

# case 3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_6\\temp_aug\chan_norm\\adaptationV1\\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_6\\temp_aug\chan_norm\\adaptationV1\\dataset_B\model\\predict_folder"
# case="case3_1"

#case 4-5 ,model 8
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\temp_aug\chan_norm\\adaptation\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\temp_aug\chan_norm\\adaptation\dataset_B\model\\predict_folder"
# case="case4_5"

#case 4-3 ,model 9
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_3\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_3\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case4_3"

#case 4-5 ,model 10
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_10"

#case 7-1 ,model 11
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_11"

#case 7-1-1 ,model 12
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_12"

#case 7-1-2 ,model 13
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_2\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_2\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_13"

#case 8-0-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_0_3\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_0_3\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_15"

# #case 8-1-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_16"

# #case 8-1-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# .dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_17"


# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_10_0_3\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_18"

#case 9-0-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_9_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_9_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_19"

# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_9_0_1\\temp_aug\\chan_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_20"

#case 11-0-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_21"

#case 11-4-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
#
# case="model_22"

#case 11-4-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"

# case="model_23"

#case 11-4-1-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_24"

#case 12-4-3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_3_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_3_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_25"

# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_3_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_26"


#case 12-4-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_27"

#case 12-4-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_1_2\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_1_2\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_28"


#case 14-0-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\temp_aug\\no_norm\\dannV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\temp_aug\\no_norm\\dannV1\dataset_B\model\\predict_folder"
# case="model_29"

#case 14-0-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\no_aug\\no_norm\\dannV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\temp_aug\\no_norm\\dannV1\dataset_B\model\\predict_folder"
# case="model_30"

# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\temp_aug\\no_norm\\dannV1\dataset_B\model\\predict_folder"
# case="model_31"

#case 14-3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"
# case="model_32"

#case 14-3-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_3\\sub\\no_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_3\\sub\\no_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"
# case="model_33"

#case 14-3-1 case 14-3-1 ensemble
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\ensemble_results\\case1\\dataset_A"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\ensemble_results\\case1\\dataset_B"
# case="model_34"

#case 12-3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_3_1\\sub\\temp_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_3_1\\sub\\temp_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"
# case="model_35"

#case 12-3-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_3_3\\sub\\no_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_3_3\\sub\\no_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"
# case="model_36"


# #case 14-3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\addaV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\addaV1\dataset_B\model\\predict_folder"
# case="model_37"

#case 14-3-1 case 14-3-1 ensemble for addaV1 no norm, no aug
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\ensemble_results\\case2\\dataset_A"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\ensemble_results\\case2\\dataset_B"
# case="model_38"

# #case 14-3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\SRDA\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\SRDA\dataset_B\model\\predict_folder"
# case="model_39"


# # #case 14-3-1
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub_20\\no_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub_20\\no_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"
# case="model_40"

# #case 14-3-1
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub_30\\no_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub_30\\no_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"

case="model_41"

file = "pred_MI_label.txt"


# np.savetxt("util/pred_MI_label.txt",MI_label,delimiter=',',fmt="%d")
pred_A = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')
final_pred = np.concatenate([pred_A,pred_B])
path = os.path.join("util",case,"answer.txt")
np.savetxt(path,final_pred,delimiter=',',fmt="%d")
