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
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
case="model_12"

file = "pred_MI_label.txt"


# np.savetxt("util/pred_MI_label.txt",MI_label,delimiter=',',fmt="%d")
pred_A = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')
final_pred = np.concatenate([pred_A,pred_B])
path = os.path.join("util",case,"answer.txt")
np.savetxt(path,final_pred,delimiter=',',fmt="%d")
