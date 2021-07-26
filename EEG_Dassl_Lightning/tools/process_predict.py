import numpy as np
import pandas as pd
import os

dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_3\\temp_aug\chan_norm\\adaptation\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_3\\temp_aug\chan_norm\\adaptation\dataset_B\model\\predict_folder"
file = "pred_MI_label.txt"


# np.savetxt("util/pred_MI_label.txt",MI_label,delimiter=',',fmt="%d")
pred_A = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')
final_pred = np.concatenate([pred_A,pred_B])
np.savetxt("util/answer.txt",final_pred,delimiter=',',fmt="%d")
