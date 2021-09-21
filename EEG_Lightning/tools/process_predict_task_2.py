import numpy as np
import pandas as pd
import os

model_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\no_aug\\no_norm\\mcdV1"

#task 2 final 2
dataset_A_0_result_path = model_path+"\\dataset_A_0\model\\predict_folder"
dataset_A_1_result_path = model_path+"\\dataset_A_1\model\\predict_folder"
dataset_A_2_result_path = model_path+"\\dataset_A_2\model\\predict_folder"
#
#
dataset_B_0_result_path = model_path+"\\dataset_B_0\model\\predict_folder"
dataset_B_1_result_path = model_path+"\\dataset_B_1\model\\predict_folder"
case="task_2"


file = "pred_MI_label.txt"



pred_A_0 = np.loadtxt(os.path.join(dataset_A_0_result_path, file), delimiter=',')
pred_A_1 = np.loadtxt(os.path.join(dataset_A_1_result_path, file), delimiter=',')
pred_A_2 = np.loadtxt(os.path.join(dataset_A_2_result_path, file), delimiter=',')


pred_B_0 = np.loadtxt(os.path.join(dataset_B_0_result_path, file), delimiter=',')
pred_B_1 = np.loadtxt(os.path.join(dataset_B_1_result_path, file), delimiter=',')

pred_A = np.concatenate([pred_A_0,pred_A_1,pred_A_2])
pred_B = np.concatenate([pred_B_0,pred_B_1])
final_pred = np.concatenate([pred_A,pred_B])
path = os.path.join("util",case,"answer.txt")
np.savetxt(path,final_pred,delimiter=',',fmt="%d")



def count(label,name=""):
    count_0 = len(np.where(label == 0)[0])
    count_1 = len(np.where(label == 1)[0])
    count_2 = len(np.where(label == 2)[0])

    print("dataset {},  has {} label 0, {} label 1, and {} label 2".format(name,count_0,count_1,count_2))


count(pred_A,"dataset A")
count(pred_B,"dataset B")
