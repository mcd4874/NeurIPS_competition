import numpy as np
import pandas as pd
import os
file = "pred_MI_label.txt"

# # case 2
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4\\temp_aug\chan_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case2"
pred_A_0 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_0 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')


#case 4-5 ,model 8
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\temp_aug\chan_norm\\adaptation\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\temp_aug\chan_norm\\adaptation\dataset_B\model\\predict_folder"
# case="case4_5"
pred_A_1 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_1 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 4-3 ,model 9
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_3\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_3\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case4_3"

pred_A_2 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_2 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 4-3-1 ,model 10
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_3_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_3_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case4_3"

pred_A_2_1 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_2_1 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')


#case 4-5 ,model 10
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_10"

pred_A_3 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_3 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 7-1 ,model 11
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_11"

pred_A_4 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_4 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')


# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7\\temp_aug\chan_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7\\temp_aug\chan_norm\\adaptationV1\dataset_B\model\\predict_folder"

dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="case2"

pred_A_4_0 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_4_0 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_4_5\\temp_aug\\chan_norm\\adaptationV1\dataset_B\model\\predict_folder"
# pred_B_4 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

def count(label,name=""):
    count_0 = len(np.where(label == 0)[0])
    count_1 = len(np.where(label == 1)[0])
    count_2 = len(np.where(label == 2)[0])
    print("dataset {},  has {} label 0, {} label 1, and {} label 2".format(name,count_0,count_1,count_2))


#compare similar
count_similar = np.sum(pred_A_2 == pred_A_3)
print("count simibar A : ",count_similar)
count(pred_A_2,name="A_2")
count(pred_A_3,name="A_3")

count_similar = np.sum(pred_A_2 == pred_A_1)
print("count simibar A : ",count_similar)
count(pred_A_2,name="A_2")
count(pred_A_1,name="A_1")

count_similar = np.sum(pred_A_2 == pred_A_0)
print("count simibar A : ",count_similar)
count(pred_A_2,name="A_2")
count(pred_A_0,name="A_0")

count_similar = np.sum(pred_B_2 == pred_B_3)
print("count simibar B : ",count_similar)
count(pred_B_2,name="B_2")
count(pred_B_3,name="B_3")

count_similar = np.sum(pred_B_2 == pred_B_1)
print("count simibar B : ",count_similar)
count(pred_B_2,name="B_2")
count(pred_B_1,name="B_1")

count_similar = np.sum(pred_B_2 == pred_B_0)
print("count simibar B : ",count_similar)
count(pred_B_2,name="B_2")
count(pred_B_0,name="B_0")


count_similar = np.sum(pred_B_2 == pred_B_4)
print("count simibar B : ",count_similar)
count(pred_B_4,name="B_4")


count_similar = np.sum(pred_A_2 == pred_A_2_1)
print("count simibar A : ",count_similar)
count(pred_A_2,name="A_2")
count(pred_A_2_1,name="A_2_1")

count_similar = np.sum(pred_B_2 == pred_B_2_1)
print("count simibar B : ",count_similar)
count(pred_B_2,name="B_2")
count(pred_B_2_1,name="B_2_1")



count_similar = np.sum(pred_A_2 == pred_A_4)
print("count simibar A : ",count_similar)
count(pred_A_2,name="A_2")
count(pred_A_4,name="A_4")

count_similar = np.sum(pred_B_2 == pred_B_4)
print("count simibar B : ",count_similar)
count(pred_B_2,name="B_2")
count(pred_B_4,name="B_4")

count_similar = np.sum(pred_A_4_0 == pred_A_4)
print("count simibar A : ",count_similar)
count(pred_A_4_0,name="A_4_0")
count(pred_A_4,name="A_4")

count_similar = np.sum(pred_B_4_0 == pred_B_4)
print("count simibar B : ",count_similar)
count(pred_B_4_0,name="B_4_0")
count(pred_B_4,name="B_4")