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


#case 7-1-1 ,model 12
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_12"

pred_A_5 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_5 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\dannV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\dannV1\dataset_B\model\\predict_folder"
# case="model_12"

pred_A_5_1 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_5_1 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\shallowcon_adaptV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_1\\temp_aug\\no_norm\\shallowcon_adaptV1\dataset_B\model\\predict_folder"
# case="model_12"

pred_A_5_2 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_5_2 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 7-1-2
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_2\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_7_1_2\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_13"

pred_A_6 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_6 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 8-0-1
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_15"

pred_A_7 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_7 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 8-1-3
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_16"

# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\no_aug\\chan_norm\\dannV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_8_1_3\\no_aug\\chan_norm\\dannV1\dataset_B\model\\predict_folder"
# case="model_16"

pred_A_8 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_8 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 9-0-3
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_9_0_1\\no_aug\\chan_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_9_0_3\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_16"

pred_A_9 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_9 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 10-0-1
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_10_0_3\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_10_0_3\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"
# case="model_16"

pred_A_10 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_10 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 11-0-3
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1_1\\no_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_11_4_1_1\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"

# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\experiment_11_0_1\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\experiment_11_0_1\\temp_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"


pred_A_11 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_11 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')

#case 12-4-3
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_3_2\\temp_aug\\no_norm\\adaptationV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_12_4_3\\no_aug\\no_norm\\adaptationV1\dataset_B\model\\predict_folder"


pred_A_12 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_12 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')


# #case 14-0-3
# dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\no_aug\\no_norm\\dannV1\dataset_A\model\\predict_folder"
# dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_0_3\\temp_aug\\no_norm\\dannV1\dataset_B\model\\predict_folder"

#case 14-3-1
dataset_A_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\mcdV1\dataset_A\model\\predict_folder"
dataset_B_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\final_result_14_3_1\\sub\\no_aug\\no_norm\\mcdV1\dataset_B\model\\predict_folder"


pred_A_14 = np.loadtxt(os.path.join(dataset_A_result_path, file), delimiter=',')
pred_B_14 = np.loadtxt(os.path.join(dataset_B_result_path, file), delimiter=',')


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

count_similar = np.sum(pred_A_5 == pred_A_4)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_4,name="A_4")

count_similar = np.sum(pred_B_5 == pred_B_4)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_4,name="B_4")

count_similar = np.sum(pred_A_5 == pred_A_6)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_6,name="A_6")

count_similar = np.sum(pred_B_5 == pred_B_6)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_6,name="B_6")

count_similar = np.sum(pred_A_5 == pred_A_5_1)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_5_1,name="A_5_1")

count_similar = np.sum(pred_B_5 == pred_B_5_1)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_5_1,name="B_5_1")

count_similar = np.sum(pred_A_5 == pred_A_5_2)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_5_2,name="A_5_2")

count_similar = np.sum(pred_B_5 == pred_B_5_2)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_5_2,name="B_5_2")

count_similar = np.sum((pred_A_5 == pred_A_5_2) & (pred_A_5 == pred_A_5_1))
print("count simibar A : ",count_similar)
# count(pred_A_5,name="A_5")
# count(pred_A_5_2,name="A_5_2")

count_similar = np.sum((pred_B_5 == pred_B_5_2) & (pred_B_5 == pred_B_5_1))
print("count simibar B : ",count_similar)
# count(pred_B_5,name="B_5")
# count(pred_B_5_2,name="B_5_2")

def combine_result(best_pred,list_pred):
    pred_output =list()
    for trial_idx in range(len(best_pred)):
        labels = [0,0,0]
        best_default_pred = int(best_pred[trial_idx])
        # print("best : ",best_default_pred)
        labels[int(best_default_pred)] = labels[int(best_default_pred)]+1
        for pred in list_pred:
            current_pred=int(pred[trial_idx])
            labels[int(current_pred)] = labels[int(current_pred)] + 1
        if labels[0]==labels[1] and labels[0]==labels[2] :
            pred_output.append(best_default_pred)
        else:
            final_label = np.argmax(labels)
            pred_output.append(final_label)
    return np.array(pred_output)

# new_A = combine_result(pred_A_14,[pred_A_5])
# count(new_A,name="new_A")
# new_B = combine_result(pred_B_14,[pred_B_5])
# count(new_B,name="new_B")
# count_similar = np.sum(pred_A_5 == new_A)
# print("count simibar A : ",count_similar)
# count(pred_A_5,name="A_5")
# count(new_A,name="new_A")
#
# new_B = combine_result(pred_B_5,[pred_B_5_1,pred_B_5_2])
# # print(new_A)
# count_similar = np.sum(pred_B_5 == new_B)
# print("count simibar AB : ",count_similar)
# count(pred_B_5,name="B_5")
# count(new_B,name="new_B")
#
# case="model_14"
# final_pred = np.concatenate([new_A,new_B])
# path = os.path.join("util",case,"answer.txt")
# np.savetxt(path,final_pred,delimiter=',',fmt="%d")


count_similar = np.sum(pred_A_5 == pred_A_7)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_7,name="A_7")

count_similar = np.sum(pred_B_5 == pred_B_7)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_7,name="B_7")

count_similar = np.sum(pred_A_5 == pred_A_8)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_8,name="A_8")

count_similar = np.sum(pred_B_5 == pred_B_8)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_8,name="B_8")

count_similar = np.sum(pred_A_5 == pred_A_9)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_9,name="A_9")

count_similar = np.sum(pred_B_5 == pred_B_9)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_9,name="B_9")

count_similar = np.sum(pred_A_5 == pred_A_10)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_10,name="A_10")

count_similar = np.sum(pred_B_5 == pred_B_10)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_10,name="B_10")

count_similar = np.sum(pred_A_5 == pred_A_11)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_11,name="A_11")

count_similar = np.sum(pred_B_5 == pred_B_11)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_11,name="B_11")

count_similar = np.sum(pred_A_5 == pred_A_12)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_12,name="A_12")

count_similar = np.sum(pred_B_5 == pred_B_12)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_12,name="B_12")

count_similar = np.sum(pred_A_5 == pred_A_14)
print("count simibar A : ",count_similar)
count(pred_A_5,name="A_5")
count(pred_A_14,name="A_14")

count_similar = np.sum(pred_B_5 == pred_B_14)
print("count simibar B : ",count_similar)
count(pred_B_5,name="B_5")
count(pred_B_14,name="B_14")