import numpy as np
import pandas as pd
import os
file = "pred_MI_label.txt"

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

def comapre(pred_1,pred_2,dataset_1="dataset_1",dataset_2="dataset_2"):
    total = len(pred_1)

    # count_list= [0]*6
    # percent_list = [0.0]*6

    count(pred_1,dataset_1)
    count(pred_2,dataset_2)

    similar_count_state = "{} and {} has similar : ".format(dataset_1,dataset_2)
    similar_percent_state = "{} and {} has similar : ".format(dataset_1,dataset_2)
    total_sim = 0
    for i in range(6):
        # pred_1_count = len(np.where(pred_1 == i)[0])
        # pred_2_count = len(np.where(pred_2 == i)[0])

        # pred_1_loc = np.where(pred_1 == i)
        # pred_2_loc = np.where(pred_1 == i)
        temp_pred_1 = np.where(pred_1 == i, 1, -1)
        temp_pred_2 = np.where(pred_2 == i, 1, -2)


        count_similar = np.sum(temp_pred_1 == temp_pred_2)
        total_sim += count_similar
        count_percent = round(count_similar/total * 100,2)

        similar_count_state+= " {} label {}".format(count_similar,i)
        similar_percent_state += " {}% label {}".format(count_percent,i)
        # count_list[i] = count_similar
        # percent_list[i] = count_percent

    print(similar_count_state)
    print(similar_percent_state)
    print("total similar {}% ".format(round(total_sim/total*100,2)))



# case_1_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_1\\quick_ver_1_1\\tune_cla\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# pred_1 = np.loadtxt(os.path.join(case_1_path, file), delimiter=',')
# count(pred_1, "case 1")
#
# case_2_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_1\\quick_ver_1_1\\reproduce\\tune_cla\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"
# pred_2 = np.loadtxt(os.path.join(case_2_path, file), delimiter=',')
# count(pred_2, "case 2")

# case_2_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_1\\task_1_final_1\\quick_ver_1_1\\al_pretrain\\2\\no_aug\\no_norm\\deepsleep_vanilla\\full_dataset\\model\\predict_folder"


case_1_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\tools\\util\\task_1_final_13"
file = "answer.txt"
pred_1 = np.loadtxt(os.path.join(case_1_path, file), delimiter=',')
count(pred_1, "case 1")

case_2_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\tools\\vernon_confirm\\task_1"
# file = "answer.txt"
file = "answer-v2.txt"
pred_2 = np.loadtxt(os.path.join(case_2_path, file), delimiter=',')
count(pred_2, "case 2")


#compare similar
# count_similar = np.sum(pred_1== pred_2)
# count_percent =
# print("count simibar  : ",count_similar)

comapre(pred_1,pred_2)

# count_similar = np.sum(pred_B_1== pred_B_2)
# print("count simibar B : ",count_similar)
#

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
