import numpy as np
import pandas as pd
import os
file = "pred_MI_label.txt"

def count(label,name=""):
    count_0 = len(np.where(label == 0)[0])
    count_1 = len(np.where(label == 1)[0])
    count_2 = len(np.where(label == 2)[0])
    count_3 = len(np.where(label == 3)[0])

    print("dataset {},  has {} label 0, {} label 1, and {} label 2, {} label 3".format(name,count_0,count_1,count_2,count_3))
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
#
# def count(label,name=""):
#     count_0 = len(np.where(label == 0)[0])
#     count_1 = len(np.where(label == 1)[0])
#     count_2 = len(np.where(label == 2)[0])
#     print("dataset {},  has {} label 0, {} label 1, and {} label 2".format(name,count_0,count_1,count_2))
#
# def load_A_B(list_A_path,list_B_path):
#     pred_A = list()
#     idx=0
#     for A_path in list_A_path:
#         pred = np.loadtxt(os.path.join(A_path, file), delimiter=',')
#         count(pred, "A_{}".format(idx))
#         pred_A.append(pred)
#         idx+=1
#     pred_A = np.concatenate(pred_A)
#
#     pred_B = list()
#     idx=0
#     for B_path in list_B_path:
#         pred = np.loadtxt(os.path.join(B_path, file), delimiter=',')
#         count(pred, "B_{}".format(idx))
#         pred_B.append(pred)
#         idx+=1
#     pred_B = np.concatenate(pred_B)
#
#     count(pred_A,"dataset_A")
#     count(pred_B,"dataset_B")
#
#     return pred_A,pred_B
#
#
# dataset_A_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\5\\no_aug\\no_norm\\mcdV1\dataset_A_0\model\\predict_folder"
# dataset_A_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\5\\no_aug\\no_norm\\mcdV1\dataset_A_1\model\\predict_folder"
# dataset_A_2_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\5\\no_aug\\no_norm\\mcdV1\dataset_A_2\model\\predict_folder"
#
# dataset_B_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\5\\no_aug\\no_norm\\mcdV1\dataset_B_0\model\\predict_folder"
# dataset_B_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\5\\no_aug\\no_norm\\mcdV1\dataset_B_1\model\\predict_folder"
#
#
#
# pred_A_1,pred_B_1 = load_A_B([dataset_A_0_result_path,dataset_A_1_result_path,dataset_A_2_result_path],[dataset_B_0_result_path,dataset_B_1_result_path])
#
#
# # dataset_A_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\tune_n_step\\1\\no_aug\\no_norm\\mcdV1\dataset_A_0\model\\predict_folder"
# # dataset_A_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\tune_n_step\\1\\no_aug\\no_norm\\mcdV1\dataset_A_1\model\\predict_folder"
# # dataset_A_2_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\tune_n_step\\1\\no_aug\\no_norm\\mcdV1\dataset_A_2\model\\predict_folder"
# #
# #
# # dataset_B_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\tune_n_step\\1\\no_aug\\no_norm\\mcdV1\dataset_B_0\model\\predict_folder"
# # dataset_B_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\tune_n_step\\1\\no_aug\\no_norm\\mcdV1\dataset_B_1\model\\predict_folder"
#
# # dataset_A_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_n_step\\2\\no_aug\\no_norm\\mcdV1\dataset_A_0\model\\predict_folder"
# # dataset_A_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_n_step\\2\\no_aug\\no_norm\\mcdV1\dataset_A_1\model\\predict_folder"
# # dataset_A_2_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_n_step\\2\\no_aug\\no_norm\\mcdV1\dataset_A_2\model\\predict_folder"
# #
# #
# # dataset_B_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_n_step\\2\\no_aug\\no_norm\\mcdV1\dataset_B_0\model\\predict_folder"
# # dataset_B_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_n_step\\2\\no_aug\\no_norm\\mcdV1\dataset_B_1\model\\predict_folder"
#
#
# # dataset_A_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\4\\no_aug\\no_norm\\mcdV1\dataset_A_0\model\\predict_folder"
# # dataset_A_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\4\\no_aug\\no_norm\\mcdV1\dataset_A_1\model\\predict_folder"
# # dataset_A_2_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\4\\no_aug\\no_norm\\mcdV1\dataset_A_2\model\\predict_folder"
# #
# #
# # dataset_B_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\4\\no_aug\\no_norm\\mcdV1\dataset_B_0\model\\predict_folder"
# # dataset_B_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Dassl_Lightning\\NeurIPS_competition\\task_2_final_2\\LA_EA\\tune_filter\\4\\no_aug\\no_norm\\mcdV1\dataset_B_1\model\\predict_folder"
#
# # exp = "task_2_final_2\\best\\1\\al_pretrain\\3\\"
# exp = ""
# dataset_A_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\NeurIPS_2\\{}no_aug\\no_norm\\mcdV1\dataset_A_0\model\\predict_folder".format(exp)
# dataset_A_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\NeurIPS_2\\{}no_aug\\no_norm\\mcdV1\dataset_A_1\model\\predict_folder".format(exp)
# dataset_A_2_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\NeurIPS_2\\{}no_aug\\no_norm\\mcdV1\dataset_A_2\model\\predict_folder".format(exp)
#
# dataset_B_0_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\NeurIPS_2\\{}no_aug\\no_norm\\mcdV1\dataset_B_0\model\\predict_folder".format(exp)
# dataset_B_1_result_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\NeurIPS_2\\{}no_aug\\no_norm\\mcdV1\dataset_B_1\model\\predict_folder".format(exp)
#
#
# pred_A_2,pred_B_2 = load_A_B([dataset_A_0_result_path,dataset_A_1_result_path,dataset_A_2_result_path],[dataset_B_0_result_path,dataset_B_1_result_path])
#
#
# #compare similar
# count_similar = np.sum(pred_A_1== pred_A_2)
# print("count simibar A : ",count_similar)
#
# count_similar = np.sum(pred_B_1== pred_B_2)
# print("count simibar B : ",count_similar)
#
#
# def combine_result(best_pred,list_pred):
#     pred_output =list()
#     for trial_idx in range(len(best_pred)):
#         labels = [0,0,0]
#         best_default_pred = int(best_pred[trial_idx])
#         # print("best : ",best_default_pred)
#         labels[int(best_default_pred)] = labels[int(best_default_pred)]+1
#         for pred in list_pred:
#             current_pred=int(pred[trial_idx])
#             labels[int(current_pred)] = labels[int(current_pred)] + 1
#         if labels[0]==labels[1] and labels[0]==labels[2] :
#             pred_output.append(best_default_pred)
#         else:
#             final_label = np.argmax(labels)
#             pred_output.append(final_label)
#     return np.array(pred_output)

# new_A = combine_result(pred_A_14,[pred_A_15])
# count(new_A,name="new_A")
# new_B = combine_result(pred_B_14,[pred_B_15])
# count(new_B,name="new_B")
# count_similar = np.sum(pred_A_14 == new_A)
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


# case_1_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\tools\\vernon_task_2\\drop_0.5"
# case_1_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\tools\\vernon_task_2\\base_setup"
case_1_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\tools\\vernon_task_2\\new_base_setup"

file = "answer.txt"
pred_1 = np.loadtxt(os.path.join(case_1_path, file), delimiter=',')
count(pred_1, "case 1")

# case_2_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\tools\\util\\task_2_final_22"
# case_2_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\tools\\vernon_task_2\\drop_0.5"
case_2_path = "C:\\wduong_folder\Dassl.pytorch-master\\NeurIPS_competition\\EEG_Lightning\\tools\\vernon_task_2\\base_setup"

file = "answer.txt"
pred_2 = np.loadtxt(os.path.join(case_2_path, file), delimiter=',')
count(pred_2, "case 2")

#compare similar
# count_similar = np.sum(pred_1== pred_2)
# count_percent =
# print("count simibar  : ",count_similar)

comapre(pred_1,pred_2)