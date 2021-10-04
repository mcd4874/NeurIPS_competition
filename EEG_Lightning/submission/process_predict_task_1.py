import numpy as np
import pandas as pd
import os

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


# model_path = "task_1\\tune_cla"
model_path = "/home/vernon/software/NeurIPS_competition/EEG_Lightning/submission/task_1/tune_cla_2"
result_path = model_path+"/deepsleep_vanilla/full_dataset/model/predict_folder"

case="task_1"

file = "pred_MI_label.txt"


pred = np.loadtxt(os.path.join(result_path, file), delimiter=',')

answer_folder = os.path.join("util",case)
if not os.path.exists(answer_folder):
    os.makedirs(answer_folder)

path = os.path.join(answer_folder,"answer.txt")
np.savetxt(path,pred,delimiter=',',fmt="%d")


count(pred,"current")

