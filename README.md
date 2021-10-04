
# Introduction
This repository contains the PyTorch code to reproduce our submission to the [NeurIPS 2021 BEETL Competition](https://beetl.ai/). 

This repository uses PyTorch + PyTorch Lightning as the training framework. We have made every attempt to ensure reproducible results (for example, using Pytorch-Lightning's ```seed_everything```). While we were able to reproduce results on the same machine, we observed slight differences in final submission accuracies (~0.5%) across different machines. Please see [this page](https://pytorch.org/docs/stable/notes/randomness.html) for more info regarding PyTorch and reproducibility.

# Requirements

We use [Anaconda](https://www.anaconda.com/products/individual) as our Python distribution and Linux as our OS. We also assume that the user has an NVIDIA GPU with >= 4GB memory, preferably a recent RTX-series GPU. 

First, create a new conda environment with 
```
conda create --name beetl python=3.8 anaconda=2021.05
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda activate beetl
```
Then, install the remaining dependencies:
```
pip install -r requirements.txt
``` 

# Running the code

The code to run our experiments can be found in ```EEG_Lightning/submission```. 

## Task 1
1. For Task 1, edit [the following lines](https://github.com/mcd4874/NeurIPS_competition/blob/main/EEG_Lightning/submission/generate_task_1_data.py#L88-L90) of ```generate_task_1_data.py``` with the path to the SleepSource and FinalSleep directories, respectively.
2.  In a new terminal, go to ```EEG_Lightning/submission/task_1_script``` and edit [the following line](https://github.com/mcd4874/NeurIPS_competition/blob/main/EEG_Lightning/submission/task_1_script/task_1_script.sh#L8) of ```task_1_script.sh``` with the path to this repository on your system.
3. Activate the ```beetl``` conda environment, and train the model with ```bash task_1_script.sh``` Training time is approximately 1.5-2 hours on an NVIDIA RTX 2080 Ti.
4. Go to ```EEG_Lightning/submission```, edit the [following line](https://github.com/mcd4874/NeurIPS_competition/blob/main/EEG_Lightning/submission/process_predict_task_1.py#L26) of ```process_predict_task_1.py``` with the path to ```/submission/task_1``` and run the script via ```python process_predict_task_1.py```.
5. The prediction file will be found at ```EEG_Lightning/submission/util/task_1/answer.txt```

## Task 2
1. For Task 2, edit [the following line](https://github.com/mcd4874/NeurIPS_competition/blob/main/EEG_Lightning/submission/generate_task_2_data.py#L49) of ```generate_task_2_data.py``` with the path to the FinalMI directory. The rest of the datasets (BCI IV 2A, Cho2017, Physionet) are assumed to be in ```~/mne_data``` as they have been downloaded using MOABB/BEETL tools.
2. In a new terminal, go to ```EEG_Lightning/submission/task_2_script``` and edit [the following line](https://github.com/mcd4874/NeurIPS_competition/blob/main/EEG_Lightning/submission/task_2_script/task_2_script.sh#L8) with the path to this repository on your system.
3. Activate the ```beetl``` conda environment, and train the model with ```bash task_2_script.sh```. Training time is approximately 45 min - 1 hour on an NVIDIA RTX 2080 Ti. 
4. Go to ```EEG_Lightning/submission```, edit the [following line](https://github.com/mcd4874/NeurIPS_competition/blob/main/EEG_Lightning/submission/process_predict_task_2.py#L9) of ```process_predict_task_2.py``` with the path to ```/submission/task_2``` and run the script via ```python process_predict_task_2.py```.
5. The prediction file will be found at ```EEG_Lightning/submission/util/task_1/answer.txt```

