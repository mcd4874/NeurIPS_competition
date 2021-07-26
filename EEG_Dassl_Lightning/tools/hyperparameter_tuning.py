import optuna, joblib
from optuna.trial import Trial, FrozenTrial
from dassl.engine import build_trainer
from dassl.utils import generate_path_for_multi_sub_model
import os
import json
import numpy as np
import logging
import sys
from yacs.config import CfgNode

def generate_json_hpo():
    params = {
        "lr": ["log_uniform",0.0001,0.01],
        "wd": ["category", [0.001,0.01]],
        "batch_size": ["category",[32,64]]
    }
    extra_fix_trials = [
        {
            "lr":0.001,
            "wd":0.01,
            "batch_size":64
        }
    ]

    convert_map = {
        "lr": "OPTIM.LR",
        "wd": "OPTIM.WEIGHT_DECAY",
        "optimizer": "OPTIM.NAME",
        "batch_size": "DATALOADER.TRAIN_X.BATCH_SIZE"
    }
    trials = 50
    number_of_random_points = 10

    output = {
        "params":params,
        "extra_fix_trials":extra_fix_trials,
        "convert_yaml_map":convert_map,
        "trial":trials,
        "number_of_random_points":number_of_random_points
    }

    file_name = "tune_params.json"
    # file_path = os.path.join(, file_name)
    with open(file_name, "w") as outfile:
        json.dump(output, outfile,indent=4)

# generate_json_hpo()

class HPO:
    def __init__(self,config:CfgNode,param_file:str,build_trainer_func=None,objective=None):

        self.build_trainer_func = build_trainer if build_trainer_func is None else build_trainer_func
        self.config = config
        self.param_file = param_file
        self.set_up_params()
        self.check_params() # check if we have correct list of params
        self.objective = self.build_objective() if objective is None else objective

    def check_params(self):
        list_params = list(self._params.keys())
        list_possible_params = list(self._convert_yaml_map.keys())
        assert all(x in list_possible_params for x in list_params)

    def set_up_params(self):
        outfile = open( self.param_file, "r")
        params_json = json.load(outfile)
        print("params raw json : {}".format(params_json))
        print("json keys : ",params_json.keys())
        self._params = params_json["params"]
        self._convert_yaml_map = params_json["convert_yaml_map"]
        self._n_trials = params_json["trial"]
        if "extra_fix_trials" in list(params_json.keys()):
            print("set up extra trials")
            self.extra_fix_trials = params_json["extra_fix_trials"]
        else:
            self.extra_fix_trials = None
        if "number_of_random_points" in list(params_json.keys()):
            print("set up random start ")
            self.number_of_random_points = params_json["number_of_random_points"]
        else:
            self.number_of_random_points = 10


    def build_objective(self,cross_fold = True):

        def update_config(trial,config):
            for param,value in self._params.items():
                match_yaml_string = self._convert_yaml_map[param]
                suggest_type = value[0]
                update_param = None
                print("current params : {} match -> {}".format(param,value))
                if suggest_type == "category":
                    update_param = trial.suggest_categorical(param, value[1])
                elif suggest_type == "log_uniform":
                    lower_bound = value[1]
                    upper_bound  = value[2]
                    update_param = trial.suggest_loguniform(param, lower_bound, upper_bound)
                if update_param is not None:
                    config.merge_from_list([match_yaml_string, update_param])
                else:
                    print("there are problem with params update")


        def cross_fold_objective(trial,config):
            """
            assume that we conduct cross test for test fold va valid fold
            current use for increment fold test
            """
            update_config(trial,config)
            START_TEST_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.START_TEST_FOLD
            END_TEST_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.END_TEST_FOLD
            START_VALID_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.START_VALID_FOLD
            END_VALID_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.END_VALID_FOLD

            val_scores = []
            for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
                # print("current test fold : ", current_test_fold)
                config.DATASET['VALID_FOLD_TEST'] = current_test_fold
                # for current_increment_fold in range(1, INCREMENT_FOLDS + 1):
                for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
                    config.DATASET['VALID_FOLD'] = current_valid_fold
                    tune_trainer = build_trainer(config)
                    # print("current test fold : ",current_test_fold)
                    best_val_loss = tune_trainer.param_tuning(trial)
                    val_scores.append(best_val_loss)
            output = np.sum(val_scores)/len(val_scores)
            return output
        if cross_fold:
            return cross_fold_objective
        return None

    def load_hpo_into_config(self,config=None,hpo_params = None):
        if config is None:
            config = self.config.clone()
        if hpo_params is None:
            output_dir = config.OUTPUT_DIR
            tune_param_path = self.generate_tune_model_folder(config=config)
            hpo_params = self.load_tune_params(tune_param_path)
        # print("update hpo params _: ",hpo_params)
        result_config = self.merge_config_with_tune_params(hpo_params,config)
        return result_config

    def run_hpo(self,save_study_func=None):
        """
        assume that we are currently running an incremental experiment option only
        """
        # conduct incremental subject experiments
        INCREMENT_FOLDS = self.config.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.NUM_INCREMENT_FOLDS
        START_INCREMENT_FOLD = self.config.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.START_INCREMENT_FOLD
        END_INCREMENT_FOLD = self.config.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL.END_INCREMENT_FOLD


        # extra_fix_trial = generate_extra_fix_trial()
        # add_extra = True

        for current_increment_fold in range(START_INCREMENT_FOLD, END_INCREMENT_FOLD + 1):
            config = self.config.clone()
            config.TRAIN_EVAL_PROCEDURE.TRAIN.INCREMENTAL['CURRENT_FOLD'] = current_increment_fold
            seed = config.DATASET.EXTRA_CONFIG.SHUFFLE_SEED
            number_of_random_points = self.number_of_random_points  # random searches to start opt process
            # study_name = "study_case_DBS/example-study"  # Unique identifier of the study.
            study_name = "example-study"
            # # storage_name = "sqlite:///{}.db".format(study_name)
            # study_name = "study_case_DBS/example-study"  # Unique identifier of the study.
            # storage_name = "mysql://root@localhost/example"
            # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed,multivariate=True),
            #                             pruner=optuna.pruners.NopPruner(), study_name=study_name,storage=storage_name,load_if_exists=True)
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed,multivariate=True),
                                        pruner=optuna.pruners.NopPruner(), study_name=study_name)
            # study.set_user_attr('cfg', config)
            # if add_extra :
            #     study.enqueue_trial(extra_fix_trial)
            # study.get_trials()
            if self.extra_fix_trials:
                for extra_fix_trial in self.extra_fix_trials:
                    study.enqueue_trial(extra_fix_trial)
                    print("enqueue extra trials : {}".format(extra_fix_trial))
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            study.optimize(lambda trial: self.objective(trial, config), n_trials=self._n_trials)

            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            # save study
            if not save_study_func:
                START_TEST_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.START_TEST_FOLD
                END_TEST_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.END_TEST_FOLD
                START_VALID_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.START_VALID_FOLD
                END_VALID_FOLD = config.TRAIN_EVAL_PROCEDURE.TRAIN.END_VALID_FOLD
                for current_test_fold in range(START_TEST_FOLD, END_TEST_FOLD + 1):
                    # print("current test fold : ", current_test_fold)
                    config.DATASET['VALID_FOLD_TEST'] = current_test_fold
                    # for current_increment_fold in range(1, INCREMENT_FOLDS + 1):
                    for current_valid_fold in range(START_VALID_FOLD, END_VALID_FOLD + 1):
                        # print("current valid fold : ",current_valid_fold)
                        config.DATASET['VALID_FOLD'] = current_valid_fold
                        tune_path = self.generate_tune_model_folder(config)
                        joblib.dump(study, os.path.join(tune_path, 'optuna_study.pkl'))
                        # save params
                        # print("current tune path : ",tune_path)
                        self.save_best_tune_params(trial, tune_path)
            else:
                tune_path = self.generate_tune_model_folder(config)
                save_study_func(trial, tune_path)


    def generate_tune_model_folder(self,config = None,folder_name = 'auto_tuning_params'):
        # generate folder to deal with train-k-folds
        if config is None:
            config = self.config
        current_path = os.path.join(config.OUTPUT_DIR,folder_name)
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        current_path = generate_path_for_multi_sub_model(config,current_path)
        return current_path

    def load_tune_params(self,output_dir=None,file_name='best_model_params.json'):
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
            print("use default output dir : {}".format(output_dir))
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            result_list = []
            # print(file_path)
            with open(file_path, "r") as outfile:
                params = json.load(outfile)
                possible_params = list(self._convert_yaml_map.keys())
                for key,value in params.items():
                    if key in possible_params:
                        result_list.append([self._convert_yaml_map[key],params[key]])
            print("load available auto tune hyper-parameter from path {}".format(file_path))
            return result_list

        print("No available auto tune params at {}".format(file_path))
        return None
    def merge_config_with_tune_params(self,tune_params_list,config = None):
        if config is None:
            config = self.config.clone()
        if tune_params_list is None:
            print("There are no available tune params to merge with config file")
            return config
        for param in tune_params_list:
            config.merge_from_list(param)
        return config


    def load_study(self,output_dir = None,file_name='optuna_study.pkl'):
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
            print("use default output dir : {}".format(output_dir))
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            study = joblib.load(file_path)
            return study
        else:
            print("No available auto tune params study exist")
            return None

    def save_best_tune_params(self,trial: FrozenTrial,tune_path:str,file_name='best_model_params.json'):
        file_path = os.path.join(tune_path,file_name)
        with open(file_path, "w") as outfile:
            json.dump(trial.params, outfile,indent=4)







