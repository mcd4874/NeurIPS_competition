

def generate_matlab_files(target_dataset,source_datasets,save_path,file_name):
    target_dataset_name = list(target_dataset.keys())[0]
    target_dataset_data = target_dataset[target_dataset_name]


    source_list = list()
    for source_dataset_name,source_dataset_data in source_datasets.items():
        source = {
            "source_domain_data":source_dataset_data[0],
            "source_domain_label":source_dataset_data[1],
            "source_label_name_map":source_dataset_data[3],
            "dataset_name":source_dataset_name,
            "subject_id": source_dataset_data[2]
        }
        source_list.append(source)

    matlab_data = {
        "source_domain": source_list,
        "target_domain": {
            "target_domain_data": target_dataset_data[0],
            "target_domain_label": target_dataset_data[1],
            "target_label_name_map": target_dataset_data[3],
            "dataset_name":target_dataset_name,
            "subject_id":target_dataset_data[2]
        }
    }


    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    data_file = '{}_transfer_learning.mat'.format(file_name)
    data_file = join(save_path,data_file)
    text_file = 'target_source_data_record.json'
    text_file = join(save_path,text_file)


    import json


    dictionary = {'target_dataet': target_dataset_name, 'source_datasets': list(source_datasets.keys())}
    with open(text_file, "w") as outfile:
        json.dump(dictionary, outfile)
    from scipy.io import savemat
    savemat(data_file, matlab_data)
