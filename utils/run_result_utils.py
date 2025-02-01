import os

import numpy as np


def load_numpy_arrays_from_folder(folder_path, data_dict):
    for foldername in ['weights_grad', 'normalized_logits', 'test_pred', 'train_pred']:
        data_dir = folder_path / foldername
        if not data_dir.exists():
            continue
        sorted_filenames = os.listdir(os.path.join(folder_path, foldername))

        for filename in sorted_filenames:  # {... model id ...}.npy
            if filename.endswith('.npy'):
                sample_num = int(filename.split('_')[-2])
                model_id = filename.split('_')[-1][:16]
                filepath = os.path.join(folder_path, foldername, filename)
                data = np.load(filepath)
                if model_id not in data_dict[sample_num]:
                    print(f"{model_id=} does not exist in the dictionary")
                else:
                    data_dict[sample_num][model_id][foldername] = data

    for sample_num in data_dict.keys():
        test_labels_path = next((folder_path / 'test_labels').glob(f'test_labels_{sample_num}_*.npy'))
        data_dict[sample_num]['test_labels'] = np.load(test_labels_path)

    return data_dict


def create_dict_from_rows(rows, add_num_of_test_predictions=False):
    dictionary_of_lists = {}

    # Iterate through each tuple in the list
    for row in rows:
        key = row[0]
        value = {
            "train_loss": row[1],
            "train_loss_normalize": row[2],
            "train_acc": row[3],
            "test_loss": row[4],
            "test_acc": row[5],
            # "weight_norm": row[6],
        }

        if add_num_of_test_predictions:
            value["num_of_0_predictions"] = row[8]
            value["num_of_1_predictions"] = row[9]

        # Check if the key already exists in the dictionary
        if key in dictionary_of_lists:
            dictionary_of_lists[key][row[7]] = value
        else:
            # If the key doesn't exist, create a new list with the first value
            dictionary_of_lists[key] = {row[7]: value}
    return dictionary_of_lists


def get_stats_per_sample(acc_aux):
    # gets a list of tuples of the form (num_samples, model_acc) and crates a dictionary of the form {num_samples: [acc1, acc2, ...]}
    acc_stats = {}
    for num_samples, model_acc in acc_aux:
        if num_samples not in acc_stats:
            acc_stats[num_samples] = []
        acc_stats[num_samples].append(model_acc)

    # gets a dictionary of the form {num_samples: [acc1, acc2, ...]} and crates a dictionary of the form {num_samples: (mean, std)}
    acc_stats_mean_std = {}
    for num_samples, acc_list in acc_stats.items():
        acc_stats_mean_std[num_samples] = (np.mean(acc_list), np.std(acc_list))

    return acc_stats_mean_std


def combine_two_keys(comparison_variables, param1, param2, new_param=None):
    # Checks
    assert param1 in comparison_variables, f"{param1} not in comparison_variables"
    assert param2 in comparison_variables, f"{param2} not in comparison_variables"
    assert len(comparison_variables[param1]) == len(comparison_variables[param2])

    if new_param is None:
        new_param = param1 + '_' + param2

    # Unify the algorithm and initialization keys
    comparison_variables[new_param] = [(val1, val2) for val1, val2 in
                                       zip(comparison_variables[param1], comparison_variables[param2])]

    del comparison_variables[param1]
    del comparison_variables[param2]
