import numpy as np
from datasets import load_dataset, get_dataset_config_names
import os
import gzip
import json

# obtain the list of all monash datasets
monash_datasets = get_dataset_config_names("monash_tsf")

# create a directory to save the data
data_root_dir = "time_series/monash"
if not os.path.exists(data_root_dir):
    os.makedirs(data_root_dir)

for dataset_name in monash_datasets:
    try:
        dataset_name = 'nn5_weekly'
        # load the dataset
        monash_dataset = load_dataset("monash_tsf", dataset_name)

        # skip multivariate datasets
        sample_target = monash_dataset['train'][0]['target']
        if len(np.array(sample_target).shape) > 1:
            print(f"Skipping {dataset_name} because it is multivariate")
            continue

        # obtain the lengths of the train, test and validation sets
        train_length = len(monash_dataset['train'][0]['target'])
        test_length = len(monash_dataset['test'][0]['target'])
        validation_length = len(monash_dataset['validation'][0]['target']) if 'validation' in monash_dataset else int((test_length + train_length) / 2)

        # calculate the prediction length
        prediction_length = validation_length - train_length

        print(f"Processing dataset: {dataset_name}")
        print(f"Prediction length: {prediction_length}")

        # process the test set
        test_arrays = []
        sequence_lengths = []
        for sample in monash_dataset['test']:
            target = np.array(sample['target'])
            if np.isnan(target).any():
                print(f"Skipping sample in test of {dataset_name} due to NaNs")
                continue

            seq_length = len(target)
            sequence_lengths.append(seq_length)
            time_idx = np.arange(seq_length)

            # append the time index to the target
            sample_array = np.column_stack((time_idx, target))
            test_arrays.append(sample_array)

        # find the maximum sequence length
        max_seq_length = max(sequence_lengths)
        num_samples = len(test_arrays)
        num_features = test_arrays[0].shape[1]

        # pad the arrays
        padded_arrays = np.full((num_samples, max_seq_length, num_features), np.nan)

        for i, sample in enumerate(test_arrays):
            seq_length = sample.shape[0]
            padded_arrays[i, :seq_length, :] = sample

        # generate split_indeces
        from sklearn.model_selection import train_test_split

        # randomly shuffle the samples
        sample_indeces = np.arange(num_samples)

        # generate the split indeces
        split_indeces = [{}]

        np.random.shuffle(sample_indeces)

        split_indeces[0] = {
            'train': sample_indeces,
            'val': sample_indeces,
            'test': sample_indeces
        }
        split_indeces = np.array(split_indeces, dtype=object)

        data_save_dir = os.path.join(data_root_dir, dataset_name)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
        else:
            print(f"{dataset_name} already exists. Skipping...")
            continue


        # save X_array as .npy.gz file
        data_save_path = os.path.join(data_save_dir, f"X.npy.gz")
        with gzip.GzipFile(data_save_path, "w") as f:
            np.save(f, padded_arrays)

        # save y_array as .npy.gz file
        y_array = np.array([0] * num_samples)

        y_save_path = os.path.join(data_save_dir, f"y.npy.gz")
        with gzip.GzipFile(y_save_path, "w") as f:
            np.save(f, y_array)

        # save split_indeces as .npy.gz file
        split_save_path = os.path.join(data_save_dir, f"split_indeces.npy.gz")
        with gzip.GzipFile(split_save_path, "wb") as f:
            np.save(f, split_indeces)

        # generate metadata
        dataset_info = {
            'name': dataset_name,
            'time_idx': 0,
            'cat_idx': [],
            'target_type': 'forecast',
            'pred_len': prediction_length,
        }

        # save metadata as .json file
        json_save_path = os.path.join(data_save_dir, f"metadata.json")
        with open(json_save_path, 'w') as json_file:
            json.dump(dataset_info, json_file, indent=4)

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        continue
