import math
import os
import datetime
import pandas as pd 
from copy import deepcopy
from tabulate import tabulate
from framework.files import load_all_data_groups
from framework.eval import Evaluator
from pathlib import Path
import yaml
import logging
from typing import Tuple


def get_dataset_alg_lists(datasets, algs, df, logger: logging.Logger) -> Tuple[list, list]:
    all_datasets = df['dataset'].unique().tolist()
    if datasets is None:
        dataset_list = all_datasets
    else:
        all_groups = load_all_data_groups()
        if datasets in all_groups:
            dataset_list = all_groups[datasets]
        else:
            with open(datasets, 'r') as f:
                conf = yaml.safe_load(f)
            dataset_list = conf['datasets']

    all_algs = df['algorithm'].unique().tolist()
    if algs is None:
        alg_list = all_algs
    else:
        with open(algs, 'r') as f:
            conf = yaml.safe_load(f)
        alg_list = conf['algorithms']

    dataset_list = set(dataset_list)
    alg_list = set(alg_list)
    all_datasets = set(all_datasets)
    all_algs = set(all_algs)
    if len(dataset_list - all_datasets) > 0 or len(alg_list - all_algs) > 0:
        logger.warning('Missing datasets: {}'.format(dataset_list - all_datasets))
        logger.warning('Missing algorithms: {}'.format(alg_list - all_algs))
        raise RuntimeError('Missing datasets or algorithms')
    
    dataset_list = list(dataset_list)
    alg_list = list(alg_list)
    return dataset_list, alg_list


def delete_missing_folds(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    # Delete (dataset, setup) that cannot cover all folds
    old_size = len(df)
    fold_list = df['fold'].unique().tolist()
    fold_set = set(fold_list)
    groups = df.groupby(['dataset', 'setup'])['fold'].unique()
    valid_ds = groups[groups.apply(lambda x: set(x) == fold_set)].index
    df = df[df.set_index(['dataset', 'setup']).index.isin(valid_ds)]
    new_size = len(df)
    logger.warning(f"  ==> Deleting {old_size - new_size} items that do not cover all folds in {fold_list}")
    return df


def delete_missing_datasets(df: pd.DataFrame, dataset_list, logger) -> pd.DataFrame:
    # Delete setup that cannot cover all datasets
    groups = df.groupby('setup')['dataset'].unique()
    # print(dataset_list)
    valid_setup = groups[groups.apply(lambda x: set(dataset_list) <= set(x))].index  # Decide if dataset_list is a subset of x
    df0 = df[df['setup'].isin(valid_setup)]
    logger.info(f"Ignoring {len(df) - len(df0)} items that do not cover all {len(dataset_list)} datasets")
    return df0
        

def check_missing_datasets(df: pd.DataFrame, dataset_list, logger):
    # Check whether there is any dataset that does not appear at all
    missing_datasets = set(dataset_list) - set(df['dataset'])
    if len(missing_datasets) > 0:
        logger.error('The following datasets are missing:')
        logger.error(f'{missing_datasets}')


def best_each_dataset_alg(df: pd.DataFrame, alg_list, dataset_list, logger: logging.Logger, evaluator: Evaluator, result_dict: dict = None) -> dict:
    # For each (dataset, alg), select the best setup with the highest val performance
    # Returns: d[dataset][alg] = (setup, val_performance, test_performance, metric)
    best_setup = {}
    n_folds = df['fold'].nunique()
    for alg in alg_list:
        logger.info(f' ==> Algorithm: {alg}')
        n = 0
        val = 0
        test = 0
        val_var = 0
        test_var = 0
        for dataset in dataset_list:
            agg_dict = {f'{col}_mean': (col, 'mean') for col in df.columns if col.startswith('val') or col.startswith('test')}
            agg_dict = agg_dict | {f'{col}_std': (col, 'std') for col in df.columns if col.startswith('val') or col.startswith('test')}
            agg_dict['dataset_type'] = ('dataset_type', 'first')
            # logger.debug('agg_dict = {}'.format(agg_dict))
            df0 = df[(df['algorithm'] == alg) & (df['dataset'] == dataset)].groupby(['setup']).agg(**agg_dict).reset_index()
            # print('Algorithm:', alg, '  Dataset:', dataset, '  n_rows:', len(df0))
            # logger.debug('df0 = {}'.format(df0))
            if df0.empty:
                logger.error('Empty entries for algorithm {}, dataset {}'.format(alg, dataset))
                best_setup[dataset][alg] = ('None', -1e8, -1e8)
                val += -1e8
                test += -1e8
            else:
                target_type = df0.loc[0, 'dataset_type']
                selection_metric = evaluator.model_selection_metric(target_type)
                a = df0.loc[df0[f'val_{selection_metric}_mean'].idxmax()]
                n += 1
                val += a.val_performance_mean
                test += a.test_performance_mean
                val_var += a.val_performance_std ** 2 / n_folds
                test_var += a.test_performance_std ** 2 / n_folds
                if not dataset in best_setup:
                    best_setup[dataset] = {}
                best_setup[dataset][alg] = (a.setup, a.val_performance_mean, a.val_performance_std, a.test_performance_mean, a.test_performance_std)
        val = val / n 
        test = test / n 
        val_std = math.sqrt(val_var) / n   
        test_std = math.sqrt(test_var) / n 
        logger.warning(f'Avg val performance = {val}')
        logger.warning(f'Avg test performance = {test}') 
        logger.warning(f'Val std = {val_std}')
        logger.warning(f'Test std = {test_std}')
        if result_dict is not None:
            result_dict[alg] = test
            result_dict[f'{alg}_std'] = test_std
    return best_setup


def read_results(result_df: pd.DataFrame, alg_list, dataset_list, logger: logging.Logger, evaluator: Evaluator):
    result_df = delete_missing_folds(result_df, logger)
    n_folds = result_df['fold'].nunique()

    logger.warning('=== Performances ===')
    logger.warning('1. Choosing hyperparameters based on average val performance across datasets')
    df = deepcopy(result_df)
    check_missing_datasets(df, dataset_list, logger)
    df = delete_missing_datasets(df, dataset_list, logger)
    result_dict = {}

    for alg in alg_list:
        logger.info(f' ==> Algorithm: {alg}')
        agg_dict = {f'{col}_mean': (col, 'mean') for col in df.columns if col.startswith('val') or col.startswith('test')}
        agg_dict = agg_dict | {f'{col}_std': (col, 'std') for col in df.columns if col.startswith('val') or col.startswith('test')}
        agg_dict['dataset_type'] = ('dataset_type', 'first')
        df1 = df[(df['algorithm'] == alg) & (df['dataset'].isin(dataset_list))].groupby(['setup', 'dataset']).agg(**agg_dict).reset_index()
        if df1.empty:
            logger.warning(f'Skipping algorithm {alg}')
            result_dict[alg + '_1'] = -1e8
            result_dict[alg + '_1_std'] = 0
        else:
            # We always use performance as the selection metric, because different datasets might have different types
            df2 = df1.groupby(['setup']).agg({**{col: 'mean' for col in df1.columns if col.startswith('val') or col.startswith('test')}, 'dataset_type': 'first'}).reset_index()
            a = df2.loc[df2['val_performance_mean'].idxmax()]
            s = a.setup
            logger.warning(f'Setup with best val performance: {s}')
            logger.warning(f'Val performance  = {a.val_performance_mean}')
            logger.warning(f'Test performance = {a.test_performance_mean}')
            result_dict[alg + '_1'] = a.test_performance_mean
            test_var = 0
            for ds in dataset_list:
                x = df1[(df1['setup'] == s) & (df1['dataset'] == ds)]
                assert len(x) == 1
                x = x.iloc[0]
                # logger.debug(f'{x.test_performance_std}')
                test_var += x.test_performance_std ** 2 / n_folds   # The variance of the average needs to be divided by n_folds
                # logger.debug('result_dict_1_std = {}'.format(result_dict[alg + '_1_std']))
            result_dict[alg + '_1_std'] = math.sqrt(test_var) / len(dataset_list)

    
    logger.warning('=======================\n')
    logger.warning('2. Choosing hyperparameters based on val performance for each dataset')
    best_setup = best_each_dataset_alg(result_df, alg_list, dataset_list, logger, evaluator, result_dict)
    
    a = [['Dataset'] + ['Algorithm', 'Val_mean', 'Test_mean', 'Val_std', 'Test_std'] * len(alg_list)]
    for ds in sorted(best_setup):
        b = [ds]
        for alg in best_setup[ds]:
            logger.debug('Dataset {} - Algorithm {} - Best: {}'.format(ds, alg, best_setup[ds][alg][0]))
            b = b + [alg, best_setup[ds][alg][1], best_setup[ds][alg][3], best_setup[ds][alg][2], best_setup[ds][alg][4]]
        a.append(b)
    logger.debug(f'\n{tabulate(a, headers="firstrow", tablefmt="grid")}')
    logger.warning(f'Dataset count: {len(best_setup)}')

    result_list = []
    for alg in alg_list:
        result_list.append([0, alg, result_dict[alg + "_1"] * 100, result_dict[alg] * 100, result_dict[alg + "_1_std"] * 100, result_dict[alg + "_std"] * 100])
        logger.info(f'{alg}: {result_dict[alg + "_1"] * 100:.4f} ({result_dict[alg + "_1_std"] * 100:.4f}) / {result_dict[alg] * 100:.4f} ({result_dict[alg + "_std"] * 100:.4f})')
    # sort the results by their second performance
    result_list = sorted(result_list, key=lambda x: x[3], reverse=True)
    for k in range(len(result_list)):
        result_list[k][0] = k + 1
    ans = [['No.', 'Algorithm', 'First', 'Second', 'First std', 'Second std']] + result_list
    logger.info(f'\n{tabulate(ans, headers="firstrow", tablefmt="grid")}')


def get_result_df(folder, file, logger: logging.Logger) -> pd.DataFrame:
    result_df = None 
    broken_file_list = []
    if folder is not None:
        results_folder = Path(folder)
        for fn in results_folder.iterdir():
            logger.info(f'Reading {fn} - modified on {datetime.datetime.fromtimestamp(os.path.getmtime(fn))}')
            try:
                with open(fn, 'r') as f:
                    df1 = pd.read_csv(f)
                    result_df = df1 if result_df is None else pd.concat([result_df, df1], ignore_index=True)
            except pd.errors.ParserError:
                logger.warning(f'File {fn} broken')
                broken_file_list.append(fn)

    if file is not None:
        df_old = pd.read_csv(file)
        result_df = df_old if result_df is None else pd.concat([result_df, df_old], ignore_index=True)
    result_df = result_df.fillna(0)
    if len(broken_file_list) > 0:
        logger.warning(f'Broken file list: {broken_file_list}')
    return result_df
