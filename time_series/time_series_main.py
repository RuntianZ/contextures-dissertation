import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # To avoid "Detect OpenMP Loop and this application may hang" warnings
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.getcwd())
import argparse
import yaml
import pandas as pd
from pathlib import Path
import importlib
import time
from copy import deepcopy
import logging
import torch
import json
from typing import Type, List, Union

from framework.base import Module, StandaloneModule
from framework.files import download_dataset, make_parent_dir, read_main_config, load_all_algs, load_all_data_groups

from framework.dataset_ts import TimeSeriesDataset, train_test_split
from framework.eval import metric_string, eval_on_dataset
from framework.utils import get_device
from analyze.utils import read_results


def load_dataset(config: dict, logger: logging.Logger) -> list:
    """
    Load a dataset
    Input:
        - config: Config containing dataset path
    Output:
    A list of dicts. Each dict in the list contains:
        - train_set: Training set
        - val_set: Validation set
        - test_set: Test set
    """
    dataset_path = Path(config['dataset_path'])
    num_folds = config['folds']
    dataset = TimeSeriesDataset.read(dataset_path)

    logger.debug(f"dataset name: {dataset.name}")
    logger.debug(f"target type: {dataset.target_type}")
    logger.debug(f"number of target classes: {dataset.num_classes}")
    logger.debug(f"number of features: {dataset.num_features}")
    logger.debug(f"number of instances: {len(dataset.X)}")
    logger.debug(f"indeces of categorical features: {dataset.cat_idx}")
    
    all_folds = []
    for fold in range(num_folds):
        train, val, test = train_test_split(dataset, fold)
        all_folds.append({'train_set': train, 'val_set': val, 'test_set': test})
    return all_folds


def concat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concat by dim 1, add dim 1 if needed"""
    for i in range(len(tensors)):
        if len(tensors[i]) == 1:
            tensors[i] = tensors[i].view(-1, 1)
    return torch.cat(tensors, dim=1)



def instantiate_modules(recipe: list, run_config: dict, logger: logging.Logger, in_mixture: bool = False, breakpoint: int = None) -> list:
    """
    Translates a recipe into a list of modules
    Does not preprocess mixture 
    """
    logger = run_config['logger']
    default_fit_config = run_config['default_fit_config']
    default_transform_batch_size = run_config['default_transform_batch_size']
    default_config = run_config['default_config']
    seed = run_config['seed']

    all_modules = []  # Each entry is a module or a dict (e.g. for fit)
    frozen_id = 0  # all_modules[i] is frozen, where i < frozen_id
    all_algs = load_all_algs()
    md_cnt = 0
    for md in recipe:
        if not 'module' in md:
            assert 'breakpoint' in md, 'Neither module nore breakpoint: {}'.format(md)
            assert not in_mixture, 'Breakpoint cannot be inside a mixture'
            if breakpoint is not None and breakpoint == int(md['breakpoint']):
                logger.warning('!== Stop at breakpoint {}'.format(breakpoint))
                assert len(all_modules) > 0, 'No modules before the breakpoint!'
                assert frozen_id == len(all_modules) or all_modules[-1]['module'] == 'fit', 'Not all modules are fit before breakpoint {}'.format(breakpoint)
                break
            continue
        md_name = md['module']
        assert not 'breakpoint' in md, 'Module {} cannot also be a breakpoint'.format(md_name)
        md_cnt += 1
        if md_name in ['fit', 'transform', 'fit_transform']:
            if frozen_id == len(all_modules):
                # All module till this point are frozen. There is nothing to fit
                logger.error(f'Module {md_cnt} ({md_name}) is ignored because all previous modules are frozen')
            else:
                # Break fit_transform into one fit and one transform
                if md_name.startswith('fit'):
                    md0 = default_fit_config | md
                    md0['module'] = 'fit'
                    all_modules.append(md0)
                if md_name.endswith('transform'):
                    md0 = {'batch_size': default_transform_batch_size} | md
                    md0['module'] = 'transform'
                    all_modules.append(md0)
                    frozen_id = len(all_modules)
        elif md_name == 'eval':
            if frozen_id != len(all_modules):
                # Some modules are not frozen. It's ok if they are not transformed
                # However, it's not ok if they are not fit. So could be dangerous here
                logger.error(f'Module {md_cnt} (eval): Not all modules are frozen. A transform is added, but be warned that it is possible that some previous modules are not fit.')
                all_modules.append({'module': 'transform', 'batch_size': default_transform_batch_size})
            all_modules.append({'module': 'eval'})
            frozen_id = len(all_modules)
        elif md_name == 'mixture':
            # We will do nothing to mixture modules here
            all_modules.append(md)
        else:
            if not md_name in all_algs:
                raise NotImplementedError('Module {} is not implemented'.format(md_name))
            alg = all_algs[md_name]
            class_path = alg['class']
            hyperparameters = alg.get('args', [])
            module_config = {}
            for k in hyperparameters:
                if isinstance(k, dict):
                    item = list(k.items())[0]
                    module_config[item[0]] = md.get(item[0], item[1])
                else:
                    if k in md:
                        module_config[k] = md[k]
                    else:
                        logger.error(f"In module {md_name}, hyperparameter {k} is not specified and does not have a default value")
                    
            module_class = import_module(class_path)
            ckpt_name = None
            if 'ckpt_name' in alg:
                ckpt_name = alg['ckpt_name']
            md_obj = module_class(module_config, default_config, logger, md_name, ckpt_name, seed)
            if seed > 0:
                seed += 100
            md_obj.args_keys = md_obj.args_keys + list(module_config.keys())
            if isinstance(md_obj, StandaloneModule) and frozen_id != len(all_modules):
                # This requires all previous modules to be frozen. If not, will perform a transform
                # However, it could be dangerous because they might have not been fit
                logger.error(f'Module {md_cnt} ({md_name}) is a standalone module, but not all previous modules are frozen. We will add a transform here, but be warned that it is possible that some previous modules are not fit.')
                all_modules.append({'module': 'transform', 'batch_size': default_transform_batch_size})
            all_modules.append(md_obj)
            if isinstance(md_obj, StandaloneModule):
                frozen_id = len(all_modules)
    if frozen_id != len(all_modules):
        if in_mixture:
            # For a recipe in a mixture, we require all modules to be frozen 
            logger.error("In this mixture recipe, not all modules are frozen. A transform is added.")
            all_modules.append({'module': 'transform', 'batch_size': default_transform_batch_size})
            frozen_id = len(all_modules)
        else:
            logger.error("Not all modules are frozen in this recipe. Expected if you are only training and not evaluating. If not, you might forget to put an eval in the end")
            # all_modules.append({'module': 'eval'})
            # frozen_id = len(all_modules)
    return all_modules


def cook(dataset_train: TimeSeriesDataset, dataset_val: TimeSeriesDataset, dataset_test: TimeSeriesDataset, all_modules: List[Union[dict, Module]], run_config: dict,
         logger: logging.Logger, return_datasets: bool = False, in_mixture: bool = False) -> list:
    """
    Cook with a recipe specified by all_modules
    Returns a list of dict. Each item corresponds to an eval module
    Examples:
      ans[0]['train_accuracy']
      ans[0]['test_performance']
    """
    logger.debug(f'cook: all_modules = {all_modules}')

    ans = []
    modules = []
    frozen_id = 0  # modules[i] is frozen, where i < frozen_id
    for md in all_modules:
        if isinstance(md, Module):
            logger.debug('init_and_load on module {}'.format(md.name))
            dataset_train = md.init_and_load(dataset_train, modules)
            modules.append(md)
            if isinstance(md, StandaloneModule):
                # Automatic fit and transform
                dataset_train = md.fit_transform(dataset_train, modules[:-1])
                frozen_id = len(modules)
                dataset_val = md.transform(dataset_val)
                dataset_test = md.transform(dataset_test)
            
        else:
            match md['module']:
                case 'fit':
                    assert not isinstance(modules[-1], StandaloneModule)
                    linked_modules = modules[frozen_id:]
                    logger.debug(f"Running train function from module {modules[-1].name}")
                    modules[-1].train(dataset_train, modules[:-1], linked_modules, **md)
                case 'transform':
                    assert not isinstance(modules[-1], StandaloneModule)
                    logger.debug(f"Running transform_all function from module {modules[-1].name}")
                    dataset_train = modules[-1].transform_all(dataset_train, modules[:-1], linked_modules, **md)
                    frozen_id = len(modules)
                    dataset_val = modules[-1].transform_all(dataset_val, modules[:-1], linked_modules, **md)
                    dataset_test = modules[-1].transform_all(dataset_test, modules[:-1], linked_modules, **md)
                case 'fit_transform':
                    assert False, 'Bug: Should not contain fit_transform'
                case 'eval':
                    assert not in_mixture, 'Does not accept eval module in a mixture'
                    results = eval_on_dataset(dataset_train, 'train', logger) | eval_on_dataset(dataset_val, 'val', logger) | eval_on_dataset(dataset_test, 'test', logger)
                    ans.append(results)
                case 'mixture':
                    outputs = {'train_X': [], 'train_y': [], 'val_X': [], 'val_y': [], 'test_X': [], 'test_y': []}
                    mode = md.get('mode', 'X')
                    assert mode in ['X', 'y', 'both']
                    for k in md:
                        if k in ['module', 'mode']:
                            continue
                        elif k == 'raw':
                            if md[k] == 'on':
                                outputs['train_X'].append(torch.clone(dataset_train.data))
                                outputs['train_y'].append(torch.clone(dataset_train.target))
                                outputs['val_X'].append(torch.clone(dataset_val.data))
                                outputs['val_y'].append(torch.clone(dataset_val.target))
                                outputs['test_X'].append(torch.clone(dataset_test.data))
                                outputs['test_y'].append(torch.clone(dataset_test.target))
                        else:
                            rp = md[k]
                            logger.debug('Mixture recipe: {}'.format(rp))
                            rp_modules = instantiate_modules(rp, run_config, logger, True)
                            new_train = deepcopy(dataset_train)
                            new_train.data = torch.clone(dataset_train.data)
                            new_train.target = torch.clone(dataset_train.target)
                            new_val = deepcopy(dataset_val)
                            new_val.data = torch.clone(dataset_val.data)
                            new_val.target = torch.clone(dataset_val.target)
                            new_test = deepcopy(dataset_test)
                            new_test.data = torch.clone(dataset_test.data)
                            new_test.target = torch.clone(dataset_test.target)
                            new_train, new_val, new_test = cook(new_train, new_val, new_test, rp_modules, run_config, logger, True, True)
                            outputs['train_X'].append(torch.clone(new_train.data))
                            outputs['train_y'].append(torch.clone(new_train.target))
                            outputs['val_X'].append(torch.clone(new_val.data))
                            outputs['val_y'].append(torch.clone(new_val.target))
                            outputs['test_X'].append(torch.clone(new_test.data))
                            outputs['test_y'].append(torch.clone(new_test.target))
                            del new_train
                            del new_val
                            del new_test
                    if mode in ['X', 'both']:
                        dataset_train.data = concat_tensors(outputs['train_X'])
                        dataset_val.data = concat_tensors(outputs['val_X'])
                        dataset_test.data = concat_tensors(outputs['test_X'])
                    if mode in ['y', 'both']:
                        dataset_train.target = concat_tensors(outputs['train_y'])
                        dataset_val.target = concat_tensors(outputs['val_y'])
                        dataset_test.target = concat_tensors(outputs['test_y'])
                case _:
                    raise RuntimeError('Unknown module {}'.format(md['module']))
    if return_datasets:
        return [dataset_train, dataset_val, dataset_test]
    else:
        return ans
    

def insert_item(item: dict, d: dict) -> dict:
    for k in item:
        if not k in d:
            d[k] = []
        d[k].append(item[k])
    return d


def import_module(class_path: str) -> Type[Module]:
    a = class_path.rfind('.')
    class_name = class_path[a+1:]
    module = importlib.import_module(class_path[:a])
    method_class = getattr(module, class_name)
    return method_class


def summarize_recipe(all_modules: List[Union[dict, Module]]) -> str:
    """The string starts with __ (2 _)"""
    ans = ''
    for md in all_modules:
        if isinstance(md, Module):
            ans += '__' + md.to_string()
        else:
            if md['module'] in ['fit', 'fit_transform']:
                ans += '__' + f'fit_epochs[{md["epochs"]}]_batch_size[{md["batch_size"]}]_optimizer[{md["optimizer"]}]_lr[{md["lr"]}]_wd[{md["wd"]}]_scheduler[{md["scheduler"]}]'
    # ans = ans[2:]
    return ans


def summarize_args(all_modules: List[Union[dict, Module]]) -> dict:
    ans = {}
    mds = {}
    for md in all_modules:
        if isinstance(md, Module):
            mdn = md.name
            if not mdn in mds:
                mds[mdn] = 0
            else:
                mds[mdn] += 1
            for k in md.args_keys:
                ans[f'{mdn}_{mds[mdn]}_{k}'] = md.config[k]
        else:
            if md['module'] == 'fit':
                mdn = 'fit'
                if not mdn in mds:
                    mds[mdn] = 0
                else:
                    mds[mdn] += 1
                for k in ['epochs', 'batch_size', 'optimizer', 'lr', 'wd', 'scheduler']:
                    ans[f'{mdn}_{mds[mdn]}_{k}'] = md[k]
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config yaml file')
    parser.add_argument('--result_file', '-r', type=str)
    parser.add_argument('--data_folder', '-f', type=str)
    parser.add_argument('--download', '-d', type=str,
                        help='If need to download the dataset, set the dataset folder here')
    parser.add_argument('--logging_level', '-l', type=str, default='INFO', help='DEBUG/INFO/WARNING/ERROR')
    parser.add_argument('--stop_on_error', '-s', action='store_true', default=False)
    parser.add_argument('--breakpoint', '-b', type=int)
    args = parser.parse_args()

    # Logger outputs warnings to stdout, and errors to stderr
    logger = logging.getLogger()
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.setLevel(getattr(logging, args.logging_level))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.WARNING)
    stderr_handler.setLevel(logging.ERROR)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    if args.download is not None:
        logger.warning('==> Downloading all datasets...')
        download_dataset(args.download)

    config = read_main_config(args.config_file)
    logger.debug('Config: {}'.format(json.dumps(config)))
    default_config = config.get('defaults', {})
    if args.data_folder is not None:
        default_config['data_folder'] = args.data_folder
    if args.result_file is not None:
        default_config['result_file'] = args.result_file
    default_transform_batch_size = default_config.get('default_transform_batch_size', 64)
    default_fit_config = {
        'epochs': 50,
        'batch_size': 64,
        'optimizer': 'adamw',
        'lr': 0.1,
        'wd': 0.0001,
        'scheduler': 'none',
    }
    for k in default_fit_config:
        k0 = 'default_fit_' + k
        if k0 in default_config:
            default_fit_config[k] = default_config[k0]

    result_df = None
    dataset_list = config.get('datasets', [])
    if 'groups' in config:
        all_groups = load_all_data_groups()
        for a in config['groups']:
            if not a in all_groups:
                raise ValueError("Group {} does not exist".format(a))
            dataset_list = dataset_list + all_groups[a]

    import numpy as np
    # Main routine
    for c in dataset_list:
        dataset_path = os.path.join(default_config['data_folder'], c)
        dataset_config = default_config | {'dataset_path': dataset_path}
        all_folds = load_dataset(dataset_config, logger)

        for rn in config['recipe']:
            logger.warning('=== Using recipe {} ==='.format(rn))
            a = rn.split('/')
            method_name = a[0]
            recipe_name = a[1]
            recipe = config['recipe'][rn] 
            logger.debug(f'recipe: {recipe}')
            has_results = False
            try:
                new_item = {}
                seed = default_config.get('seed', -1)
                for fold, fold_data in enumerate(all_folds):
                    # 1. Instantiate the modules
                    run_config = {
                        'logger': logger,
                        'default_transform_batch_size': default_transform_batch_size,
                        'default_fit_config': default_fit_config,
                        'default_config': default_config,
                        'seed': seed,
                    }
                    all_modules = instantiate_modules(recipe, run_config, logger, False, args.breakpoint)
                                
                    # 2. Cook the modules
                    train_set = fold_data['train_set']
                    val_set = fold_data['val_set']
                    test_set = fold_data['test_set']
                    logger.warning(f" ==> Start training:   Dataset  {train_set.name}   (Fold {fold})   -   Type  {train_set.target_type}   -   Recipe  {rn}")
                    _t1 = time.time()
                    # We deepcopy the datasets to prevent pollution for other recipes
                    dataset_train = deepcopy(train_set)
                    dataset_val = deepcopy(val_set)
                    dataset_test = deepcopy(test_set)
                    results = cook(dataset_train, dataset_val, dataset_test, all_modules, run_config, logger)
                    _t2 = time.time()
                    logger.debug(f"Recipe  {rn}  -  Dataset  {train_set.name}  -  Type  {train_set.target_type}  -  Fold  {fold}")
                    logger.info(f"Training completed in {_t2 - _t1} seconds.")
                    if len(results) == 0:
                        logger.warning("### No eval results")
                    else:
                        assert len(results) == 1, "Currently do not support multiple eval modules"
                        results = results[0] | summarize_args(all_modules) | {
                            'seed': seed,
                            'dataset': train_set.name,
                            'fold': fold,
                            'dataset_type': train_set.target_type,
                            'num_classes': train_set.num_classes,
                            'num_train_samples': train_set.num_instances,
                            'algorithm': method_name,
                            'recipe': recipe_name,
                            'setup': f'seed[{seed}]' + summarize_recipe(all_modules),
                        }
                        logger.debug(f'results = {results}')
                        insert_item(results, new_item)
                        has_results = True
                        logger.warning('### Train perf = {}    Val perf = {}    Test perf = {}'.format(results['train_performance'], results['val_performance'], results['test_performance']))
            except Exception as e:
                logger.exception(f'!== Error of type {type(e).__name__} when running recipe {rn} on dataset {c}')
                if args.stop_on_error or isinstance(e, AssertionError):
                    raise

            if has_results:
                _t1 = time.time()
                new_df = pd.DataFrame(new_item)
                result_df = new_df if result_df is None else pd.concat([new_df, result_df], ignore_index=True)
                result_file = default_config.get('result_file', None)
                if result_file is not None:
                    make_parent_dir(result_file)
                    result_df = result_df.loc[:, ~result_df.columns.astype(str).str.contains("Unnamed")]
                    result_df.to_csv(result_file)
                _t2 = time.time()
                logger.debug(f'Result saving time : {_t2 - _t1}')
    
    if result_df is None:
        logger.warning('!== No eval results')
    else:
        alg_list = result_df['algorithm'].unique().tolist()
        dataset_list = result_df['dataset'].unique().tolist()   
        logger.warning('=== Final report ===')
        logger.info('Number of datasets: {}'.format(len(dataset_list)))
        logger.info('Number of algorithms: {}'.format(len(alg_list)))
        read_results(result_df, alg_list, dataset_list, logger)


if __name__ == '__main__':
    main()
