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
import time
from copy import deepcopy
import logging
import numpy as np 
import torch
import json
from typing import Type, List, Union
import random 

from framework.base import Module, StandaloneModule, LinkedModule
from framework.files import download_dataset, make_parent_dir, read_main_config, load_all_algs, load_all_data_groups, load_single_config_yaml
from framework.dataset import TabularDataset
from framework.eval import eval_on_dataset, Evaluator
from framework.utils import import_module, DatasetLoader
from framework.base import summarize_recipe
from algorithms.basic import DummyModule
from analyze.utils import read_results


def load_dataset(config: dict, logger: logging.Logger) -> dict:
    """
    Load a dataset
    Input:
        - config: Config containing dataset path
    Output:
    A dict. Each key is a dataset name, and the value is a list of folds
    Each fold is a dict that contains:
        - train_set: Training set
        - val_set: Validation set
        - test_set: Test set
    """
    assert 'task' in config
    loader_config = load_single_config_yaml('config/loader.yaml')
    loader = import_module(loader_config[config['task']])
    loader = loader()
    assert isinstance(loader, DatasetLoader)
    return loader.load(config, logger)
        

def concat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concat by dim 1, add dim 1 if needed"""
    for i in range(len(tensors)):
        if len(tensors[i]) == 1:
            tensors[i] = tensors[i].view(-1, 1)
    return torch.cat(tensors, dim=1)


def instantiate_modules(recipe: list, run_config: dict, logger: logging.Logger, in_mixture: bool = False, breakpoint: int = None, no_load: bool = False) -> list:
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
            assert 'breakpoint' in md, 'Neither module nor breakpoint: {}'.format(md)
            # assert not in_mixture, 'Breakpoint cannot be inside a mixture'
            if breakpoint is not None and breakpoint == int(md['breakpoint']):
                logger.warning('!== Stop at breakpoint {}'.format(breakpoint))
                assert len(all_modules) > 0, 'No modules before the breakpoint!'
                assert frozen_id == len(all_modules) or all_modules[-1]['module'] == 'fit', 'Not all modules are fit before breakpoint {}'.format(breakpoint)
                all_modules.append('breakpoint')
                frozen_id = len(all_modules)
                break
            continue
        md_name = md['module']
        assert not 'breakpoint' in md, 'Module {} cannot also be a breakpoint'.format(md_name)
        md_cnt += 1
        if md_name in ['fit', 'transform', 'fit_transform']:
            if frozen_id == len(all_modules):
                # All module until this point are frozen. There is nothing to fit
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
            frozen_id = len(all_modules)  # Mixtures are assumed to contain transform at the end
        else:
            if not md_name in all_algs:
                raise NotImplementedError('Module {} is not implemented'.format(md_name))
            alg = all_algs[md_name]
            class_path = alg['class']
            hyperparameters = alg.get('args', [])
            module_config = {}
            for k in hyperparameters:
                if isinstance(k, dict):   # This arg has a default value
                    item = list(k.items())[0]
                    module_config[item[0]] = deepcopy(md.get(item[0], item[1]))
                else:
                    assert k in md, f"In module {md_name}, hyperparameter {k} is not specified and does not have a default value"
                    module_config[k] = deepcopy(md[k])
            
            logger.debug('Module name = {}, config = {}'.format(md_name, module_config))
            module_class = import_module(class_path)
            ckpt_name = None
            if 'ckpt_name' in alg:
                ckpt_name = alg['ckpt_name']
            md_obj = module_class(module_config, default_config, logger, md_name, ckpt_name, seed)
            if 'should_load' in md:
                md_obj.config['should_load'] = md['should_load']
            if no_load:
                md_obj.config['should_load'] = False
            if 'should_save' in md:
                md_obj.config['should_save'] = md['should_save']
            if 'delete_ckpt' in md:
                md_obj.config['delete_ckpt'] = md['delete_ckpt']
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


def cook(dataset_train: TabularDataset, dataset_val: TabularDataset, dataset_test: TabularDataset, all_modules: List[Union[dict, Module]], run_config: dict,
         logger: logging.Logger, evaluator: Evaluator, breakpoint: int, return_datasets: bool = False, in_mixture: bool = False, prev_modules: list = [], no_load: bool = False, no_train: bool = False):
    """
    Cook with a recipe specified by all_modules
    Returns a list of dict. Each item corresponds to an eval module
    Examples:
      ans[0]['train_accuracy']
      ans[0]['test_performance']
    """
    logger.debug(f' ==> cook: all_modules = {all_modules}, prev_modules = {prev_modules}')
    ans = []
    modules = deepcopy(prev_modules)
    old_len = len(modules)
    frozen_id = len(modules) # modules[i] is frozen, where i < frozen_id

    # Hide the targets of val and test sets
    val_y = dataset_val.y
    dataset_val.y = np.zeros_like(dataset_val.y)
    test_y = dataset_test.y
    dataset_test.y = np.zeros_like(dataset_test.y)

    for md in all_modules:
        if isinstance(md, Module):
            modules.append(md)
            if isinstance(md, StandaloneModule):
                logger.debug('init_and_load on module {}'.format(md.name))
                dataset_train = md.init_and_load(dataset_train, modules, len(modules) - 1, None)
                frozen_id = len(modules)
                # Automatic fit and transform
                dataset_train = md.fit_transform(dataset_train, modules)
                dataset_val = md.transform(dataset_val)
                dataset_test = md.transform(dataset_test)
        elif md == 'breakpoint':
            return 'breakpoint', None
        else:
            match md['module']:
                case 'fit':
                    assert not isinstance(modules[-1], StandaloneModule), f'{modules}'  # Bug in instantiate
                    # First, init and load all linked modules
                    for i in range(frozen_id, len(modules)):
                        assert isinstance(modules[i], LinkedModule), 'modules = {}, i = {}'.format(modules, i)
                        logger.debug('init_and_load on module {}'.format(modules[i].name))
                        dataset_train = modules[i].init_and_load(dataset_train, modules, i, md)
                    # Then, train on the final module (will train all)
                    if not no_train:
                        logger.debug(f"Running train function from module {modules[-1].name}")
                        modules[-1].train(dataset_train, modules, frozen_id, md)
                case 'transform':
                    assert isinstance(modules[-1], LinkedModule), f'modules = {modules}' # Bug in instantiate
                    # if not no_train:
                    logger.debug(f"Running transform_all function from module {modules[-1].name}")
                    dataset_train = modules[-1].transform_all(dataset_train, modules, frozen_id, md)
                    dataset_val = modules[-1].transform_all(dataset_val, modules, frozen_id, md)
                    dataset_test = modules[-1].transform_all(dataset_test, modules, frozen_id, md)
                    frozen_id = len(modules)
                case 'fit_transform':
                    assert False, 'Bug: Should not contain fit_transform'
                case 'eval':
                    assert not in_mixture, 'Does not accept eval module in a mixture'
                    if not no_train:
                        # Temporarily recover val_y and test_y
                        dataset_val.y = val_y
                        dataset_test.y = test_y
                        results = eval_on_dataset(evaluator, dataset_train, 'train', logger) | eval_on_dataset(evaluator, dataset_val, 'val', logger) | eval_on_dataset(evaluator, dataset_test, 'test', logger)
                        dataset_val.y = np.zeros_like(dataset_val.y)
                        dataset_test.y = np.zeros_like(dataset_test.y)
                        ans.append(results)
                    frozen_id = len(modules)
                case 'mixture':
                    assert not in_mixture, 'A mixture cannot be inside another mixture'
                    outputs = {'train_X': [], 'train_y': [], 'val_X': [], 'val_y': [], 'test_X': [], 'test_y': []}
                    mode = md.get('mode', 'X')
                    assert mode in ['X', 'y', 'both']
                    assert frozen_id == len(modules), 'Not all modules frozen before mixture'
                    cnt = 0
                    dummy_name = 'mixture'
                    dummy_config = {}
                    dummy_args_keys = []
                    for mixture_recipe in sorted(md):
                        if mixture_recipe in ['module', 'mode']:
                            continue
                        rp = md[mixture_recipe]  # recipe list
                        logger.debug('Mixture recipe {}: {}'.format(mixture_recipe, rp))
                        rp_modules = instantiate_modules(rp, run_config, logger, True, breakpoint, no_load)
                        new_train = deepcopy(dataset_train)
                        new_train.data = torch.clone(dataset_train.data)
                        new_train.target = torch.clone(dataset_train.target)
                        new_val = deepcopy(dataset_val)
                        new_val.data = torch.clone(dataset_val.data)
                        new_val.target = torch.clone(dataset_val.target)
                        new_test = deepcopy(dataset_test)
                        new_test.data = torch.clone(dataset_test.data)
                        new_test.target = torch.clone(dataset_test.target)
                        new_modules = deepcopy(modules)
                        mans = cook(new_train, new_val, new_test, rp_modules, run_config, logger, evaluator, breakpoint, True, True, new_modules)
                        if mans == 'breakpoint':
                            del new_train
                            del new_val
                            del new_test
                            del outputs
                            return 'breakpoint'
                        new_train, new_val, new_test, extra_modules = mans
                        outputs['train_X'].append(torch.clone(new_train.data))
                        outputs['train_y'].append(torch.clone(new_train.target))
                        outputs['val_X'].append(torch.clone(new_val.data))
                        outputs['val_y'].append(torch.clone(new_val.target))
                        outputs['test_X'].append(torch.clone(new_test.data))
                        outputs['test_y'].append(torch.clone(new_test.target))
                        del new_train
                        del new_val
                        del new_test
                        rp_name = summarize_recipe(extra_modules)
                        rp_name = rp_name[2:]
                        logger.debug('extra_modules = {}'.format(extra_modules))
                        logger.debug('rp_name = {}'.format(rp_name))
                        dummy_name += f'_recipe{cnt}[{rp_name}]'
                        for i in range(len(extra_modules)):
                            for k in sorted(extra_modules[i].args_keys):
                                new_key = f'recipe{cnt}_module{i}_{k}'
                                dummy_args_keys.append(new_key)
                                assert not new_key in dummy_config, f'Repeated dummy key: {new_key}'
                                dummy_config[new_key] = extra_modules[i].config[k]
                        del extra_modules
                        del new_modules
                        cnt += 1
                    dummy_module = DummyModule(f'mixture', dummy_name)
                    dummy_module.args_keys = dummy_args_keys
                    dummy_module.config = dummy_config
                    if mode in ['X', 'both']:
                        dataset_train.data = concat_tensors(outputs['train_X'])
                        dataset_val.data = concat_tensors(outputs['val_X'])
                        dataset_test.data = concat_tensors(outputs['test_X'])
                    if mode in ['y', 'both']:
                        dataset_train.target = concat_tensors(outputs['train_y'])
                        dataset_val.target = concat_tensors(outputs['val_y'])
                        dataset_test.target = concat_tensors(outputs['test_y'])
                    del outputs
                    modules.append(dummy_module)
                    frozen_id = len(modules)
                case _:
                    raise RuntimeError('Unknown module {}'.format(md['module']))
    if return_datasets:
        return [dataset_train, dataset_val, dataset_test, modules[old_len:]]
    else:
        return ans, modules
    

def insert_item(item: dict, d: dict) -> dict:
    for k in item:
        if not k in d:
            d[k] = []
        d[k].append(item[k])
    return d


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
            for k in sorted(md.args_keys):
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


def work(all_folds, dataset_name, method_name, recipe_name, logger, default_config, default_fit_config, default_transform_batch_size, recipe, evaluator, breakpoint, no_load) -> dict:
    new_item = {}
    seed = default_config.get('seed', -1)
    if not 'magic_number' in default_config:
        magic_number = random.randint(0, 99999999)
        default_config['magic_number'] = magic_number
    no_train = default_config.get('no_train', False)
    if no_train:
        default_config['save'] = False
    for fold, fold_data in enumerate(all_folds):
        logger.warning('# Fold {}'.format(fold))
        # 1. Instantiate the modules
        run_config = {
            'logger': logger,
            'default_transform_batch_size': default_transform_batch_size,
            'default_fit_config': default_fit_config,
            'default_config': default_config,
            'seed': seed,
            'no_load': no_load,
        }
        all_modules = instantiate_modules(recipe, run_config, logger, False, breakpoint, no_load)
        # logger.debug('all_modules = {}'.format(all_modules))
                    
        # 2. Cook the modules
        train_set = fold_data['train_set']
        val_set = fold_data['val_set']
        test_set = fold_data['test_set']
        logger.warning(f" ==> Start training:   Dataset  {train_set.name}   (Fold {fold})   -   Type  {train_set.target_type}   -   Algorithm  {method_name}   -   Recipe  {recipe_name}")
        _t1 = time.time()
        # We deepcopy the datasets to prevent pollution for other recipes
        dataset_train = deepcopy(train_set)
        dataset_val = deepcopy(val_set)
        dataset_test = deepcopy(test_set)
        results, result_modules = cook(dataset_train, dataset_val, dataset_test, all_modules, run_config, logger, evaluator, breakpoint, no_load=no_load, no_train=no_train)
        _t2 = time.time()
        logger.debug(f"Recipe  {recipe_name}  -  Dataset  {train_set.name}  -  Type  {train_set.target_type}  -  Fold  {fold}")
        train_time = _t2 - _t1
        logger.info(f"Training completed in {train_time} seconds.")
        if results == 'breakpoint':
            logger.warning('### Stop at breakpoint')
        elif len(results) == 0:
            logger.warning("### No eval results")
        else:
            assert len(results) == 1, "Currently do not support multiple eval modules"
            setup_name = f'seed[{seed}]' + summarize_recipe(result_modules)
            logger.debug('setup = {}'.format(setup_name))
            results = results[0] | summarize_args(result_modules) | {
                'seed': seed,
                'dataset': dataset_name,
                'fold': fold,
                'dataset_type': train_set.target_type,
                'num_classes': train_set.num_classes,
                'num_train_samples': train_set.num_instances,
                'algorithm': method_name,
                'recipe': recipe_name,
                'setup': setup_name,
                'train_time': train_time,
            }
            logger.debug(f'results = {results}')
            insert_item(results, new_item)
            logger.warning('### Train perf = {}    Val perf = {}    Test perf = {}'.format(results['train_performance'], results['val_performance'], results['test_performance']))
        for md in all_modules:
            del md
        del all_modules
        del dataset_train
        del dataset_val
        del dataset_test
    return new_item


def exp_main(args, config: dict = None):
    # Logger outputs warnings to stdout, and errors to stderr
    logger = logging.getLogger()
    logging.basicConfig(level=getattr(logging, args.logging_level), format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # Handler: This part needs fixing
    # stdout_handler = logging.StreamHandler(sys.stdout)
    # stderr_handler = logging.StreamHandler(sys.stderr)
    # stdout_handler.setLevel(logging.WARNING)
    # stderr_handler.setLevel(logging.ERROR)
    # logger.addHandler(stdout_handler)
    # logger.addHandler(stderr_handler)

    if args.download:
        assert args.data_folder is not None, "Download failed because data folder is not specified"
        logger.warning('==> Downloading all datasets to {}...'.format(args.data_folder))
        download_dataset(args.data_folder)

    if config is None:
        config = read_main_config(args.config_file)
    logger.info('Config: {}'.format(json.dumps(config)))
    default_config = config.get('defaults', {})
    debug = default_config.get('debug', False)
    if debug:
        logger.debug('Using debug mode, env config will be ignored')
        args.ignore_env = True
    env_config = load_single_config_yaml('config/env.yaml')
    if not args.ignore_env:
        if 'data_folder' in env_config:
            default_config['data_folder'] = env_config['data_folder']
        if not 'backup_folders' in env_config:
            env_config['backup_folders'] = []
        default_config['ssh_root'] = env_config.get('ssh_root', [])
        if 'ckpt_folder' in default_config:
            for i in range(len(env_config['backup_folders'])):
                env_config['backup_folders'][i] = os.path.join(env_config['backup_folders'][i], default_config['ckpt_folder'])
            for i in range(len(default_config['ssh_root'])):
                default_config['ssh_root'][i] = os.path.join(default_config['ssh_root'][i], default_config['ckpt_folder'])
        if 'ckpt_folder' in env_config:
            if 'ckpt_folder' in default_config:
                default_config['ckpt_folder'] = os.path.join(env_config['ckpt_folder'], default_config['ckpt_folder'])
            else:
                default_config['ckpt_folder'] = env_config['ckpt_folder']
        if not 'backup_folders' in default_config:
            default_config['backup_folders'] = []
        default_config['backup_folders'] = env_config['backup_folders'] + default_config['backup_folders']
        if 'results_folder' in env_config:
            default_config['result_file'] = os.path.join(env_config['results_folder'], default_config['result_file'])

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
        'scheduler': 'None',
    }
    for k in default_fit_config:
        k0 = 'default_fit_' + k
        if k0 in default_config:
            default_fit_config[k] = default_config[k0]
    if args.no_load:
        default_config['load'] = False
    logger.debug('default_config = {}'.format(default_config))
    assert (not 'should_save' in default_config) and (not 'should_load' in default_config), 'should_save and should_load should be set for individual modules'

    result_df = None
    dataset_list = config.get('datasets', [])
    if 'groups' in config:
        all_groups = load_all_data_groups()
        for a in config['groups']:
            if not a in all_groups:
                raise ValueError("Group {} does not exist".format(a))
            dataset_list = dataset_list + all_groups[a]

    if not 'task' in default_config:
        default_config['task'] = 'prediction'
    task = default_config['task']
    eval_config = load_single_config_yaml('config/eval.yaml')
    assert task in eval_config
    evaluator = import_module(eval_config[task])
    evaluator = evaluator()

    # Main routine
    for c in dataset_list:
        dataset_path = os.path.join(default_config['data_folder'], c)
        dataset_config = default_config | {'dataset_path': dataset_path}
        loaded_datasets = load_dataset(dataset_config, logger)

        for rn in config['recipe']:
            for ds in loaded_datasets:
                logger.warning('=== Using recipe {} on dataset {} ==='.format(rn, ds))
                all_folds = loaded_datasets[ds]
                a = rn.split('/')
                method_name = a[0]
                recipe_name = a[1]
                recipe = config['recipe'][rn] 
                logger.debug(f'recipe: {recipe}')
                new_item = {}
                try:
                    new_item = work(all_folds, ds, method_name, recipe_name, logger, default_config, default_fit_config, default_transform_batch_size, recipe, evaluator, args.breakpoint, args.no_load)
                except Exception as e:
                    logger.exception(f'!== Error of type {type(e).__name__} when running recipe {recipe_name} on dataset {c}')
                    if args.stop_on_error or isinstance(e, AssertionError) or isinstance(e, PermissionError) or "CUDA" in str(e):
                        raise

                if bool(new_item) and not args.no_result:
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
        read_results(result_df, alg_list, dataset_list, logger, evaluator)

