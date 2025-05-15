import logging
import pandas as pd 
import argparse
import yaml
import math
import os
import sys
sys.path.append(os.getcwd())
from analyze.utils import get_result_df, best_each_dataset_alg, get_dataset_alg_lists
from tabulate import tabulate
import statistics
from framework.utils import import_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str)
    parser.add_argument('file2', type=str)
    parser.add_argument('--datasets', '-d', type=str, help='Datasets to be analyzed')
    parser.add_argument('--task', '-t', type=str, default='prediction', help='Task type')
    parser.add_argument('--std', '-s', type=float, default=1, help='How many std of difference is considered the same')
    parser.add_argument('--folds', '-f', type=int, default=10)
    parser.add_argument('--logging_level', '-l', type=str, default='INFO', help='DEBUG/INFO/WARNING/ERROR')
    args = parser.parse_args()
    logger = logging.getLogger()
    logging.basicConfig(level=getattr(logging, args.logging_level), format='%(message)s')

    task = args.task
    with open('config/eval.yaml', 'r') as f:
        eval_config = yaml.safe_load(f)
    assert task in eval_config
    evaluator = import_module(eval_config[task])
    evaluator = evaluator()


    result_df1 = get_result_df(None, args.file1, logger)
    result_df2 = get_result_df(None, args.file2, logger)
    result_df = pd.concat([result_df1, result_df2], ignore_index=True)
    dataset_list, alg_list = get_dataset_alg_lists(args.datasets, None, result_df, logger)
    assert len(alg_list) == 2
    logger.info(f'Algorithm 1 = {alg_list[0]}')
    logger.info(f'Algorithm 2 = {alg_list[1]}')
    logger.info(f'Number of datasets = {len(dataset_list)}')
    logger.info(f'Number of algorithms = {len(alg_list)}')

    best_dict = best_each_dataset_alg(result_df, alg_list, dataset_list, logger, evaluator)
    win1 = 0
    lose1 = 0
    verb = []
    for dataset in dataset_list:
        perf1 = best_dict[dataset][alg_list[0]][3]
        perf2 = best_dict[dataset][alg_list[1]][3]
        std1 = best_dict[dataset][alg_list[0]][4]
        std2 = best_dict[dataset][alg_list[1]][4]
        res = None 
        if perf1 - args.std * std1 / math.sqrt(args.folds) > perf2:
            win1 += 1
            res = 'Win'
        if perf2 - args.std * std2 / math.sqrt(args.folds) > perf1:
            lose1 += 1
            res = 'Lose'
        if res is None:
            res = 'Draw'
        verb.append([dataset, res])
        logger.debug(f'dataset={dataset}, perf1 = {perf1}, perf2 = {perf2}, std1 = {std1}, std2 = {std2}, result={res}')
    verb.sort(key=lambda x: x[0])
    verb = [['Dataset', 'Result']] + verb
    logger.debug(tabulate(verb, headers="firstrow", tablefmt="grid"))
        
    ans = [['Alg1', 'Alg2', 'Alg1 win rate', 'Alg1 lose rate'], [alg_list[0], alg_list[1], win1 / len(dataset_list), lose1 / len(dataset_list)]]
    logger.info(tabulate(ans, headers="firstrow", tablefmt="grid"))


if __name__ == '__main__':
    main()
