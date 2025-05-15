import logging
import pandas as pd 
import argparse
import yaml
import os
import sys
sys.path.append(os.getcwd())
from analyze.utils import get_result_df, best_each_dataset_alg, get_dataset_alg_lists
from tabulate import tabulate
import statistics
from framework.utils import import_module


# Ranking is based on test performance, with the best setup selected via val performance
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', '-r', type=str)
    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--algorithms', '-a', type=str, help='Algorithms to be analyzed')
    parser.add_argument('--datasets', '-d', type=str, help='Datasets to be analyzed')
    parser.add_argument('--cutoff', '-c', type=int, help='Cut off at this rank. Otherwise some rank might have too much impact')
    parser.add_argument('--task', '-t', type=str, default='prediction', help='Task type')
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

    result_df = get_result_df(args.results_folder, args.file, logger)
    dataset_list, alg_list = get_dataset_alg_lists(args.datasets, args.algorithms, result_df, logger)
    logger.info('Number of datasets: {}'.format(len(dataset_list)))
    logger.info('Comparing between {} algorithms'.format(len(alg_list)))

    best_dict = best_each_dataset_alg(result_df, alg_list, dataset_list, logger, evaluator)
    rank_dict = {a: [] for a in alg_list}
    for dataset in dataset_list:
        perf_list = [(a, best_dict[dataset][a][3]) for a in alg_list]
        perf_list = sorted(perf_list, key=lambda x: x[1], reverse=True)
        cnt = 1
        cur = 1
        cur_perf = 10000.0
        tol = 0.001   # 0.1% accuracy difference is tolerated
        for i in range(len(perf_list)):
            if cur_perf - perf_list[i][1] > tol:
                cur = cnt
                cur_perf = perf_list[i][1]
            rk = cur if args.cutoff is None else min(cur, args.cutoff)
            rank_dict[perf_list[i][0]].append(rk)
            cnt += 1
    
    ans = []
    for alg in rank_dict:
        b = rank_dict[alg]
        ans.append([0, alg, min(b), max(b), statistics.mean(b), statistics.median(b)])
        #statistics.quantiles(b, n=4)[-1]
    ans = sorted(ans, key=lambda x: x[4])  # Sort by mean rank
    for k in range(len(ans)):
        ans[k][0] = k + 1
    ans = [['No.', 'Algorithm', 'Min', 'Max', 'Mean', 'Median']] + ans
    logger.info(tabulate(ans, headers="firstrow", tablefmt="grid"))


if __name__ == '__main__':
    main()
