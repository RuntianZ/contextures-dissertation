import pandas as pd 
import argparse
import yaml
import os
import sys
sys.path.append(os.getcwd())
from analyze.utils import get_result_df, best_each_dataset_alg, get_dataset_alg_lists
from tabulate import tabulate
import statistics
import matplotlib.pyplot as plt


# Ranking is based on test performance, with the best setup selected via val performance
def main():

    methods = ['tabpfn', 'vicreg_with_raw', 'catboost', 'vicreg_sft_with_raw', 'scarf_target_with_raw_vmpe']

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', '-r', type=str)
    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--algorithms', '-a', type=str, help='Algorithms to be analyzed')
    parser.add_argument('--datasets', '-d', type=str, help='Datasets to be analyzed')
    parser.add_argument('--cutoff', '-c', type=int, help='Cut off at this rank. Otherwise some rank might have too much impact')
    parser.add_argument('--save', '-s', type=str)
    args = parser.parse_args()

    result_df = get_result_df(args.results_folder, args.file)
    dataset_list, alg_list = get_dataset_alg_lists(args.datasets, args.algorithms, result_df)
    print('Ranking on {} datasets'.format(len(dataset_list)))

    if args.algorithms is None:
        alg_list = result_df['algorithm'].unique().tolist()
    else:
        with open(args.algorithms, 'r') as f:
            conf = yaml.safe_load(f)
        alg_list = [x['algorithm'] for x in conf['methods']]

    print('Comparing between {} algorithms'.format(len(alg_list)))

    best_dict = best_each_dataset_alg(result_df, alg_list, dataset_list)
    rank_dict = {a: [] for a in alg_list}
    for dataset in dataset_list:
        perf_list = [(a, best_dict[dataset][a][2]) for a in alg_list]
        perf_list = sorted(perf_list, key=lambda x: x[1], reverse=True)
        # print(perf_list)
        cnt = 1
        cur = 1
        cur_perf = 10000.0
        tol = 0.001   # 0.1% accuracy difference is tolerated
        for i in range(len(perf_list)):
            # print(perf_list[i][1])
            if cur_perf - perf_list[i][1] > tol:
                cur = cnt
                cur_perf = perf_list[i][1]
            rk = cur if args.cutoff is None else min(cur, args.cutoff)
            rank_dict[perf_list[i][0]].append(rk)
            cnt += 1
    
    for alg in methods:
        b = rank_dict[alg]
        b = sorted(b)
        plt.plot(b, label=alg)
    plt.legend()
    plt.grid(True)
    plt.savefig(args.save, dpi=300, bbox_inches='tight')

    # ans = []
    # for alg in rank_dict:
    #     b = rank_dict[alg]
    #     ans.append([alg, min(b), max(b), statistics.mean(b), statistics.median(b)])
    #     #statistics.quantiles(b, n=4)[-1]
    # ans = sorted(ans, key=lambda x: x[3])  # Sort by mean rank
    # ans = [['Algorithm', 'Min', 'Max', 'Mean', 'Median']] + ans
    # print(tabulate(ans, headers="firstrow", tablefmt="grid"))


if __name__ == '__main__':
    main()
