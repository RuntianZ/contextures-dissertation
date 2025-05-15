import pandas as pd 
import argparse
import yaml
import os
import sys
sys.path.append(os.getcwd())
import logging
from analyze.utils import get_result_df

from framework.files import make_parent_dir
from framework.utils import import_module
from analyze.utils import read_results, get_dataset_alg_lists

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', '-r', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--algorithms', '-a', type=str, help='Algorithms to be analyzed')
    parser.add_argument('--datasets', '-d', type=str, help='Datasets to be analyzed')
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
    logger.warning('Found {} algorithms'.format(len(alg_list)))

    logger.warning('=== Final report ===')
    logger.warning('Number of datasets: {}'.format(len(dataset_list)))
    logger.warning('Number of algorithms: {}'.format(len(alg_list)))
    read_results(result_df, alg_list, dataset_list, logger, evaluator)

    if args.output is not None:
        logger.warning(f'Saving combined results to {args.output}...')
        make_parent_dir(args.output)
        result_df = result_df.loc[:, ~result_df.columns.str.contains("Unnamed")]
        result_df.to_csv(args.output)
        logger.warning('Save complete')

if __name__ == "__main__":
    main()
