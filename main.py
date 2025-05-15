import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # To avoid "Detect OpenMP Loop and this application may hang" warnings
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.getcwd())
import argparse 
from exp.main import exp_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config yaml file')
    parser.add_argument('--result_file', '-r', type=str)
    parser.add_argument('--data_folder', '-f', type=str)
    parser.add_argument('--download', '-d', action='store_true', default=False)
    parser.add_argument('--logging_level', '-l', type=str, default='WARNING', help='DEBUG/INFO/WARNING/ERROR')
    parser.add_argument('--stop_on_error', '-s', action='store_true', default=False)
    parser.add_argument('--breakpoint', '-b', type=int)
    parser.add_argument('--ignore_env', '-i', action='store_true', default=False)
    parser.add_argument('--no_result', '-n', action='store_true', default=False, help='If true, will not write result files')
    parser.add_argument('--no_load', '-o', action='store_true', default=False, help='If true, will not load at any time')
    args = parser.parse_args()
    exp_main(args)


if __name__ == '__main__':
    main()
