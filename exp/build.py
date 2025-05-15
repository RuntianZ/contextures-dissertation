import argparse
import yaml
import random
import os 
import sys
sys.path.append(os.getcwd())
from copy import deepcopy
from framework.files import makedirs, read_main_config, get_jobname_list, load_all_algs, load_single_config_yaml
from typing import Tuple


def work(recipe: list, location: list, breakpoint: int = None) -> Tuple[list, bool, bool]:
    """
    Simulate the grid search config using a tree + DFS
    location indicates the current node on the tree
    location = [1, 0] means:
      - if recipe[1] is a mixture, then at component 0
      - if recipe[1] is not a mixture, then at argument 0

    location = [1, [2, 3]] means:
      - at module recipe[1], argument 2 (which is a subrecipe), module subrecipe[3]

    If current node is recipe, returns a list of lists (each list is a single recipe)
    If current node is module, returns a list of dicts (each dict is a single module)

    Breakpoints are not accepted in mixture recipes, so won't pass the breakpoint parameter in recurrence call
    """
    # print('work:', location)
    all_algs = load_all_algs()
    has_eval = False
    reach_breakpoint = False
    a = recipe
    state = 'recipe'
    for x in location:
        match state:
            case 'recipe':
                assert isinstance(x, int)
                assert x < len(a)
                a = a[x]
                state = 'module'
            case 'module':
                if isinstance(x, int):
                    assert x < len(a)
                    md_name = a['module']
                    arg_name = list(a.keys())[x]
                    assert arg_name != 'switch' and arg_name != 'module'
                    a = a[arg_name]
                    assert md_name == 'mixture'
                    assert arg_name != 'mode'
                    state = 'recipe'
                else:
                    assert isinstance(x, list)
                    assert len(x) == 2
                    args = list(a.keys())
                    a = a[args[x[0]]]
                    assert isinstance(a, list)
                    assert x[1] < len(a) 
                    a = a[x[1]]
                    state = 'module'
            case _:
                assert False, 'Bug: Unknown state'
    # print('state:', state)

    # Reach the current node
    match state:
        case 'recipe':
            ans = [[]]
            for i in range(len(a)):
                loc = location + [i]
                md = a[i]
                assert isinstance(md, dict)
                # print('md =', md)
                all_mds, h, rb = work(recipe, loc, breakpoint)  # A list of dicts, each dict is a possible setup of the module
                has_eval = has_eval or h
                new_ans = []
                for x in ans:
                    new_ans += [(x + [y]) for y in all_mds]
                ans = ans + new_ans if 'switch' in md else new_ans
                if rb:
                    # this could only happen when there is a mixture module, with a breakpoint in it; we should stop here
                    assert md['module'] == 'mixture'
                    # print('breakpoint in mixture')
                    reach_breakpoint = True
                    break
                if not 'module' in md:
                    # Breakpoint. Stop if reached
                    # print('check: breakpoint = {}, md[breakpoint] = {}'.format(breakpoint, md['breakpoint']))
                    if breakpoint is not None and int(md['breakpoint']) == breakpoint:
                        # print('module breakpoint')
                        reach_breakpoint = True
                        break

        case 'module':
            if not 'module' in a:
                assert 'breakpoint' in a, 'Neither module nor breakpoint'
                assert not 'switch' in a, 'Breakpoints do not support switch'
                # ans = []
                # if breakpoint is not None and int(a['breakpoint']) == breakpoint:
                #     print('Breakpoint!')
                #     reach_breakpoint = True
                ans = [{'breakpoint': a['breakpoint']}]
            elif a['module'] == 'mixture':
                ans = [{'module': 'mixture'}]
                args = list(a.keys())
                for i in range(len(args)):
                    k = args[i]
                    if k == 'module' or k == 'switch':
                        continue
                    elif k == 'mode':
                        if isinstance(a[k], list):
                            new_ans = []
                            for x in ans:
                                new_ans += [(x | {k: y}) for y in a[k]]
                            ans = new_ans
                        else:
                            for j in range(len(ans)):
                                ans[j] = ans[j] | {k: a[k]}
                    else:  # This arg is a recipe
                        loc = location + [i]
                        all_rps, h, rb = work(recipe, loc, breakpoint)  # A list of lists
                        has_eval = has_eval or h
                        new_ans = []
                        for x in ans:
                            new_ans += [(x | {k: y}) for y in all_rps]
                        ans = new_ans
                        if rb:
                            # If reach breakpoint in the recipe, later recipes will not be trained
                            # print('breakpoint in mixture recipe')
                            reach_breakpoint = True
                            break

            else:
                if a['module'] == 'eval':
                    has_eval = True
                ans = [{'module': a['module']}]
                args = list(a.keys())

                sr_args = []
                if a['module'] in all_algs and 'subrecipe' in all_algs[a['module']]:
                    sr_args = all_algs[a['module']]['subrecipe']
                sa_args = []
                if a['module'] in all_algs and 'subargs' in all_algs[a['module']]:
                    sa_args = all_algs[a['module']]['subargs']

                for i in range(len(args)):
                    k = args[i]
                    if k == 'module' or k == 'switch':
                        continue

                    if k in sr_args:   # subrecipe
                        if isinstance(a[k], list):
                            has_switch = False
                            ans2 = [[]]
                            for j in range(len(a[k])):
                                md = a[k][j]
                                assert isinstance(md, dict)
                                if 'module' in md:
                                    loc = location + [[i, j]]
                                    all_mds, h, rb = work(recipe, loc, breakpoint)
                                    assert not h, "Eval module cannot be inside a module"
                                    assert not rb, "Subrecipes do not allow breakpoints"
                                    new_ans = []
                                    for x in ans2:
                                        new_ans += [(x + [y]) for y in all_mds]
                                    ans2 = ans2 + new_ans if 'switch' in md else new_ans
                                else:
                                    assert 'switch' in md, "Neither module nor switch encountered inside a module"
                                    has_switch = True
                            new_ans = [(x | {k: 'None'}) for x in ans] if has_switch else []
                            for x in ans:
                                new_ans += [(x | {k: y}) for y in ans2]
                            ans = new_ans

                        else:
                            for j in range(len(ans)):
                                ans[j] = ans[j] | {k: a[k]}
                    
                    elif k in sa_args:   # subargs
                        # print('subargs: {}'.format(k))
                        if isinstance(a[k], list):
                            ans2 = [[]]
                            for j in range(len(a[k])):   # a[k][j] is a list of dict, each dict is the args of one learner
                                ans3 = [{}]
                                if not isinstance(a[k][j], dict):
                                    a[k][j] = {'dummy_arg': None}
                                for argk in a[k][j]:
                                    if isinstance(a[k][j][argk], list):
                                        new_ans = []
                                        for x in ans3:
                                            new_ans += [(x | {argk: y}) for y in a[k][j][argk]]
                                        ans3 = new_ans
                                    else:
                                        ans3 = [(x | {argk: a[k][j][argk]}) for x in ans3]
                                # print(f'ans3 = {ans3}')
                                new_ans = []
                                for x in ans2:
                                    new_ans += [(x + [y]) for y in ans3]
                                ans2 = new_ans
                                # print(f'ans2 = {ans2}')
                            # At this point, ans2 is a list of lists (each list is a possible combination, each item is a dict, that is config of a learner)
                            new_ans = []
                            for x in ans:
                                new_ans += [(x | {k: y}) for y in ans2]
                            ans = new_ans

                        else:
                            for j in range(len(ans)):
                                ans[j] = ans[j] | {k: a[k]}                            

                    else:
                        if isinstance(a[k], list):
                            new_ans = []
                            for x in ans:
                                new_ans += [(x | {k: y}) for y in a[k]]
                            ans = new_ans
                        else:
                            for j in range(len(ans)):
                                ans[j] = ans[j] | {k: a[k]}
                
    return ans, has_eval, reach_breakpoint


def build_main(args, return_config=False):
    """
    If return_recipes, return a dict, which is equivalent to reading a run config yaml file
    """
    llist = get_jobname_list()
    config = read_main_config(args.yaml_file)
    all_algs_recipes = {}
    cnt = 0
    double_breakpoint = getattr(args, 'double_breakpoint', None)
    assert args.breakpoint is None or double_breakpoint is None
    for alg_name in config['recipe']:
        assert not '/' in alg_name, 'Algorithm name should not contain "/"'
        recipe = config['recipe'][alg_name]
        assert not alg_name in all_algs_recipes, f'Repeated algorithm: {alg_name}'
        all_recipes, has_eval, rb = work(recipe, [], args.breakpoint)  # A list of lists, each is a list of modules (dicts)
        if rb:
            print('! The breakpoint is triggered. This could lead to fewer recipes.')
        assert len(all_recipes) > 0, f'The recipe list of {alg_name} is empty'
        all_algs_recipes[alg_name] = all_recipes
        print('Algorithm {} has {} recipes'.format(alg_name, len(all_recipes)))
        if not has_eval and args.breakpoint is None:
            print('!!! WARNING: Algorithm {} has no eval module, and you are not using a breakpoint. You might forget the eval module.'.format(alg_name))
        cnt += len(all_recipes)
    print('Total: {} recipes'.format(cnt))
    all_rps = []
    for alg_name in all_algs_recipes:
        all_rps += [('{}/recipe {}'.format(alg_name, i), all_algs_recipes[alg_name][i]) for i in range(len(all_algs_recipes[alg_name]))]
    
    if return_config:
        a = {}
        assert 'defaults' in config
        # assert 'result_file' in config['defaults']
        a['defaults'] = deepcopy(config['defaults'])
        if 'groups' in config:
            a['groups'] = deepcopy(config['groups'])
        if 'datasets' in config:
            a['datasets'] = deepcopy(config['datasets'])
        a['recipe'] = {x[0]: x[1] for x in all_rps}
        if 'results_folder' in config:
            a['results_folder'] = config['results_folder']
        return a

    magic_number = random.randint(0, 99999999)
    ans = []
    random.shuffle(all_rps)
    num_files = args.num_files.replace(' ', '')
    num_files = num_files.split(',')
    for i in range(len(num_files)):
        num_files[i] = int(num_files[i])
    n = sum(num_files)
    results_folder = config['results_folder']
    if args.results_subfolder is not None:
        results_folder = os.path.join(results_folder, args.results_subfolder)
    for i in range(n):
        a = {}
        if 'defaults' in config:
            a['defaults'] = deepcopy(config['defaults'])
        else:
            a['defaults'] = {}
        a['defaults']['result_file'] = os.path.join(results_folder, f"{args.job_name}{llist[i]}.csv")
        if not 'magic_number' in a['defaults']:
            a['defaults']['magic_number'] = magic_number
        if 'groups' in config:
            a['groups'] = deepcopy(config['groups'])
        if 'datasets' in config:
            a['datasets'] = deepcopy(config['datasets'])
        left = i * cnt // n
        right = (i + 1) * cnt // n
        print('Group {}: {} recipes'.format(i, right - left))
        # print(left, right)
        a['recipe'] = {x[0]: x[1] for x in all_rps[left: right]}
        ans.append(a)

    partitions = args.partition.replace(' ', '')
    partitions = partitions.split(',')
    part_list = []
    np = len(partitions)

    if len(num_files) == 1:
        j = 0
        for i in range(n):
            part_list.append(partitions[j])
            if (j + 1) * n // np <= i + 1:
                j += 1
    else:
        assert np == len(num_files)
        for j in range(np):
            for i in range(num_files[j]):
                part_list.append(partitions[j])
    # print(part_list)

    makedirs(args.folder)
    startn = os.path.join(args.folder, "start.sh")
    exclude_nodes = getattr(args, 'exclude', None)
    with open(startn, 'w') as fstart:
        for i in range(n):
            fn = os.path.join(args.folder, f"{llist[i]}.yaml")
            with open(fn, 'w') as f:
                yaml.dump(ans[i], f, default_flow_style=False)
            shn = os.path.join(args.folder, f"{llist[i]}.sh")
            with open(shn, 'w') as f:
                jn = args.job_name + llist[i]
                f.write('#!/bin/bash\n\n')
                f.write(f'#SBATCH --job-name={jn}\n')
                f.write(f'#SBATCH --output={args.outputs_folder}/{jn}.out\n')
                f.write(f'#SBATCH --error={args.outputs_folder}/{jn}.err\n')
                f.write(f'#SBATCH --partition={part_list[i]}\n#SBATCH --nodes=1\n')
                if args.partition == 'preempt':
                    f.write('#SBATCH --requeue\n')   # Allow preempted jobs to requeue
                f.write('#SBATCH --tasks-per-node=1\n')
                f.write(f'#SBATCH --cpus-per-task={args.cpu}\n')
                if args.gpu is not None:
                    f.write(f'#SBATCH --gres=gpu:{args.gpu}:1\n')
                if args.nodelist is not None:
                    nodelist = args.nodelist.split(',')
                    nodei = i * len(nodelist) // n
                    f.write(f'#SBATCH --nodelist={nodelist[nodei]}\n')
                    # f.write(f'#SBATCH --nodelist={args.nodelist}\n')
                if exclude_nodes is not None:
                    f.write(f'#SBATCH --exclude={exclude_nodes}\n')
                f.write(f'#SBATCH --mem={args.mem}G\n')
                f.write(f'#SBATCH --time={args.hours}:00:00\n')
                f.write(f'#SBATCH --mail-type=END\n#SBATCH --mail-user={args.email}\n')
                f.write(f'\nsource ~/.bashrc\nconda init\nconda activate {args.conda}\n')
                f.write(f'cd {args.root}\npwd\nnvidia-smi\nulimit -n 16384\n\nexport PYTHONPATH={args.root}\n')
                bp_str = '' if args.breakpoint is None else f'-b {args.breakpoint}'
                soe_str = '-s' if args.stop_on_error else ''
                bash_str = f'python main.py {fn} -l {args.logging_level} {bp_str} {soe_str}\n'
                if double_breakpoint is not None:
                    bp_str = f'-b {double_breakpoint}'
                    bash_str = f'python main.py {fn} -l {args.logging_level} {bp_str} {soe_str}\n' + bash_str
                f.write(bash_str)
            runn = os.path.join(args.folder, f"run{llist[i]}.sh")
            with open(runn, 'w') as f:
                jn = args.job_name + llist[i]
                f.write('#!/bin/bash\n\n')
                f.write(f'if squeue -h -u "{args.username}" -n "{jn}" | grep -q .; then\n')
                f.write(f'    echo "The job {jn} is already running. Exiting."\n')
                f.write(f'else\n')
                f.write(f'    sbatch {shn}\n')
                f.write('fi\n\n')
            fstart.write(f'bash {runn}\n')
    if args.breakpoint is not None:
        print(f"Using breakpoint: {args.breakpoint}")
    print(f"Start the experiment using {startn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_file', type=str)
    parser.add_argument('--num_files', '-n', type=str, default='1')
    parser.add_argument('--folder', '-f', type=str, default='sweep')
    parser.add_argument('--job_name', '-j', type=str, default='job')
    parser.add_argument('--results_subfolder', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--cpu', '-c', type=int, default=4)
    parser.add_argument('--nodelist', '-w', type=str)
    parser.add_argument('--exclude', '-x', type=str, default='babel-2-13')
    parser.add_argument('--breakpoint', '-b', type=int)
    parser.add_argument('--stop_on_error', '-s', action='store_true', default=False)
    # parser.add_argument('--train', '-t', default=False, action='store_true')
    parser.add_argument('--mem', '-m', default=32, type=int)
    parser.add_argument('--hours', default=48, type=int)
    parser.add_argument('--email', '-e', type=str)
    parser.add_argument('--root', '-r', type=str)
    parser.add_argument('--conda', '-a', type=str)
    parser.add_argument('--logging_level', '-l', type=str, default='INFO', help='DEBUG/INFO/WARNING/ERROR')
    parser.add_argument('--outputs_folder', '-o', type=str)
    parser.add_argument('--username', '-u', type=str)
    parser.add_argument('--partition', '-p', type=str, default='general', help='general, debug or preempt')
    parser.add_argument('--double_breakpoint', '-d', type=int, help='If set, will generate two commands in the bash file, first with breakpoint then without')
    args = parser.parse_args()
    env_config = load_single_config_yaml('config/env.yaml')
    if args.username is None:
        args.username = env_config['username']
    if args.email is None:
        args.email = env_config['email']
    if args.conda is None:
        args.conda = env_config['conda']
    if args.root is None:
        args.root = env_config['root']
    if args.outputs_folder is None:
        args.outputs_folder = env_config['outputs_folder']
    build_main(args)


if __name__ == '__main__':
    main()
