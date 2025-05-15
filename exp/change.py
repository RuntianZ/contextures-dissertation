import os
import sys
sys.path.append(os.getcwd())
import argparse
import re
import subprocess
import time
from framework.files import get_jobname_list

def main():
    llist = get_jobname_list()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('names', type=str)
    parser.add_argument('server', type=str)
    args = parser.parse_args()
    all_scripts = []

    x = args.names.find('-')
    if x == -1:
        left = llist.index(args.names)
        right = left
    else:
        left = llist.index(args.names[:x])
        right = llist.index(args.names[x+1:])
    for x in range(left, right+1):
        fn = os.path.join(args.folder, f'{llist[x]}.sh')
        with open(fn, 'r') as f:
            content = f.read()
        content = re.sub(r'(nodelist=).*?(\n)', r'\1{}\2'.format(args.server), content)
        with open(fn, 'w') as f:
            f.write(content)
        print(f'{fn} modified')

        sn = os.path.join(args.folder, f'run{llist[x]}.sh')
        if os.path.exists(sn):
            with open(sn, 'r') as fsn:
                content = fsn.read()
                pattern = r'The job \s*([^"]*) is already running'
                match = re.search(pattern, content)
                if match:
                    jn = match.group(1).strip()
                    # print(jn)
                    subprocess.run(f'scancel $(squeue -h -u "rzhai" -n "{jn}" -o "%i" | head -n1) 2>/dev/null && echo "Stopping job {jn}" || echo "Job {jn} is not running"', shell=True)
            all_scripts.append(f'bash {sn}')

    if len(all_scripts) > 0:
        print('Please wait...')
        time.sleep(30)
        for sc in all_scripts: 
            subprocess.run(sc, shell=True)


if __name__ == "__main__":
    main()