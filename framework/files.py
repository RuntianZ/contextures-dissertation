import os
import urllib.request
import tempfile
from urllib.error import HTTPError, URLError
import sys
import time
import ssl
import tempfile
import tarfile
from tqdm import tqdm
import shutil
import torch
import yaml
import logging
import hashlib

ssl._create_default_https_context = ssl._create_unverified_context
_ckpt_max_id = 5
_ssh_client = None
_ssh_installed = True

try:
    import paramiko
    from paramiko.ssh_exception import NoValidConnectionsError, AuthenticationException, SSHException
    from scp import SCPClient
except ImportError:
    _ssh_installed = False


def read_main_config(path: str) -> dict:
    with open(path, 'r') as f:
        contents = f.readlines()
    ans = ''
    for c in contents:
        left = c.find('[')
        right = c.rfind(']')
        if left == -1 or right == -1 or right < left:
            ans += c
            continue
        c0 = c[:left] + '"' + c[left:right+1] + '"' + c[right+1:]
        ans += c0
    return yaml.safe_load(ans)
    

def get_jobname_list() -> list:
    a = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 
         'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    llist = []
    for x in a:
        for y in a:
            llist.append(x + y)
    llist = a + llist
    return llist


def hash_path(path: str) -> str:
    a = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    vocab = [f'0{x}' for x in a] + [f'1{x}' for x in a] + [f'2{x}' for x in a] + [f'3{x}' for x in a] + [f'4{x}' for x in a]
    b = list(hashlib.md5(path.encode('utf-8')).digest())
    c = [vocab[j] for j in b]
    return ''.join(c)


def save_ckpt(state_dict: dict, path: str, device, train_config: dict, logger: logging.Logger) -> None:
    """
    path is raw path, needs to change it to its short form
    """
    logger.debug(f'save_ckpt path = {path}')
    ckpt_folder = train_config['ckpt_folder']
    sha256_hash = hashlib.sha256()
    sha256_hash.update(path.encode('utf-8'))
    hex_digest = sha256_hash.hexdigest()
    state_dict['sha256'] = hex_digest
    state_dict['magic_number'] = train_config['magic_number']
    cnt = 0
    while cnt <= _ckpt_max_id:
        p = f'{path}_{cnt}'
        p = hash_path(p) + '.pth'
        local_path = os.path.join(ckpt_folder, p)
        if not os.path.exists(local_path):
            make_parent_dir(local_path)
            torch.save(state_dict, local_path)
            logger.info(f'Checkpoint successfully saved to {local_path}')
            return
        cnt += 1 
    logger.info('Failed to save checkpoint. All hashes have been used')



def load_ckpt(path: str, device, train_config: dict, logger: logging.Logger) -> dict:
    """
    Load a ckpt locally or on SSH server. If does not exist, returns None
    Specify SSH server in config/ssh.yaml
    """
    logger.debug(f'load_ckpt path = {path}')
    data = None
    use_ssh = train_config.get('use_ssh', False)

    all_folders = [train_config['ckpt_folder']]
    if 'backup_folders' in train_config:
        all_folders = all_folders + train_config['backup_folders']
    # logger.info('all_folders = {}'.format(all_folders))

    cnt = 0
    while cnt <= _ckpt_max_id:
        for ckpt_folder in all_folders:
            p = f'{path}_{cnt}'
            p = hash_path(p) + '.pth'
            local_path = os.path.join(ckpt_folder, p)
            # logger.info('load local_path = {}'.format(local_path))
            if os.path.exists(local_path):
                logger.debug('Checkpoint exists at {}'.format(local_path))
                f = None
                try:
                    f = open(local_path, 'rb')
                    data = torch.load(f, map_location=device)
                    f.close()
                except (EOFError, OSError, RuntimeError):
                    logger.debug('Broken checkpoint')
                    if (f is not None) and (not f.closed):
                        f.close()
                    os.remove(local_path)
                    data = None
            if data is not None and isinstance(data, dict) and 'sha256' in data:
                sha256_hash = hashlib.sha256()
                sha256_hash.update(path.encode('utf-8'))
                hex_digest = sha256_hash.hexdigest()
                if data['sha256'] == hex_digest:
                    should_delete = train_config.get('delete_ckpt', False)
                    magic_number = data.get('magic_number', -1)
                    if should_delete and magic_number != train_config['magic_number']:
                        os.remove(local_path)
                        logger.info(f'Deleted: {local_path}')
                        return None
                    logger.info(f'Checkpoint successfully loaded from {local_path}')
                    return data
        cnt += 1
        
    if data is not None:
        return data
    else:
        logger.debug('Checkpoint not found locally')
        global _ssh_client, _ssh_installed
        use_ssh = use_ssh and _ssh_installed
        if use_ssh:
            if _ssh_client is None:
                try:
                    _ssh_client = paramiko.SSHClient()
                    _ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())     
                    env_config = load_single_config_yaml('config/env.yaml')
                    _ssh_client.connect(env_config["ssh_server"], port=22, username=env_config["ssh_username"], password=env_config["ssh_password"])
                except (NoValidConnectionsError, AuthenticationException, SSHException, TimeoutError, EOFError, OSError):
                    logger.debug('SSH server connection failed')
                    _ssh_client = None

            if _ssh_client is not None:
                try:
                    for ssh_root in train_config["ssh_root"]:
                        cnt = 0
                        while cnt <= _ckpt_max_id:
                            p1 = f'{path}_{cnt}'
                            p1 = hash_path(p1) + '.pth'
                            fpath = os.path.join(ssh_root, p1)
                            # logger.info('fpath = {}'.format(fpath))
                            stdin, stdout, stderr = _ssh_client.exec_command(f"(ls {fpath} && echo yes) || echo no")
                            opt = stdout.readlines()
                            if opt[-1].startswith("yes"):
                                logger.debug('Checkpoint exists on SSH server, loading...')
                                with tempfile.TemporaryDirectory() as tmpdirname:
                                    scp = SCPClient(_ssh_client.get_transport())
                                    scp.get(fpath, tmpdirname)
                                    fn = os.path.basename(fpath)
                                    fn = os.path.join(tmpdirname, fn)
                                    f = None
                                    try:
                                        f = open(fn, 'rb')
                                        data = torch.load(f, map_location=device)
                                    except (EOFError, OSError, RuntimeError):
                                        logger.debug('Broken checkpoint on the server')
                                        data = None
                                    finally:
                                        if (f is not None) and (not f.closed):
                                            f.close()
                            cnt += 1
                            if data is not None:
                                break
                        if data is not None:
                            logger.info(f'Checkpoint successfully loaded from SSH server, path: {fpath}')
                            break
                    if data is None:
                        logger.debug('Checkpoint not found on SSH server')
                except (NoValidConnectionsError, AuthenticationException, SSHException, TimeoutError, EOFError, OSError):
                    logger.debug('SSH download failed')
        else:
            logger.debug('SSH is not used')
    return data


def download_dataset(target_folder):
    makedirs(target_folder)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, 'datasets.tar.gz')
        download_and_merge('https://github.com/RuntianZ/tabular-datasets/raw/main/datasets.tar.gz', filename, 159)
        with tarfile.open(filename) as f:
            f.extractall(target_folder)
    

_load_config_yaml_cache = {}
def load_config_yaml(path: str) -> dict:
    if path in _load_config_yaml_cache:
        return _load_config_yaml_cache[path]
    ans = None
    retries = 0
    while ans is None:
        try:
            ans = {}
            with open(path, 'r') as f:
                file_list = yaml.safe_load(f)
                assert isinstance(file_list, list)
                for file in file_list:
                    assert os.path.exists(file)
                    with open(file, 'r') as f1:
                        algs = yaml.safe_load(f1)
                        assert isinstance(algs, dict)
                        for k in algs:
                            assert not k in ans, f"'{k}' appears more than once in the config"
                        ans = ans | algs
        except:
            # Sometimes during file updates (such as git pull), this could fail, so we allow retry
            retries += 1
            time.sleep(1)
            if retries >= 10:
                _load_config_yaml_cache[path] = {}
                return {}
            ans = None
    assert ans is not None
    _load_config_yaml_cache[path] = ans
    return ans


def load_single_config_yaml(path: str) -> dict:
    if path in _load_config_yaml_cache:
        return _load_config_yaml_cache[path]
    ans = None
    retries = 0
    while ans is None:
        try:
            with open(path, 'r') as f:
                ans = yaml.safe_load(f)
            assert isinstance(ans, dict)
        except:
            retries += 1
            time.sleep(1)
            if retries >= 10:
                _load_config_yaml_cache[path] = {}
                return {}
            ans = None
    assert ans is not None
    _load_config_yaml_cache[path] = ans
    return ans


def load_all_algs() -> dict:
    return load_config_yaml('config/modules.yaml')


def load_all_data_groups() -> dict:
    return load_config_yaml('config/data_groups.yaml')


################################
def makedirs(directory):
    if not os.path.exists(directory):
        p = True
        retries = 0
        while p:
            try:
                os.makedirs(directory, exist_ok=True)
                p = False
            except:
                retries += 1
                time.sleep(1)
                p = True
        



def make_parent_dir(path):
    dirname = os.path.dirname(os.path.abspath(path))
    makedirs(dirname)


def handle_long_filename(fn: str, limit: int = 127) -> list:
    """
    For example, for aaabbbccc, if limit=3, then will return aaa/bbb/ccc
    """
    ans = []
    while len(fn) > limit:
        k = limit
        while fn[k] in ['[', ']', '_', '-', '.']:
            k -= 1
            assert k > 0
        ans.append(fn[:k])
        fn = fn[k:]
        # ans = ans + fn[:k] + '/'
        # fn = fn[k:]
    ans.append(fn)
    # ans = ans + fn
    return ans


def get_path_from_list(names: list) -> str:
    ans = []
    for x in names:
        ans = ans + handle_long_filename(x)
    return os.path.join(*ans)


def download(url, localpath):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    makedirs(os.path.dirname(os.path.abspath(localpath)))
    for i in range(20):
        try:
            urllib.request.urlretrieve(url, localpath)
            return 0
        except (HTTPError, URLError) as e:
            sys.stderr.write('HTTPError. Retry {}\n'.format(i + 1))
            time.sleep(5)
    raise RuntimeError('HTTPError occurs 20 times')


def merge_file(fromfile, tofile):
    if not os.path.isdir(os.path.dirname(os.path.abspath(tofile))):
        os.makedirs(os.path.dirname(os.path.abspath(tofile)))
    partnum = 1
    with open(tofile, 'wb') as fout:
        while True:
            fname = '{}.part{}'.format(fromfile, partnum)
            if not os.path.exists(fname):
                break
            with open(fname, 'rb') as fin:
                chunk = fin.read()
                fout.write(chunk)
            partnum += 1
        partnum -= 1
    return partnum


def download_and_merge(url, localpath, part_num):
    tmp = tempfile.TemporaryDirectory()
    if part_num == 0:
        download(url, localpath)
    else:
        for i in tqdm(range(part_num)):
            file_url = '{}.part{}'.format(url, i + 1)
            local_pth = '{}/tmp.part{}'.format(tmp.name, i + 1)
            download(file_url, local_pth)
        merge_file('{}/tmp'.format(tmp.name), localpath)


def split(fromfile, tofile, chunksize=20000000):
    todir = os.path.dirname(os.path.abspath(tofile))
    if not os.path.isdir(todir):
        os.makedirs(todir)
    partnum = 0
    with open(fromfile, 'rb') as fin:
        while True:
            chunk = fin.read(chunksize)
            if not chunk:
                break
            partnum += 1
            fname = '{}.part{}'.format(tofile, partnum)
            with open(fname, 'wb') as fout:
                fout.write(chunk)
    return partnum



