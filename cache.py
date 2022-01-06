# from https://github.com/google/jax/issues/2116#issuecomment-580322624
#from jax.tree_util import pytree
import pickle
from pathlib import Path
from typing import Union

suffix = '.pickle'

def get_or_generate_data(path_template: str, fn, *args):
    path = path_template % args
    path = Path(path)
    if not path.is_file():
        print('Generating data...')
        data = fn(*args)
        print('... Saving data...')
        save(data, path)
    else:
        print('Loading data...')
        data = load(path)
    print('... done!')
    return data

def save(data, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise CacheFileExistsError()
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path: Union[str, Path]):# -> pytree:
    path = Path(path)
    if not path.is_file():
        raise CacheNotFoundError()
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

class CacheFileExistsError(Exception):
    pass

class CacheNotFoundError(Exception):
    pass
