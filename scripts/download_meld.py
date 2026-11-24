"""Download official MELD raw release; expect ~11 GB download and substantial disk usage."""
import argparse
import subprocess
import tarfile
from pathlib import Path

URLS = ['https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz',
        'https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz']

def safe_extract(tf, dest):
    base = dest.resolve()
    for member in tf.getmembers():
        target = (base / member.name).resolve()
        if not target.is_relative_to(base) or member.issym() or member.islnk():
            raise ValueError(f'Unsafe archive entry: {member.name}')
    tf.extractall(base, filter='data')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data')
    p.add_argument('--url', default=URLS[0])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / 'MELD.Raw.tar.gz'
    if not archive.exists():
        print('Downloading ~11 GB; this may take a while...', flush=True)
        subprocess.run(['wget', '-c', '--show-progress', '-O', str(archive), args.url], check=True)
    print('Extracting main archive...', flush=True)
    with tarfile.open(archive, 'r:gz') as tf:
        safe_extract(tf, out)
    root = out / 'MELD.Raw'
    if not root.exists():
        candidates = list(out.rglob('train_sent_emo.csv'))
        if not candidates:
            raise RuntimeError('Cannot locate extracted MELD CSV files')
        root = candidates[0].parent
    for split in ['train', 'dev', 'test']:
        nested = root / f'{split}.tar.gz'
        if nested.exists():
            print(f'Extracting {nested.name}...', flush=True)
            with tarfile.open(nested, 'r:gz') as tf:
                safe_extract(tf, root)
    print(f'Dataset root: {root.resolve()}')
    for split, folder in [('train','train_splits'),('dev','dev_splits_complete'),('test','output_repeated_splits_test')]:
        print(split, 'CSV:', (root/f'{split}_sent_emo.csv').exists(), 'videos:', (root/folder).exists())

if __name__ == '__main__':
    main()
