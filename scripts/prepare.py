import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from emobot.data import read_split
from emobot.features import media_features
from emobot.model import TextEncoder
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/MELD.Raw')
    p.add_argument('--cache',default='cache')
    p.add_argument('--splits',nargs='+',choices=['train','dev','test'],default=['train','dev','test'])
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--limit',type=int,default=0,help='Debug only: first N rows per split; never use for final results')
    args=p.parse_args()
    cache=Path(args.cache); cache.mkdir(parents=True,exist_ok=True)
    encoder=TextEncoder('cuda' if torch.cuda.is_available() else 'cpu')
    for split in args.splits:
        target=cache/f'{split}.npz'
        if target.exists():
            print(f'Skipping existing {target} (delete to rebuild)'); continue
        df=read_split(args.data,split)
        if args.limit: df=df.iloc[:args.limit].copy()
        print(f'Encoding {split}: {len(df)} rows, device={encoder.device}',flush=True)
        texts=encoder.encode(df.Utterance.fillna('').astype(str).tolist(),args.batch_size)
        media=np.stack([media_features(path) for path in tqdm(df.clip,desc=f'{split} media')])
        np.savez_compressed(target,text=texts.astype('float32'),media=media,
                            labels=df.label.to_numpy(dtype='int64'),keys=df.key.to_numpy(dtype=str),
                            utterances=df.Utterance.fillna('').astype(str).to_numpy(dtype=str))
        print(f'Saved {target}; media availability: audio={media[:,-2].sum():.0f}, video={media[:,-1].sum():.0f}')

if __name__=='__main__': main()
