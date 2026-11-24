from pathlib import Path
import pandas as pd

LABELS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
LABEL_TO_ID = {v: i for i, v in enumerate(LABELS)}
SPLIT_CSV = {'train': 'train_sent_emo.csv', 'dev': 'dev_sent_emo.csv', 'test': 'test_sent_emo.csv'}
SPLIT_MEDIA = {'train': 'train_splits', 'dev': 'dev_splits_complete', 'test': 'output_repeated_splits_test'}


def read_split(root, split):
    root = Path(root)
    if split not in SPLIT_CSV:
        raise ValueError(f'Unknown split: {split}')
    path = root / SPLIT_CSV[split]
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run download_meld.py first.')
    df = pd.read_csv(path)
    for col in ('Utterance', 'Emotion', 'Dialogue_ID', 'Utterance_ID'):
        if col not in df:
            raise ValueError(f'Missing column {col} in {path}')
    df['label'] = df['Emotion'].astype(str).str.lower().map(LABEL_TO_ID)
    if df['label'].isna().any():
        raise ValueError('Unexpected emotion labels')
    df['clip'] = [str(root / SPLIT_MEDIA[split] / f'dia{d}_utt{u}.mp4') for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    df['key'] = [f'{split}:dia{d}_utt{u}' for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    return df.reset_index(drop=True)
