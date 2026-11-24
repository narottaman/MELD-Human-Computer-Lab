"""Held-out test evaluation. Never fits parameters or selects checkpoints."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from emobot.model import FusionNet
from emobot.data import LABELS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',default='cache')
    p.add_argument('--checkpoint',default='artifacts/best.pt')
    p.add_argument('--out',default='artifacts/test_metrics.json')
    args=p.parse_args()
    with np.load(Path(args.cache)/'test.npz') as z:
        x=z['text'].copy(); media=z['media'].copy(); labels=z['labels'].copy(); keys=z['keys'].copy()
    with np.load(Path(args.cache)/'train.npz') as z: trainkeys=set(z['keys'].tolist())
    with np.load(Path(args.cache)/'dev.npz') as z: devkeys=set(z['keys'].tolist())
    assert not (set(keys)&trainkeys) and not (set(keys)&devkeys), 'Data leakage detected'
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=FusionNet().to(device).eval();model.load_state_dict(state['model'])
    mean=np.asarray(state['media_mean']);std=np.asarray(state['media_std'])
    start=time.perf_counter();pred=[]
    with torch.inference_mode():
        for i in range(0,len(labels),128):
            a=torch.from_numpy(x[i:i+128]).to(device)
            b=torch.from_numpy(((media[i:i+128]-mean)/std).astype('float32')).to(device)
            pred.extend(model(a,b).argmax(-1).cpu().tolist())
    elapsed=time.perf_counter()-start
    metrics={'split':'test','samples':len(labels),'accuracy':float(accuracy_score(labels,pred)),
             'macro_f1':float(f1_score(labels,pred,average='macro',labels=range(7),zero_division=0)),
             'weighted_f1':float(f1_score(labels,pred,average='weighted',labels=range(7),zero_division=0)),
             'classification_report':classification_report(labels,pred,labels=range(7),target_names=LABELS,output_dict=True,zero_division=0),
             'confusion_matrix':confusion_matrix(labels,pred,labels=range(7)).tolist(),
             'fusion_only_ms_per_utterance':1000*elapsed/max(1,len(labels)),
             'note':'Fusion-only speed excludes text embedding and audio/video extraction; benchmark full path separately.'}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(metrics,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('classification_report','confusion_matrix')},indent=2))
    print(classification_report(labels,pred,labels=range(7),target_names=LABELS,zero_division=0))
    print('Saved',path)

if __name__=='__main__':main()
