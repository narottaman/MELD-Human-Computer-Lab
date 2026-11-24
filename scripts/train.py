import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import f1_score,accuracy_score
from emobot.model import FusionNet,ENCODER
from emobot.data import LABELS


def load(path):
    with np.load(path) as z: return {k:z[k].copy() for k in z.files}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',default='cache')
    p.add_argument('--out',default='artifacts')
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train=load(Path(args.cache)/'train.npz'); dev=load(Path(args.cache)/'dev.npz')
    assert not (set(train['keys']) & set(dev['keys'])), 'Train/dev overlap!'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Training on',device)
    mean=train['media'].mean(axis=0); std=np.maximum(train['media'].std(axis=0),1e-4)
    def tensors(data):
        return (torch.tensor(data['text'],dtype=torch.float32),
                torch.tensor((data['media']-mean)/std,dtype=torch.float32),
                torch.tensor(data['labels'],dtype=torch.long))
    loader=DataLoader(TensorDataset(*tensors(train)),batch_size=args.batch_size,shuffle=True)
    dtext,dmedia,dy=[x.to(device) for x in tensors(dev)]
    counts=np.bincount(train['labels'],minlength=7)
    weights=(counts.sum()/(7*np.maximum(counts,1)))**.5
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    model=FusionNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    best=-1
    history=[]
    for epoch in range(args.epochs):
        model.train();total=0
        for tx,mx,y in loader:
            tx,mx,y=tx.to(device),mx.to(device),y.to(device)
            optimizer.zero_grad();loss=criterion(model(tx,mx),y);loss.backward();optimizer.step();total+=loss.item()*len(y)
        model.eval()
        with torch.inference_mode(): pred=model(dtext,dmedia).argmax(-1).cpu().numpy()
        macro=f1_score(dev['labels'],pred,labels=range(7),average='macro',zero_division=0)
        acc=accuracy_score(dev['labels'],pred)
        history.append({'epoch':epoch+1,'train_loss':total/len(train['labels']),'dev_macro_f1':float(macro),'dev_accuracy':float(acc)})
        print(f'epoch {epoch+1}: loss={total/len(train["labels"]):.4f} dev_macro_f1={macro:.4f} dev_acc={acc:.4f}',flush=True)
        if macro>best:
            best=macro
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'encoder':ENCODER,'media_mean':mean,'media_std':std,
                        'labels':LABELS,'seed':args.seed,'dev_macro_f1':float(best)},out/'best.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    print('Best dev macro F1:',best,'checkpoint:',out/'best.pt')

if __name__=='__main__':main()
