"""OracleShield World Model trainer.

The supplied NSL-KDD workbook has no timestamps, so its row order is used only as a
reproducible prototype sequence. For the final SIH benchmark, pass a timestamped
CIC-IDS2018/CTU-13/PCAP-derived dataset to the same state builder.
"""
import argparse, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from oracle_shield_world_model import build_state_series, STATE_NAMES, WorldModel

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)

p=argparse.ArgumentParser()
p.add_argument('--data',default='Copy of DOC-20260825-WA0002.xlsx')
p.add_argument('--out',default='world_model.pt')
p.add_argument('--meta',default='world_model_meta.json')
p.add_argument('--window',type=int,default=200)
p.add_argument('--seq',type=int,default=8)
p.add_argument('--epochs',type=int,default=8)
p.add_argument('--batch',type=int,default=64)
args=p.parse_args()

df=pd.read_excel(args.data)
train=df[df['split'].astype(str).str.lower().eq('train')].reset_index(drop=True)
test=df[df['split'].astype(str).str.lower().eq('test')].reset_index(drop=True)
train_states, train_labels=build_state_series(train,args.window)
test_states, test_labels=build_state_series(test,args.window)

scaler=StandardScaler(); train_s=scaler.fit_transform(train_states).astype('float32'); test_s=scaler.transform(test_states).astype('float32')
classes=['normal','probe','r2l','u2r','dos']; c2i={c:i for i,c in enumerate(classes)}

def make_sequences(arr,labels):
    X=[];Y=[];L=[]
    for i in range(args.seq,len(arr)):
        X.append(arr[i-args.seq:i]); Y.append(arr[i]); L.append(c2i.get(labels[i],0))
    return np.asarray(X,'float32'),np.asarray(Y,'float32'),np.asarray(L,'int64')

Xtr,Ytr,Ltr=make_sequences(train_s,train_labels); Xte,Yte,Lte=make_sequences(test_s,test_labels)
if len(Xtr)==0 or len(Xte)==0: raise SystemExit('Not enough windows for the selected window/sequence length.')

model=WorldModel(Xtr.shape[-1],hidden=64,classes=len(classes)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
reg=nn.SmoothL1Loss(); ce=nn.CrossEntropyLoss()
loader=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(Ytr),torch.tensor(Ltr)),batch_size=args.batch,shuffle=True)

for epoch in range(args.epochs):
    model.train(); total=0
    for xb,yb,lb in loader:
        opt.zero_grad(); next_state,logits,_=model(xb)
        loss=reg(next_state,yb)+0.5*ce(logits,lb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total += loss.item()*len(xb)
    print(f'epoch {epoch+1}/{args.epochs} loss={total/len(Xtr):.4f}')

model.eval()
with torch.no_grad():
    yp,logits,_=model(torch.tensor(Xte)); pred=logits.argmax(1).numpy(); mae=float(torch.mean(torch.abs(yp-torch.tensor(Yte))).item())
metrics={'stage_accuracy':accuracy_score(Lte,pred),'stage_macro_f1':f1_score(Lte,pred,average='macro',zero_division=0),'stage_macro_precision':precision_score(Lte,pred,average='macro',zero_division=0),'stage_macro_recall':recall_score(Lte,pred,average='macro',zero_division=0),'next_state_mae_standardized':mae}

ckpt={'model':model.state_dict(),'input_dim':Xtr.shape[-1],'classes':classes,'window_size':args.window,'sequence_length':args.seq,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'state_names':STATE_NAMES}
torch.save(ckpt,args.out)
with open(args.meta,'w',encoding='utf-8') as f: json.dump({'metrics':metrics,'classes':classes,'state_names':STATE_NAMES,'window_size':args.window,'sequence_length':args.seq,'data':args.data,'temporal_note':'NSL-KDD has no timestamps; row order is used only as a reproducible prototype sequence.'},f,indent=2)
print(json.dumps(metrics,indent=2)); print(f'Saved {args.out} and {args.meta}')
