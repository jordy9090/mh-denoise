import argparse, json, random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]
DIM2ID = {d:i for i,d in enumerate(DIMS)}


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex,*names,default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x):
    x=str(x).strip().lower().replace("-","_").replace(" ","_")
    return x if x in DIM2ID else "overall_quality"


def label_vec(ex):
    y=torch.zeros(len(DIMS),dtype=torch.float)
    if "target_dimensions" in ex and isinstance(ex["target_dimensions"], list):
        dims=[norm_dim(x) for x in ex["target_dimensions"]]
    else:
        dims=[norm_dim(get_field(ex,"target_dimension","dimension","violated_dimension"))]
    for d in dims:
        y[DIM2ID[d]]=1.0
    return y


def build_text(ex):
    q=get_field(ex,"question","query","user_question")
    u=get_field(ex,"unsafe_response","corrupted_response","bad_response")
    return (
        "Question:\n"+q.strip()+
        "\n\nUnsafe response:\n"+u.strip()+
        "\n\nTask: identify all violated mental-health response quality dimensions."
    )


class RouterDS(Dataset):
    def __init__(self,path,tok,max_len=512):
        self.rows=read_jsonl(path)
        self.tok=tok
        self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        ex=self.rows[idx]
        enc=self.tok(build_text(ex),truncation=True,max_length=self.max_len,padding=False)
        return {"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"],"labels":label_vec(ex)}


class Collator:
    def __init__(self,tok): self.tok=tok
    def __call__(self,batch):
        m=max(len(x["input_ids"]) for x in batch)
        pad=self.tok.pad_token_id
        ids=[]; masks=[]; labels=[]
        for x in batch:
            n=m-len(x["input_ids"])
            ids.append(x["input_ids"]+[pad]*n)
            masks.append(x["attention_mask"]+[0]*n)
            labels.append(x["labels"])
        return {
            "input_ids":torch.tensor(ids,dtype=torch.long),
            "attention_mask":torch.tensor(masks,dtype=torch.long),
            "labels":torch.stack(labels)
        }


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    loss_fn=nn.BCEWithLogitsLoss()
    losses=[]; exact=0; top1=0; total=0
    for batch in loader:
        labels=batch.pop("labels").to(device)
        batch={k:v.to(device) for k,v in batch.items()}
        logits=model(**batch).logits.float()
        loss=loss_fn(logits,labels)
        if torch.isfinite(loss): losses.append(float(loss.item()))
        probs=torch.sigmoid(logits)
        pred=(probs>=0.5).float()
        exact += int((pred==labels).all(dim=1).sum().item())
        top1 += int((probs.argmax(dim=1)==labels.argmax(dim=1)).sum().item())
        total += labels.size(0)
    model.train()
    return sum(losses)/max(1,len(losses)), exact/max(1,total), top1/max(1,total)


def save(model,tok,out):
    out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    with open(out/"dims.json","w",encoding="utf-8") as f:
        json.dump(DIMS,f,indent=2)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_file",required=True)
    ap.add_argument("--valid_file",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--model",default="bert-base-uncased")
    ap.add_argument("--batch_size",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--eval_every",type=int,default=25)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token or tok.unk_token
    model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=len(DIMS),problem_type="multi_label_classification")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train=RouterDS(args.train_file,tok)
    valid=RouterDS(args.valid_file,tok)
    tl=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=Collator(tok),num_workers=2)
    vl=DataLoader(valid,batch_size=args.batch_size,shuffle=False,collate_fn=Collator(tok),num_workers=2)

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01,eps=1e-8)
    total=len(tl)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,int(0.1*total),total)
    loss_fn=nn.BCEWithLogitsLoss()

    print("train rows:",len(train),"valid rows:",len(valid),"dims:",DIMS)
    best=999; step=0
    model.train()
    for ep in range(args.epochs):
        pbar=tqdm(tl,desc=f"epoch {ep+1}/{args.epochs}")
        for batch in pbar:
            labels=batch.pop("labels").to(device)
            batch={k:v.to(device) for k,v in batch.items()}
            logits=model(**batch).logits.float()
            loss=loss_fn(logits,labels)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss skip")
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step+=1
            pbar.set_postfix({"loss":round(float(loss.item()),4),"step":step})
            if step%args.eval_every==0:
                ev,exact,top1=evaluate(model,vl,device)
                print(f"[eval] step={step} loss={ev:.4f} exact={exact:.4f} top1={top1:.4f}")
                if ev<best:
                    best=ev; save(model,tok,out/"best"); print("[save] best ->",out/"best")
    save(model,tok,out/"final")
    print("[done] best:",best)
    print("[done] final ->",out/"final")


if __name__=="__main__":
    main()
