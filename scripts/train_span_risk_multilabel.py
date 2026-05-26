import argparse, json, random, re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


DIMS=["overall_quality","empathy","specificity","medical_advice","factual_consistency","toxicity"]
DIM2ID={d:i for i,d in enumerate(DIMS)}

PATTERNS={
"medical_advice":["diagnosis","diagnosable","disorder","borderline personality disorder","adjustment disorder","social anxiety","depression","cognitive restructuring","medication","medicine","treatment","formal diagnosis","you should","you need to","you must","requires"],
"toxicity":["too sensitive","overreact","overreacting","loosen up","big deal","mountain out of a molehill","dramatic","paranoid","crazy","your fault","probably just"],
"factual_consistency":["clearly","always","never","direct indicator","underlying","suggests","means that","drives","manifest","proves","obviously","emotional stability","resilient"],
"specificity":["things","stuff","generally","sometimes","common","normal","focus on other things","helpful approach","wellness","take care","try to handle"],
"empathy":["observe","noticing","common behavior","pattern","detached","complicated","it happens","people do"],
"overall_quality":["just","maybe","things","stuff","handle it","figure out","resolve over time","next steps yourself","complicated"]
}


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def get_field(ex,*names,default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x):
    x=str(x).strip().lower().replace("-","_").replace(" ","_")
    return x if x in DIM2ID else "overall_quality"


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+",text.strip()) if p.strip()]


def contains(text, phrase):
    return phrase.lower() in text.lower()


def build_examples(rows):
    out=[]
    for ex in rows:
        q=get_field(ex,"question","query","user_question")
        u=get_field(ex,"unsafe_response","corrupted_response","bad_response")
        y=get_field(ex,"safe_response","target_response","response")
        d=norm_dim(get_field(ex,"target_dimension","dimension","violated_dimension"))

        candidates=[]
        for s in split_sentences(u) or [u]:
            candidates.append((s,"unsafe"))
        for s in (split_sentences(y) or [y])[:4]:
            candidates.append((s,"safe"))

        for span,source in candidates:
            vec=torch.zeros(len(DIMS),dtype=torch.float)
            for dim,ps in PATTERNS.items():
                if any(contains(span,p) for p in ps):
                    vec[DIM2ID[dim]]=1.0
            # Primary synthetic dimension gets weak positive if unsafe span has any generic risky cue.
            if source=="unsafe" and vec.sum()>0:
                vec[DIM2ID[d]]=max(vec[DIM2ID[d]],1.0)
            out.append({"question":q,"span":span,"labels":vec.tolist(),"source":source})
    return out


def build_text(ex):
    return (
        "Question:\n"+ex["question"].strip()+
        "\n\nCandidate span:\n"+ex["span"].strip()+
        "\n\nTask: predict which counseling quality dimensions this span may violate."
    )


class RiskDS(Dataset):
    def __init__(self,examples,tok,max_len=384):
        self.examples=examples; self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.examples)
    def __getitem__(self,idx):
        ex=self.examples[idx]
        enc=self.tok(build_text(ex),truncation=True,max_length=self.max_len,padding=False)
        return {"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"],"labels":torch.tensor(ex["labels"],dtype=torch.float)}


class Collator:
    def __init__(self,tok): self.tok=tok
    def __call__(self,batch):
        m=max(len(x["input_ids"]) for x in batch); pad=self.tok.pad_token_id
        ids=[]; masks=[]; labels=[]
        for x in batch:
            n=m-len(x["input_ids"])
            ids.append(x["input_ids"]+[pad]*n)
            masks.append(x["attention_mask"]+[0]*n)
            labels.append(x["labels"])
        return {"input_ids":torch.tensor(ids,dtype=torch.long),"attention_mask":torch.tensor(masks,dtype=torch.long),"labels":torch.stack(labels)}


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); loss_fn=nn.BCEWithLogitsLoss(); losses=[]; f1s=[]
    tp=fp=fn=0
    for batch in loader:
        labels=batch.pop("labels").to(device)
        batch={k:v.to(device) for k,v in batch.items()}
        logits=model(**batch).logits.float()
        loss=loss_fn(logits,labels)
        if torch.isfinite(loss): losses.append(float(loss.item()))
        pred=(torch.sigmoid(logits)>=0.5).long()
        gold=labels.long()
        tp += int(((pred==1)&(gold==1)).sum().item())
        fp += int(((pred==1)&(gold==0)).sum().item())
        fn += int(((pred==0)&(gold==1)).sum().item())
    f1=2*tp/max(1,2*tp+fp+fn)
    model.train()
    return sum(losses)/max(1,len(losses)),f1


def save(model,tok,out):
    out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    with open(out/"dims.json","w",encoding="utf-8") as f: json.dump(DIMS,f,indent=2)
    with open(out/"patterns.json","w",encoding="utf-8") as f: json.dump(PATTERNS,f,indent=2,ensure_ascii=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_file",required=True)
    ap.add_argument("--valid_file",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--model",default="bert-base-uncased")
    ap.add_argument("--batch_size",type=int,default=16)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--eval_every",type=int,default=25)
    args=ap.parse_args()

    train_ex=build_examples(read_jsonl(args.train_file))
    valid_ex=build_examples(read_jsonl(args.valid_file))
    print("train span examples:",len(train_ex))
    print("valid span examples:",len(valid_ex))

    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=len(DIMS),problem_type="multi_label_classification")
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device)

    tl=DataLoader(RiskDS(train_ex,tok),batch_size=args.batch_size,shuffle=True,collate_fn=Collator(tok),num_workers=2)
    vl=DataLoader(RiskDS(valid_ex,tok),batch_size=args.batch_size,shuffle=False,collate_fn=Collator(tok),num_workers=2)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    total=len(tl)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,int(0.1*total),total)
    loss_fn=nn.BCEWithLogitsLoss()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    best=999; step=0

    for ep in range(args.epochs):
        pbar=tqdm(tl,desc=f"epoch {ep+1}/{args.epochs}")
        for batch in pbar:
            labels=batch.pop("labels").to(device)
            batch={k:v.to(device) for k,v in batch.items()}
            logits=model(**batch).logits.float()
            loss=loss_fn(logits,labels)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step+=1
            pbar.set_postfix({"loss":round(float(loss.item()),4),"step":step})
            if step%args.eval_every==0:
                ev,f1=evaluate(model,vl,device)
                print(f"[eval] step={step} loss={ev:.4f} micro_f1={f1:.4f}")
                if ev<best:
                    best=ev; save(model,tok,out/"best"); print("[save] best ->",out/"best")
    save(model,tok,out/"final")
    print("[done] best:",best)


if __name__=="__main__":
    main()
