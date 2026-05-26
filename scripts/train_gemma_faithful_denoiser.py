import argparse, json, math, random, re
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def fmt_probs(p):
    if not isinstance(p,dict) or not p: return "not available"
    items=sorted(p.items(), key=lambda x:x[1], reverse=True)
    return ", ".join(f"{k}={v:.2f}" for k,v in items[:3])


def build_prompt(ex):
    q=ex["question"].strip()
    u=ex["unsafe_response"].strip()
    d=ex["condition_dim"]
    z=ex.get("z_t","").strip()
    c=ex.get("corruption_source","bridge")
    t=float(ex.get("corruption_level",1.0))
    if not z:
        z_block="No intermediate draft is provided. Rewrite directly from the unsafe response."
    else:
        z_block=z
    return (
        "Task: Rewrite the unsafe counseling response into a safe response.\n"
        "Rules: give only the final response. Do not include analysis, labels, probabilities, or hidden reasoning.\n"
        "Avoid diagnosis, prescriptions, blame, and unsupported clinical claims.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Violation dimension:\n{d}\n\n"
        f"Intermediate draft z_t:\n{z_block}\n\n"
        f"Corruption source: {c}; level: {t:.2f}\n\n"
        "Final safe response:\n"
    )


def find_span_ranges(target, spans):
    ranges=[]
    low=target.lower()
    for sp in spans:
        s=str(sp.get("safe_span","")).strip()
        if not s: continue
        pos=low.find(s.lower())
        if pos>=0:
            ranges.append((pos,pos+len(s),float(sp.get("risk_score",0.5))))
    return ranges


class FaithfulDataset(Dataset):
    def __init__(self,path,tok,max_source_len=1024,max_target_len=256,alpha=1.5):
        self.rows=read_jsonl(path)
        self.tok=tok
        self.max_source_len=max_source_len
        self.max_target_len=max_target_len
        self.alpha=alpha

    def __len__(self): return len(self.rows)

    def __getitem__(self,idx):
        ex=self.rows[idx]
        prompt=build_prompt(ex)
        target=ex["safe_response"].strip()
        eos=self.tok.eos_token or ""
        if eos and not target.endswith(eos):
            target=target+eos

        pids=self.tok(prompt,add_special_tokens=True,truncation=True,max_length=self.max_source_len)["input_ids"]

        # offset mapping for token-level weights
        try:
            enc_t=self.tok(target,add_special_tokens=False,truncation=True,max_length=self.max_target_len,return_offsets_mapping=True)
            tids=enc_t["input_ids"]
            offsets=enc_t["offset_mapping"]
        except Exception:
            tids=self.tok(target,add_special_tokens=False,truncation=True,max_length=self.max_target_len)["input_ids"]
            offsets=[(0,0)]*len(tids)

        ranges=find_span_ranges(target, ex.get("target_weight_spans",[]))
        tweights=[]
        for a,b in offsets:
            w=1.0
            for s,e,r in ranges:
                if not (b <= s or a >= e):
                    w=max(w,1.0+self.alpha*r)
            tweights.append(w)

        input_ids=pids+tids
        labels=[-100]*len(pids)+tids
        attn=[1]*len(input_ids)
        weights=[0.0]*len(pids)+tweights
        ex_weight=float(ex.get("loss_weight",1.0))

        return {
            "input_ids":input_ids,
            "attention_mask":attn,
            "labels":labels,
            "token_weights":weights,
            "loss_weight":ex_weight,
        }


@dataclass
class Collator:
    tok: object
    max_len: int
    def __call__(self,batch):
        max_len=min(self.max_len,max(len(x["input_ids"]) for x in batch))
        pad=self.tok.pad_token_id
        ids=[]; masks=[]; labels=[]; tw=[]; ew=[]
        for x in batch:
            ids0=x["input_ids"][:max_len]
            masks0=x["attention_mask"][:max_len]
            lab0=x["labels"][:max_len]
            tw0=x["token_weights"][:max_len]
            n=max_len-len(ids0)
            ids.append(ids0+[pad]*n)
            masks.append(masks0+[0]*n)
            labels.append(lab0+[-100]*n)
            tw.append(tw0+[0.0]*n)
            ew.append(float(x["loss_weight"]))
        return {
            "input_ids":torch.tensor(ids,dtype=torch.long),
            "attention_mask":torch.tensor(masks,dtype=torch.long),
            "labels":torch.tensor(labels,dtype=torch.long),
            "token_weights":torch.tensor(tw,dtype=torch.float),
            "loss_weight":torch.tensor(ew,dtype=torch.float),
        }


def weighted_loss(logits,labels,token_weights,example_weights):
    shift_logits=logits[:,:-1,:].contiguous()
    shift_labels=labels[:,1:].contiguous()
    shift_tw=token_weights[:,1:].contiguous()
    vocab=shift_logits.size(-1)
    loss=F.cross_entropy(
        shift_logits.view(-1,vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none"
    ).view(shift_labels.size())
    mask=(shift_labels!=-100).float()
    weights=torch.where(mask>0, torch.clamp(shift_tw,min=1.0), torch.zeros_like(shift_tw))
    per=(loss*weights).sum(dim=1)/weights.sum(dim=1).clamp_min(1.0)
    return (per*example_weights.to(per.device)).mean()


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); losses=[]
    for batch in loader:
        labels=batch.pop("labels").to(device)
        tw=batch.pop("token_weights").to(device)
        ew=batch.pop("loss_weight").to(device)
        batch={k:v.to(device) for k,v in batch.items()}
        logits=model(**batch).logits
        loss=weighted_loss(logits,labels,tw,ew)
        if torch.isfinite(loss): losses.append(float(loss.item()))
    model.train()
    return sum(losses)/max(1,len(losses))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_file",required=True)
    ap.add_argument("--valid_file",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--model",default="google/gemma-4-E4B-it")
    ap.add_argument("--max_source_len",type=int,default=1024)
    ap.add_argument("--max_target_len",type=int,default=256)
    ap.add_argument("--batch_size",type=int,default=2)
    ap.add_argument("--grad_accum",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--eval_every",type=int,default=25)
    ap.add_argument("--save_every",type=int,default=100)
    ap.add_argument("--lora_r",type=int,default=16)
    ap.add_argument("--lora_alpha",type=int,default=32)
    ap.add_argument("--target_modules",default="linear")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"

    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=bnb,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    model=prepare_model_for_kbit_training(model)
    lcfg=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=[x.strip() for x in args.target_modules.split(",")])
    model=get_peft_model(model,lcfg)
    model.print_trainable_parameters()
    device=next(model.parameters()).device

    train=FaithfulDataset(args.train_file,tok,args.max_source_len,args.max_target_len)
    valid=FaithfulDataset(args.valid_file,tok,args.max_source_len,args.max_target_len)
    coll=Collator(tok,args.max_source_len+args.max_target_len)
    tl=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=coll,num_workers=2)
    vl=DataLoader(valid,batch_size=args.batch_size,shuffle=False,collate_fn=coll,num_workers=2)

    updates=math.ceil(len(tl)/args.grad_accum)*args.epochs
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    sched=get_linear_schedule_with_warmup(opt,int(0.06*updates),updates)
    print("train rows:",len(train),"valid rows:",len(valid),"updates:",updates)

    best=float("inf"); g=0
    model.train(); opt.zero_grad(set_to_none=True)
    for ep in range(args.epochs):
        pbar=tqdm(tl,desc=f"epoch {ep+1}/{args.epochs}")
        for step,batch in enumerate(pbar):
            labels=batch.pop("labels").to(device)
            tw=batch.pop("token_weights").to(device)
            ew=batch.pop("loss_weight").to(device)
            batch={k:v.to(device) for k,v in batch.items()}
            logits=model(**batch).logits
            loss=weighted_loss(logits,labels,tw,ew)/args.grad_accum
            if not torch.isfinite(loss):
                print("[warn] non-finite loss; skip"); opt.zero_grad(set_to_none=True); continue
            loss.backward()
            if (step+1)%args.grad_accum==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); g+=1
                pbar.set_postfix({"loss":round(float(loss.item()*args.grad_accum),4),"gstep":g})
                if g%args.eval_every==0:
                    ev=evaluate(model,vl,device)
                    print(f"[eval] step={g} loss={ev:.4f}")
                    if ev<best:
                        best=ev
                        model.save_pretrained(out/"best"); tok.save_pretrained(out/"best")
                        print("[save] best ->",out/"best")
                if g%args.save_every==0:
                    model.save_pretrained(out/f"step_{g}"); tok.save_pretrained(out/f"step_{g}")
                    print("[save] checkpoint ->",out/f"step_{g}")
    model.save_pretrained(out/"final"); tok.save_pretrained(out/"final")
    with open(out/"train_args.json","w",encoding="utf-8") as f: json.dump(vars(args),f,indent=2,ensure_ascii=False)
    print("[done] best eval loss:",best)
    print("[done] final ->",out/"final")


if __name__=="__main__":
    main()
