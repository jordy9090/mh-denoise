import argparse, json, math, random, re
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup


DIMS=["overall_quality","empathy","specificity","medical_advice","factual_consistency","toxicity"]


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


class MultiAspectLoRALinear(nn.Module):
    def __init__(self, base, num_aspects=6, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = getattr(base, "in_features")
        self.out_features = getattr(base, "out_features")
        self.num_aspects = num_aspects
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        # Important: this wrapper is inserted after the base model is already placed
        # by device_map="auto". New parameters must be created on the same device.
        try:
            base_param = next(base.parameters())
            device = base_param.device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep LoRA params in bf16 on GPU to match Gemma hidden states.
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.A_sh = nn.Parameter(torch.empty(r, self.in_features, device=device, dtype=dtype))
        self.B_sh = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        self.A_exp = nn.Parameter(torch.empty(num_aspects, r, self.in_features, device=device, dtype=dtype))
        self.B_exp = nn.Parameter(torch.zeros(num_aspects, self.out_features, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.A_sh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_exp, a=math.sqrt(5))
        self._gates = None

    def set_gates(self, g):
        self._gates = g

    def forward(self, x):
        base_out = self.base(x)
        xd = self.dropout(x)

        # Safety: if model/device_map moves hidden states, cast LoRA tensors at use time.
        device = xd.device
        dtype = xd.dtype

        A_sh = self.A_sh.to(device=device, dtype=dtype)
        B_sh = self.B_sh.to(device=device, dtype=dtype)
        A_exp = self.A_exp.to(device=device, dtype=dtype)
        B_exp = self.B_exp.to(device=device, dtype=dtype)

        sh = F.linear(F.linear(xd, A_sh), B_sh) * self.scaling

        if xd.dim() == 3 and self._gates is not None and self._gates.size(0) == xd.size(0):
            gates = self._gates.to(device=device, dtype=dtype)
            exp = 0
            for k in range(self.num_aspects):
                ok = F.linear(F.linear(xd, A_exp[k]), B_exp[k]) * self.scaling
                exp = exp + ok * gates[:, k].view(-1, 1, 1)
        else:
            if self._gates is None:
                gates = torch.ones(self.num_aspects, device=device, dtype=dtype) / self.num_aspects
            else:
                gates = self._gates.mean(dim=0).to(device=device, dtype=dtype)
            exp = 0
            for k in range(self.num_aspects):
                ok = F.linear(F.linear(xd, A_exp[k]), B_exp[k]) * self.scaling
                exp = exp + ok * gates[k]

        return base_out + sh + exp


def replace_with_moe_lora(model,target_names=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),r=8,alpha=16,dropout=0.05):
    count=0
    for name,module in list(model.named_modules()):
        if any(name.endswith(t) for t in target_names):
            parent_name=name.rsplit(".",1)[0]
            child_name=name.rsplit(".",1)[1]
            parent=model.get_submodule(parent_name)
            old=getattr(parent,child_name)
            if hasattr(old,"in_features") and hasattr(old,"out_features"):
                setattr(parent,child_name,MultiAspectLoRALinear(old,num_aspects=len(DIMS),r=r,alpha=alpha,dropout=dropout))
                count+=1
    print("replaced linear modules with MultiAspectLoRA:",count)
    return model


def set_all_gates(model,g):
    eps=1e-4
    tau=(g+eps)/(g+eps).sum(dim=1,keepdim=True).clamp_min(eps)
    for m in model.modules():
        if isinstance(m,MultiAspectLoRALinear):
            m.set_gates(tau)


def trainable_moe_params(model):
    return [p for n,p in model.named_parameters() if p.requires_grad]


def build_prompt(ex):
    q=ex["question"].strip()
    u=ex["unsafe_response"].strip()
    z=ex.get("z_t","").strip()
    source=ex.get("source","bridge")
    t=ex.get("t",1)
    if not z:
        z="No draft."
    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Draft:\n{z}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Response:\n"
    )


def find_ranges(target,spans):
    ranges=[]
    low=target.lower()
    for sp in spans:
        s=str(sp.get("safe_span","")).strip()
        if not s: continue
        pos=low.find(s.lower())
        if pos>=0:
            ranges.append((pos,pos+len(s),float(sp.get("risk",0.0))))
    return ranges


class FullDS(Dataset):
    def __init__(self,path,tok,max_source_len=1024,max_target_len=256,lambda_y=1.5):
        self.rows=read_jsonl(path); self.tok=tok
        self.max_source_len=max_source_len; self.max_target_len=max_target_len; self.lambda_y=lambda_y
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        ex=self.rows[idx]
        prompt=build_prompt(ex)
        target=ex["safe_response"].strip()
        if self.tok.eos_token and not target.endswith(self.tok.eos_token):
            target+=self.tok.eos_token
        pids=self.tok(prompt,add_special_tokens=True,truncation=True,max_length=self.max_source_len)["input_ids"]
        try:
            enc=self.tok(target,add_special_tokens=False,truncation=True,max_length=self.max_target_len,return_offsets_mapping=True)
            tids=enc["input_ids"]; offs=enc["offset_mapping"]
        except Exception:
            tids=self.tok(target,add_special_tokens=False,truncation=True,max_length=self.max_target_len)["input_ids"]
            offs=[(0,0)]*len(tids)
        ranges=find_ranges(target,ex.get("target_weight_spans",[]))
        tw=[]
        for a,b in offs:
            w=1.0
            for s,e,r in ranges:
                if not (b<=s or a>=e):
                    w=max(w,1.0+self.lambda_y*r)
            tw.append(w)
        ids=pids+tids
        labels=[-100]*len(pids)+tids
        token_weights=[0.0]*len(pids)+tw
        g=torch.tensor(ex["g"],dtype=torch.float)
        return {"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels,"token_weights":token_weights,"g":g}


@dataclass
class Collator:
    tok: object
    max_len: int
    def __call__(self,batch):
        m=min(self.max_len,max(len(x["input_ids"]) for x in batch))
        pad=self.tok.pad_token_id
        ids=[]; masks=[]; labels=[]; tw=[]; gs=[]
        for x in batch:
            ids0=x["input_ids"][:m]; masks0=x["attention_mask"][:m]; lab0=x["labels"][:m]; tw0=x["token_weights"][:m]
            n=m-len(ids0)
            ids.append(ids0+[pad]*n); masks.append(masks0+[0]*n); labels.append(lab0+[-100]*n); tw.append(tw0+[0.0]*n); gs.append(x["g"])
        return {
            "input_ids":torch.tensor(ids,dtype=torch.long),
            "attention_mask":torch.tensor(masks,dtype=torch.long),
            "labels":torch.tensor(labels,dtype=torch.long),
            "token_weights":torch.tensor(tw,dtype=torch.float),
            "g":torch.stack(gs)
        }


def weighted_loss(logits,labels,tw):
    # fp32 CE is more stable for custom MoE LoRA.
    sl=logits[:,:-1,:].float().contiguous()
    y=labels[:,1:].contiguous()
    w=tw[:,1:].contiguous().float()
    vocab=sl.size(-1)

    if not torch.isfinite(sl).all():
        return torch.tensor(float("nan"), device=logits.device)

    loss=F.cross_entropy(
        sl.view(-1,vocab),
        y.view(-1),
        ignore_index=-100,
        reduction="none"
    ).view(y.size())

    mask=(y!=-100).float()
    # Keep risk weighting but prevent extreme weights from destabilizing training.
    weights=torch.where(mask>0,torch.clamp(w,min=1.0,max=2.5),torch.zeros_like(w))
    per=(loss*weights).sum(dim=1)/weights.sum(dim=1).clamp_min(1.0)
    return per.mean()


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); losses=[]
    for batch in loader:
        labels=batch.pop("labels").to(device)
        tw=batch.pop("token_weights").to(device)
        g=batch.pop("g").to(device)
        batch={k:v.to(device) for k,v in batch.items()}
        set_all_gates(model,g)
        logits=model(**batch).logits
        loss=weighted_loss(logits,labels,tw)
        if torch.isfinite(loss): losses.append(float(loss.item()))
    model.train()
    return sum(losses)/max(1,len(losses))


def save_moe(model,tok,out):
    out.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if any(x in k for x in ["A_sh","B_sh","A_exp","B_exp"])}
    torch.save(state,out/"moe_lora.pt")
    tok.save_pretrained(out)
    with open(out/"dims.json","w") as f: json.dump(DIMS,f,indent=2)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_file",required=True)
    ap.add_argument("--valid_file",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--model",default="google/gemma-4-E4B-it")
    ap.add_argument("--batch_size",type=int,default=2)
    ap.add_argument("--grad_accum",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--max_source_len",type=int,default=768)
    ap.add_argument("--max_target_len",type=int,default=192)
    ap.add_argument("--target_modules",default="q_proj,v_proj")
    ap.add_argument("--r",type=int,default=4)
    ap.add_argument("--alpha",type=int,default=16)
    ap.add_argument("--eval_every",type=int,default=25)
    ap.add_argument("--save_every",type=int,default=100)
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"

    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=bnb,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
    model.config.use_cache=False
    model.gradient_checkpointing_enable()

    for p in model.parameters(): p.requires_grad=False
    model=replace_with_moe_lora(model,target_names=tuple(x.strip() for x in args.target_modules.split(",") if x.strip()),r=args.r,alpha=args.alpha,dropout=0.05)
    device=next(model.parameters()).device
    ntrain=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:",ntrain)

    train=FullDS(args.train_file,tok,max_source_len=args.max_source_len,max_target_len=args.max_target_len)
    valid=FullDS(args.valid_file,tok,max_source_len=args.max_source_len,max_target_len=args.max_target_len)
    coll=Collator(tok,args.max_source_len+args.max_target_len)
    tl=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=coll,num_workers=2)
    vl=DataLoader(valid,batch_size=args.batch_size,shuffle=False,collate_fn=coll,num_workers=2)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.01)
    updates=math.ceil(len(tl)/args.grad_accum)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,int(0.06*updates),updates)

    best=999; step=0
    model.train(); opt.zero_grad(set_to_none=True)
    for ep in range(args.epochs):
        pbar=tqdm(tl,desc=f"epoch {ep+1}/{args.epochs}")
        for i,batch in enumerate(pbar):
            labels=batch.pop("labels").to(device)
            tw=batch.pop("token_weights").to(device)
            g=batch.pop("g").to(device)
            batch={k:v.to(device) for k,v in batch.items()}
            set_all_gates(model,g)
            logits=model(**batch).logits
            loss=weighted_loss(logits,labels,tw)/args.grad_accum
            if not torch.isfinite(loss):
                print("[warn] non-finite loss"); opt.zero_grad(set_to_none=True); continue
            loss.backward()
            if (i+1)%args.grad_accum==0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
                pbar.set_postfix({"loss":round(float(loss.item()*args.grad_accum),4),"step":step})
                if step%args.eval_every==0:
                    ev=evaluate(model,vl,device)
                    print(f"[eval] step={step} loss={ev:.4f}")
                    if ev<best:
                        best=ev; save_moe(model,tok,out/"best"); print("[save] best ->",out/"best")
                if step%args.save_every==0:
                    save_moe(model,tok,out/f"step_{step}")
    save_moe(model,tok,out/"final")
    print("[done] best",best)


if __name__=="__main__":
    main()
