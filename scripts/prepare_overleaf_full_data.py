import argparse, json, random, re, math
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DIMS=["overall_quality","empathy","specificity","medical_advice","factual_consistency","toxicity"]
DIM2ID={d:i for i,d in enumerate(DIMS)}


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(rows,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")


def get_field(ex,*names,default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x):
    x=str(x).strip().lower().replace("-","_").replace(" ","_")
    return x if x in DIM2ID else "overall_quality"


def d_vector(ex):
    v=[0.0]*len(DIMS)
    if "target_dimensions" in ex and isinstance(ex["target_dimensions"],list):
        ds=[norm_dim(x) for x in ex["target_dimensions"]]
    else:
        ds=[norm_dim(get_field(ex,"target_dimension","dimension","violated_dimension"))]
    for d in ds: v[DIM2ID[d]]=1.0
    return v


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+",text.strip()) if p.strip()]


def sim(a,b):
    al,bl=a.lower(),b.lower()
    wa=set(re.findall(r"[a-zA-Z']+",al)); wb=set(re.findall(r"[a-zA-Z']+",bl))
    j=len(wa&wb)/max(1,len(wa|wb))
    seq=SequenceMatcher(None,al,bl).ratio()
    return 0.55*seq+0.45*j


def monotonic_align(u,y):
    U=split_sentences(u) or [u]
    Y=split_sentences(y) or [y]
    n,m=len(U),len(Y)
    dp=[[0.0]*(m+1) for _ in range(n+1)]
    back=[[None]*(m+1) for _ in range(n+1)]
    gap=-0.15
    for i in range(1,n+1):
        dp[i][0]=dp[i-1][0]+gap; back[i][0]="del"
    for j in range(1,m+1):
        dp[0][j]=dp[0][j-1]+gap; back[0][j]="ins"
    for i in range(1,n+1):
        for j in range(1,m+1):
            s=sim(U[i-1],Y[j-1])
            opts=[
                (dp[i-1][j-1]+s,"sub"),
                (dp[i-1][j]+gap,"del"),
                (dp[i][j-1]+gap,"ins"),
            ]
            best=max(opts,key=lambda x:x[0])
            dp[i][j]=best[0]; back[i][j]=best[1]
    i,j=n,m; pairs=[]
    while i>0 or j>0:
        op=back[i][j]
        if op=="sub":
            pairs.append({"a":U[i-1],"b":Y[j-1],"op":"substitute","align_score":sim(U[i-1],Y[j-1])})
            i-=1; j-=1
        elif op=="del":
            pairs.append({"a":U[i-1],"b":"","op":"delete","align_score":0.0})
            i-=1
        elif op=="ins":
            pairs.append({"a":"","b":Y[j-1],"op":"insert","align_score":0.0})
            j-=1
        else:
            break
    pairs=list(reversed(pairs))
    return pairs


@torch.no_grad()
def predict_router(rows,router_dir):
    tok=AutoTokenizer.from_pretrained(router_dir)
    model=AutoModelForSequenceClassification.from_pretrained(router_dir)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    out=[]
    for i in tqdm(range(0,len(rows),16),desc="router"):
        batch=rows[i:i+16]
        texts=[]
        for ex in batch:
            q=get_field(ex,"question","query","user_question")
            u=get_field(ex,"unsafe_response","corrupted_response","bad_response")
            texts.append("Question:\n"+q+"\n\nUnsafe response:\n"+u+"\n\nTask: identify all violated mental-health response quality dimensions.")
        enc=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
        probs=torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()
        out.extend(probs)
    return out


@torch.no_grad()
def score_risk(rows,risk_dir):
    tok=AutoTokenizer.from_pretrained(risk_dir)
    model=AutoModelForSequenceClassification.from_pretrained(risk_dir)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    flat=[]
    for ri,row in enumerate(rows):
        q=row["question"]
        for li,p in enumerate(row["alignments"]):
            a=p["a"]
            text="Question:\n"+q+"\n\nCandidate span:\n"+a+"\n\nTask: predict which counseling quality dimensions this span may violate."
            flat.append((ri,li,text))
    scores=[[ [0.0]*len(DIMS) for _ in r["alignments"] ] for r in rows]
    for i in tqdm(range(0,len(flat),32),desc="risk"):
        batch=flat[i:i+32]
        enc=tok([x[2] for x in batch],return_tensors="pt",padding=True,truncation=True,max_length=384).to(device)
        probs=torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()
        for (ri,li,_),p in zip(batch,probs):
            scores[ri][li]=[float(x) for x in p]
    return scores


def choose_g(d,hat,rng,p_tf):
    if rng.random()<p_tf:
        return d,"oracle"
    return hat,"router_pred"


def r_l(g,risk_vec):
    return max(float(g[k])*float(risk_vec[k]) for k in range(len(DIMS)))


def sample_bridge_state(beta,r,base_rho,lambda_mask,rng):
    pi=max(0.0,min(1.0,base_rho+lambda_mask*beta*r))
    p_safe=1.0-beta
    p_mask=beta*pi
    z=rng.random()
    if z<p_safe: return "SAFE"
    if z<p_safe+p_mask: return "MASK"
    return "UNSAFE"


def realize(pair,state,mask_token):
    if state=="SAFE": return pair["b"]
    if state=="UNSAFE": return pair["a"]
    if state=="MASK": return mask_token
    return ""


def build_z(pairs,source,t,T,base_rho,lambda_mask,mask_token,rng):
    beta=t/float(T)
    parts=[]; states=[]; weight_spans=[]
    for p in pairs:
        r=float(p["r"])
        if source=="empty":
            continue
        elif source=="bridge":
            st=sample_bridge_state(beta,r,base_rho,lambda_mask,rng)
        elif source=="unsafe":
            if p["a"]=="":
                st="MASK" if rng.random()<0.25 else "SAFE"
            else:
                pi=max(0.0,min(1.0,base_rho+lambda_mask*beta*r))
                st="MASK" if rng.random()<beta*pi else "UNSAFE"
        elif source=="safe":
            if p["b"]=="":
                st="MASK"
            else:
                st="MASK" if rng.random()<beta*(base_rho+0.5*r) else "SAFE"
        else:
            raise ValueError(source)
        states.append(st)
        txt=realize(p,st,mask_token)
        if txt:
            parts.append(txt)
        if p["b"] and r>0:
            weight_spans.append({"safe_span":p["b"],"unsafe_span":p["a"],"risk":r,"state":st,"op":p["op"]})
    return " ".join(parts),states,weight_spans


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--router_dir",required=True)
    ap.add_argument("--risk_scorer_dir",required=True)
    ap.add_argument("--examples_per_row",type=int,default=8)
    ap.add_argument("--p_tf",type=float,default=0.65)
    ap.add_argument("--T",type=int,default=4)
    ap.add_argument("--rho",type=float,default=0.15)
    ap.add_argument("--lambda_mask",type=float,default=0.75)
    ap.add_argument("--mask_token",default="<MASK>")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    rng=random.Random(args.seed)

    base=read_jsonl(args.input)
    hats=predict_router(base,args.router_dir)

    staged=[]
    for ex,hat in zip(base,hats):
        q=get_field(ex,"question","query","user_question")
        u=get_field(ex,"unsafe_response","corrupted_response","bad_response")
        y=get_field(ex,"safe_response","target_response","response")
        d=d_vector(ex)
        staged.append({"question":q,"unsafe_response":u,"safe_response":y,"d":d,"hat_d":hat,"alignments":monotonic_align(u,y)})

    risks=score_risk(staged,args.risk_scorer_dir)
    for row,rv in zip(staged,risks):
        for p,risk_vec in zip(row["alignments"],rv):
            p["risk_vec"]=risk_vec

    sources=["bridge","unsafe","safe","empty"]
    out=[]
    for idx,row in enumerate(staged):
        for k in range(args.examples_per_row):
            g,src=choose_g(row["d"],row["hat_d"],rng,args.p_tf)
            source=sources[k%len(sources)]
            t=(k%args.T)+1
            pairs=[]
            for p in row["alignments"]:
                pp=dict(p)
                pp["r"]=r_l(g,pp["risk_vec"])
                pairs.append(pp)
            z,states,weight_spans=build_z(pairs,source,t,args.T,args.rho,args.lambda_mask,args.mask_token,rng)
            out.append({
                "question":row["question"],
                "unsafe_response":row["unsafe_response"],
                "safe_response":row["safe_response"],
                "d":row["d"],
                "hat_d":row["hat_d"],
                "g":g,
                "g_source":src,
                "source":source,
                "t":t,
                "T":args.T,
                "beta":t/float(args.T),
                "z_t":z,
                "states":states,
                "alignments":pairs,
                "target_weight_spans":weight_spans
            })
    write_jsonl(out,args.output)
    print("wrote",len(out),args.output)
    print("source:",Counter(x["source"] for x in out))
    print("g_source:",Counter(x["g_source"] for x in out))
    with open(str(Path(args.output).with_suffix(".manifest.json")),"w",encoding="utf-8") as f:
        json.dump(vars(args)|{"n_input":len(base),"n_output":len(out),"dims":DIMS},f,indent=2,ensure_ascii=False)


if __name__=="__main__":
    main()
