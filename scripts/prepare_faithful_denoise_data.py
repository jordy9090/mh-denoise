import argparse, json, random, re, math
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]
DIM2ID = {d:i for i,d in enumerate(DIMS)}


def read_jsonl(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")


def get_field(ex,*names,default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x):
    x=str(x).strip().lower().replace("-","_").replace(" ","_")
    return x if x in DIM2ID else "overall_quality"


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def sim(a,b):
    al, bl = a.lower(), b.lower()
    sa=set(re.findall(r"[a-zA-Z']+", al)); sb=set(re.findall(r"[a-zA-Z']+", bl))
    j=len(sa&sb)/max(1,len(sa|sb))
    seq=SequenceMatcher(None,al,bl).ratio()
    return 0.5*j + 0.5*seq


def align_sentences(u, y):
    us=split_sentences(u)
    ys=split_sentences(y)
    if not us:
        us=[u.strip()]
    if not ys:
        ys=[y.strip()]
    units=[]
    for i,usent in enumerate(us):
        best_j, best_s, best_score = 0, ys[0], -1
        for j, ysent in enumerate(ys):
            sc=sim(usent, ysent)
            if sc>best_score:
                best_j,best_s,best_score=j,ysent,sc
        units.append({
            "unsafe_span": usent,
            "safe_span": best_s,
            "unsafe_index": i,
            "safe_index": best_j,
            "align_score": best_score,
        })
    return units


@torch.no_grad()
def router_predict(rows, router_dir):
    if not router_dir:
        return [None]*len(rows)
    tok=AutoTokenizer.from_pretrained(router_dir)
    model=AutoModelForSequenceClassification.from_pretrained(router_dir)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    out=[]
    for i in tqdm(range(0,len(rows),16), desc="router"):
        batch=rows[i:i+16]
        texts=[]
        for ex in batch:
            q=get_field(ex,"question","query","user_question")
            u=get_field(ex,"unsafe_response","corrupted_response","bad_response")
            texts.append("Question:\n"+q+"\n\nUnsafe response:\n"+u+"\n\nTask: identify the primary violated response-quality dimension.")
        enc=tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        probs=torch.softmax(model(**enc).logits, dim=-1).detach().cpu().tolist()
        out.extend(probs)
    return out


@torch.no_grad()
def score_risks(rows, risk_dir):
    tok=AutoTokenizer.from_pretrained(risk_dir)
    model=AutoModelForSequenceClassification.from_pretrained(risk_dir)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()

    flat=[]
    for ridx,row in enumerate(rows):
        q=row["question"]; dim=row["condition_dim"]
        for uidx,unit in enumerate(row["alignments"]):
            text=("Question:\n"+q+"\n\nViolation dimension:\n"+dim+
                  "\n\nCandidate span:\n"+unit["unsafe_span"]+
                  "\n\nTask: predict whether this span is a safety or quality risk for the given dimension.")
            flat.append((ridx,uidx,text))
    scores=[[0.0]*len(r["alignments"]) for r in rows]
    for i in tqdm(range(0,len(flat),32), desc="risk scorer"):
        batch=flat[i:i+32]
        enc=tok([x[2] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=384).to(device)
        prob=torch.sigmoid(model(**enc).logits.squeeze(-1)).detach().cpu().tolist()
        for (ridx,uidx,_),p in zip(batch,prob):
            scores[ridx][uidx]=float(p)
    return scores


def choose_condition(oracle_dim, router_probs, rng, teacher_force_prob):
    if router_probs is None or rng.random() < teacher_force_prob:
        probs=[0.0]*len(DIMS); probs[DIM2ID[oracle_dim]]=1.0
        source="oracle"
    else:
        probs=router_probs; source="router_pred"
    top=max(range(len(probs)), key=lambda i: probs[i])
    return {
        "condition_dim": DIMS[top],
        "condition_source": source,
        "condition_probs": {DIMS[i]: float(probs[i]) for i in range(len(DIMS))}
    }


def sample_state(ctype, t, r, rng):
    if ctype == "empty":
        return "EMPTY"
    if ctype == "safe":
        # clean-side denoising: mostly safe, some masked
        p_mask = min(0.50, 0.15 + 0.45*t*r)
        return "MASK" if rng.random() < p_mask else "SAFE"
    if ctype == "unsafe":
        # inference-like: mostly unsafe, high-risk spans more likely masked
        p_mask = min(0.70, 0.10 + 0.70*t*r)
        return "MASK" if rng.random() < p_mask else "UNSAFE"
    if ctype == "bridge":
        # edit bridge between safe and unsafe
        p_mask = min(0.45, 0.10 + 0.45*t*r)
        p_unsafe = min(0.70, 0.20 + 0.60*t*(1.0 - 0.25*r))
        z = rng.random()
        if z < p_mask:
            return "MASK"
        if z < p_mask + p_unsafe:
            return "UNSAFE"
        return "SAFE"
    raise ValueError(ctype)


def build_z(units, ctype, t, rng):
    if ctype == "empty":
        return "", [], []
    parts=[]
    states=[]
    weighted=[]
    for unit in units:
        r=float(unit["risk_score"])
        st=sample_state(ctype,t,r,rng)
        states.append(st)
        if st=="SAFE":
            parts.append(unit["safe_span"])
        elif st=="UNSAFE":
            parts.append(unit["unsafe_span"])
        elif st=="MASK":
            parts.append("[EDIT]")
        else:
            parts.append(unit["unsafe_span"])
        if r >= 0.50:
            weighted.append({
                "safe_span": unit["safe_span"],
                "unsafe_span": unit["unsafe_span"],
                "risk_score": r,
                "state": st,
                "align_score": unit["align_score"]
            })
    return " ".join(parts), states, weighted


def make_rows(base_rows, router_probs, risk_dir, examples_per_row, teacher_force_prob, seed):
    rng=random.Random(seed)
    staged=[]
    for i,ex in enumerate(base_rows):
        q=get_field(ex,"question","query","user_question")
        u=get_field(ex,"unsafe_response","corrupted_response","bad_response")
        y=get_field(ex,"safe_response","target_response","response")
        d=norm_dim(get_field(ex,"target_dimension","dimension","violated_dimension"))
        cond=choose_condition(d, router_probs[i], rng, teacher_force_prob)
        aligns=align_sentences(u,y)
        staged.append({
            "question":q, "unsafe_response":u, "safe_response":y, "target_dimension":d,
            **cond, "alignments": aligns
        })

    risk_scores=score_risks(staged, risk_dir)
    for row, scs in zip(staged, risk_scores):
        for unit,score in zip(row["alignments"], scs):
            unit["risk_score"]=score

    out=[]
    ctypes=["empty","safe","unsafe","bridge"]
    tvals=[0.25,0.50,0.75,1.00]
    for idx,row in enumerate(staged):
        for k in range(examples_per_row):
            ctype=ctypes[k%len(ctypes)]
            t=tvals[(idx+k)%len(tvals)]
            z,states,weighted=build_z(row["alignments"], ctype, t, rng)
            loss_weight=1.0 + 0.20*t + (0.25 if ctype in ["unsafe","bridge"] else 0.0)
            out.append({
                "question": row["question"],
                "unsafe_response": row["unsafe_response"],
                "safe_response": row["safe_response"],
                "target_dimension": row["target_dimension"],
                "condition_dim": row["condition_dim"],
                "condition_source": row["condition_source"],
                "condition_probs": row["condition_probs"],
                "corruption_source": ctype,
                "corruption_level": float(t),
                "z_t": z,
                "transition_states": states,
                "alignments": row["alignments"],
                "target_weight_spans": weighted,
                "loss_weight": float(loss_weight),
            })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--examples_per_row", type=int, default=8)
    ap.add_argument("--teacher_force_prob", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    base=read_jsonl(args.input)
    rprobs=router_predict(base,args.router_dir)
    out=make_rows(base,rprobs,args.risk_scorer_dir,args.examples_per_row,args.teacher_force_prob,args.seed)
    write_jsonl(out,args.output)
    print("wrote", len(out), "to", args.output)
    print("source dist:", Counter(x["corruption_source"] for x in out))
    print("condition source:", Counter(x["condition_source"] for x in out))
    with open(str(Path(args.output).with_suffix(".manifest.json")), "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "output": args.output,
            "n_input": len(base),
            "n_output": len(out),
            "examples_per_row": args.examples_per_row,
            "notes": "Overleaf-faithful v1: alignment A(u,y), learned risk scorer r_l(g), SAFE/UNSAFE/MASK transitions, source mixture, target_weight_spans."
        }, f, indent=2, ensure_ascii=False)


if __name__=="__main__":
    main()
