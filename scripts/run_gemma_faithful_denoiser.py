import argparse, json, re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import PeftModel


DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
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


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def build_prompt_from_dataset_row(ex):
    q=ex["question"].strip()
    u=ex["unsafe_response"].strip()
    d=ex.get("condition_dim", ex.get("target_dimension","overall_quality"))
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


def cleanup(text):
    text=text.strip()
    # Cut off common leaked labels.
    bad_markers=[
        "Question:", "Unsafe response:", "Violation dimension:", "Intermediate draft",
        "Corruption source:", "Final safe response:", "Task:", "Rules:",
        "Analysis:", "Thought:", "Reasoning:", "Risk", "Aspect"
    ]
    for m in bad_markers:
        if m in text:
            parts=text.split(m)
            # keep text before a later leaked section when possible
            if parts[0].strip():
                text=parts[0].strip()
            else:
                text=parts[-1].strip()
    text=text.replace("[EDIT]","").strip()
    text=re.sub(r"\s+"," ",text)
    text=re.sub(r"\s+([.,!?;:])",r"\1",text)
    text=re.sub(r"([,.;:])\1{1,}",r"\1",text)
    return text.strip()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir",required=True)
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--max_source_len",type=int,default=1024)
    ap.add_argument("--max_new_tokens",type=int,default=220)
    ap.add_argument("--temperature",type=float,default=0.0)
    ap.add_argument("--top_p",type=float,default=0.9)
    ap.add_argument("--repetition_penalty",type=float,default=1.15)
    ap.add_argument("--no_repeat_ngram_size",type=int,default=4)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.adapter_dir,trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    tok.padding_side="left"

    bnb=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base=AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model=PeftModel.from_pretrained(base,args.adapter_dir)
    model.eval()
    device=next(model.parameters()).device

    outs=[]
    for ex in tqdm(read_jsonl(args.input)):
        prompt=build_prompt_from_dataset_row(ex)
        enc=tok(prompt,return_tensors="pt",truncation=True,max_length=args.max_source_len).to(device)
        kwargs=dict(
            **enc,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if args.temperature and args.temperature>0:
            kwargs.update(dict(do_sample=True,temperature=args.temperature,top_p=args.top_p))
        else:
            kwargs.update(dict(do_sample=False))

        with torch.no_grad():
            gen=model.generate(**kwargs)

        new=gen[0][enc["input_ids"].shape[-1]:]
        raw=tok.decode(new,skip_special_tokens=True)
        out=dict(ex)
        out["faithful_response_raw"]=raw
        out["faithful_response"]=cleanup(raw)
        out["method"]="faithful_overleaf_denoiser"
        outs.append(out)

    write_jsonl(outs,args.output)
    print("saved to",args.output)


if __name__=="__main__":
    main()
