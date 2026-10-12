"""Preserved local Llama evaluator for ordinary drafts, SFT, and candidates.

This driver performs local GPU evaluation only when explicitly executed. The
ordinary GPT-Select aggregator reuses these stored judgments without re-evaluation.
"""
import argparse
import gc,hashlib,json,pathlib,sys,time
import run_independent as judge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=pathlib.Path,required=True)
    parser.add_argument('--rubric-contract',type=pathlib.Path,required=True)
    parser.add_argument('--llama-model',type=pathlib.Path,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    ROOT=args.source_dir
    judge.MODEL=args.llama_model
    judge.BATCH_SIZE=args.batch_size
    targets=('natural_judge_requests.jsonl','natural_judge_assignments.jsonl',
             'natural_judgments.jsonl','natural_judgments.manifest.json',
             'natural_judgments.first10.json','natural_evaluation_summary.json',
             'natural_evaluation.log')
    if any((ROOT/name).exists() for name in targets):
        parser.error('Use a fresh evaluation directory; existing judgments are frozen')
    import torch
    torch.set_num_threads(1)
    while not (ROOT/'natural_final.jsonl').exists():time.sleep(10)
    rows=[json.loads(x) for x in (ROOT/'natural_final.jsonl').read_text().splitlines() if x.strip()]
    assert len(rows)==60
    contract=json.loads(args.rubric_contract.read_text())
    prompts=contract['prompts'];unique={};assignments=[]
    for row in rows:
        for side,field in [('NaturalDraft','unsafe_response'),('SFT','sft_response'),('Candidate','denoiser_response'),('Final','final_response')]:
            text=row.get(field,'')
            if not text:continue
            ident=hashlib.sha256((row['question']+'\0'+text).encode()).hexdigest()
            unique.setdefault(ident,{'request_id':ident,'parser':'pointwise_expert_rubric','system_prompt':prompts['system_prompt'],'user_prompt':prompts['user_template'].format(question=row['question'],response=text),'max_tokens':768,'metadata':{'question':row['question'],'response':text,'question_response_sha256':ident}})
            assignments.append({'canonical_id':row['canonical_id'],'method':side,'request_id':ident})
    for name,data in [('natural_judge_requests.jsonl',list(unique.values())),('natural_judge_assignments.jsonl',assignments)]:
        (ROOT/name).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in data)+'\n')
    tokenizer,model=judge.load_local_generator(judge.MODEL)
    result=judge.run_queue(model,tokenizer,ROOT/'natural_judge_requests.jsonl',ROOT/'natural_judgments.jsonl','natural_independent')
    (ROOT/'natural_evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
