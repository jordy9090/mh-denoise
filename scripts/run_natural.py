"""Preserved ordinary Mistral inference driver using the unchanged cascade.

All model, checkpoint and data locations are explicit; running this driver performs
GPU generation and is not part of the static verification procedure.
"""
import argparse
import gc,json,pathlib,subprocess,sys,time
from run_independent import generate_batch,render_chat,load_local_generator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=pathlib.Path,required=True)
    parser.add_argument('--checkpoint-dir',type=pathlib.Path,required=True)
    parser.add_argument('--mistral-model',type=pathlib.Path,required=True)
    parser.add_argument('--gemma-model',type=pathlib.Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    args=parser.parse_args()
    ROOT=args.source_dir
    RUN=args.checkpoint_dir
    SCRIPTS=pathlib.Path(__file__).resolve().parent
    targets=('natural_drafts.jsonl','natural_sft.jsonl','natural_sft.manifest.json',
             'natural_final.jsonl','natural_final.manifest.json','natural.log',
             'natural_sft.log','natural_refinement.log')
    if any((ROOT/name).exists() for name in targets):
        parser.error('Use a fresh source directory; existing ordinary outputs are frozen')
    import torch
    torch.set_num_threads(1)
    MODEL=args.mistral_model
    input_rows=[json.loads(line) for line in (ROOT/'canonical_input.jsonl').read_text().splitlines()]
    started=time.monotonic();tokenizer,model=load_local_generator(MODEL);drafts=[]
    for start in range(0,len(input_rows),8):
        batch=input_rows[start:start+8]
        prompts=[render_chat(tokenizer,'You are a helpful assistant.',row['question']) for row in batch]
        generations=generate_batch(model,tokenizer,prompts,768)
        for row,(response,generation) in zip(batch,generations):
            drafts.append({'canonical_id':row['canonical_id'],'question':row['question'],'unsafe_response':response.strip(),'safe_response':row['safe_response'],'source':row['source'],'source_group_id':row['source_group_id'],'duplicate_cluster_id':row.get('duplicate_cluster_id'),'question_normalized_sha256':row['question_normalized_sha256'],'split':'test','draft_generation_model':'Mistral-7B-Instruct-v0.3','draft_generation':generation,'source_kind':'ordinary_generation'})
        print(json.dumps({'stage':'ordinary_drafts','completed':len(drafts),'total':len(input_rows)}),flush=True)
    (ROOT/'natural_drafts.jsonl').write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in drafts)+'\n')
    del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    base=str(args.gemma_model)
    python=args.python
    sft=[python,str(SCRIPTS/'build_sft_outputs_for_risk_tuning.py'),'--base_model',base,'--adapter_dir',str(RUN/'gemma/sft/final'),'--input',str(ROOT/'natural_drafts.jsonl'),'--output',str(ROOT/'natural_sft.jsonl'),'--max_source_len','1024','--max_new_tokens','1536','--temperature','0','--repetition_penalty','1.15','--no_repeat_ngram_size','4','--sft_prompt_style','sft_plain']
    with (ROOT/'natural_sft.log').open('w') as log:subprocess.run(sft,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'stage':'sft_complete'}),flush=True)
    tau=json.loads((RUN/'valid_risk_thresholds.json').read_text())['thresholds']['full']['threshold']
    refine=[python,str(SCRIPTS/'run_gemma_selective_risk_refinement.py'),'--base_model',base,'--sft_adapter_dir',str(RUN/'gemma/sft/final'),'--risk_adapter_dir',str(RUN/'gemma/proposed_mask_off/final'),'--router_dir',str(RUN/'router/final'),'--risk_scorer_dir',str(RUN/'scorer/final'),'--input',str(ROOT/'natural_sft.jsonl'),'--output',str(ROOT/'natural_final.jsonl'),'--risk_contract','fullpaper_v1','--component_mode','full','--reuse_sft_response','--zt_strategy','no_mask','--gate_strategy','overall','--gate_risk_threshold',str(tau),'--risk_threshold',str(tau),'--max_source_len','4096','--max_new_tokens','1536','--temperature','0','--repetition_penalty','1.15','--no_repeat_ngram_size','4','--min_risk_delta','0.01','--specificity_min_ratio','0.6','--keyword_overlap_slack','0.15','--min_word_count','20','--router_max_len','512','--risk_max_len','384']
    with (ROOT/'natural_refinement.log').open('w') as log:subprocess.run(refine,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'status':'complete_natural_transfer_inference','rows':len(drafts),'runtime_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    main()
