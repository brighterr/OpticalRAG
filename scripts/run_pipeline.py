"""Reproduce the fixed LongBench-v2 configuration with explicit local assets."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, default=ROOT/'assets')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--font', type=Path, required=True)
    ap.add_argument('--cache', type=Path, help='Existing cache or new cache directory')
    ap.add_argument('--stage', choices=['all','cache','index','eval'], default='all')
    ap.add_argument('--run-id', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--limit', type=int, default=0, help='0 = full 503 samples; positive = smoke only')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id): ap.error('Invalid run-id')
    if args.limit < 0 or args.limit > 503: ap.error('limit must be in [0,503]')
    cfg = json.loads((ROOT/'configs/longbench.json').read_text())
    assets = args.assets.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    font = args.font.expanduser().resolve()
    # Keep smoke caches separate from full runs.
    suffix = 'full' if args.limit == 0 else f'smoke{args.limit}'
    cache = (args.cache or assets/f'cache/longbench_f15_m20_l13_{suffix}').expanduser().resolve()
    out = ROOT/'runs'/args.run_id
    if out.exists(): raise FileExistsError('Choose a new run-id; output directory already exists')
    data = assets/'data/LongBench-v2/train'
    model = assets/'models/Qwen2.5-7B-Instruct'
    visual = assets/'models/DeepSeek-OCR'
    py = sys.executable
    common = ['--model_path',str(model),'--deepseek_path',str(visual),'--ckpt',str(checkpoint),
              '--opticalrag_context_budget_chars','0','--opticalrag_context_sampling','head',
              '--opticalrag_query_pooling','max','--opticalrag_num_query_tokens','16',
              '--opticalrag_retrieval_enable','1','--opticalrag_retrieval_backend','offline_index',
              '--opticalrag_retrieval_score_type','dot','--opticalrag_retrieval_topk','256',
              '--opticalrag_retrieval_topk_factor','8','--opticalrag_retrieval_rerank_factor','4',
              '--opticalrag_retrieval_preserve_order','1','--opticalrag_use_downsample','1',
              '--opticalrag_offline_cache_dir',str(cache),'--opticalrag_offline_index_dir',str(cache/'indexes'),
              '--max_samples',str(args.limit)]
    commands = {
        'cache':[py,str(ROOT/'scripts/build_visual_cache_ddp.py'),'--cache_dir',str(cache),
                 '--deepseek_path',str(visual),'--longbench_local_path',str(data),
                 '--uid_prefix','longbench_train','--chunk_chars','24000','--page_batch','4',
                 '--render_mode','text','--page_size','640','--font_size','15','--margin','20',
                 '--line_height_scale','1.3','--max_samples',str(args.limit)],
        'index':[py,str(ROOT/'scripts/build_visual_index.py'),'--cache_dir',str(cache),'--score_type','dot'],
        'eval':[py,str(ROOT/'scripts/eval_longbench.py'),'--mode','opticalrag','--decode_mode','choice_score',
                '--longbench_local_path',str(data),'--output_jsonl',str(out/'preds.jsonl'),
                '--summary_json',str(out/'summary.json'),'--baseline_context_budget_chars','12000',
                '--opticalrag_visual_num_windows','1','--opticalrag_visual_pages_per_window','4',
                '--opticalrag_offline_uid_prefix','longbench_train']+common}
    if args.plan_only:
        print(json.dumps(commands,indent=2)); return
    for p, expected in [(checkpoint,cfg['checkpoint_sha256']), (font,cfg['font_sha256']),
                        (data/'data-00000-of-00001.arrow',cfg['arrow_sha256'])]:
        if sha(p) != expected: raise ValueError(f'Asset hash mismatch: {p}')
    # Verify downloaded models against the evaluated immutable manifests.
    needed = [] if args.stage == 'index' else ['DeepSeek-OCR']
    if args.stage in ['all','eval']: needed.append('Qwen2.5-7B-Instruct')
    for name in needed:
        manifest = json.loads((ROOT/'configs'/f'{name}.json').read_text())
        for item in manifest['files']:
            if sha(assets/'models'/name/item['path']) != item['sha256']:
                raise ValueError(f'Model hash mismatch: {name}/{item["path"]}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
               OPTICALRAG_RENDER_FONT_PATH=str(font),OPTICALRAG_RENDER_FONT_SIZE='15',OPTICALRAG_RENDER_MARGIN='20',
               OPTICALRAG_RENDER_LINE_HEIGHT_SCALE='1.3')
    out.mkdir(parents=True)
    record = dict(cfg, commands=commands, limit=args.limit,
                  code_sha256={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')},
                  python=sys.version)
    (out/'config.json').write_text(json.dumps(record,indent=2)+'\n')
    def status(value, stage):
        (out/'status.json').write_text(json.dumps({'status':value,'stage':stage},indent=2)+'\n')
    try:
        for stage, cmd in commands.items():
            if args.stage != 'all' and args.stage != stage: continue
            status('running',stage)
            with (out/f'{stage}.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if args.stage in ['all','eval']:
            subprocess.run([py,str(ROOT/'scripts/audit_results.py'),str(out),'--data',str(data),
                            '--expected-count',str(args.limit or 503)],check=True,env=env)
        status('complete',args.stage)
    except Exception:
        status('failed',args.stage)
        raise
    print(f'Results: {out}')

if __name__ == '__main__': main()
