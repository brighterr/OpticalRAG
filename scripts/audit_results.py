"""Check source coverage, exact choice scores and real retrieval evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--expected-count',type=int,default=503)
    args=ap.parse_args()
    from datasets import load_from_disk
    data=list(load_from_disk(str(args.data)))
    rows=[json.loads(x) for x in (args.run/'preds.jsonl').open()]
    assert len(data)==503
    assert 0 < args.expected_count <= 503
    assert len(rows)==args.expected_count
    assert len({r['_id'] for r in rows})==len(rows)
    hits=0
    for row, source in zip(rows,data):
        assert row['_id']==source['_id'] and row['gold']==source['answer']
        assert row['context']==source['context'] and row['question']==source['question']
        assert row['pred'] in ['A','B','C','D']
        correct=row['pred']==source['answer']
        assert bool(row['judge'])==correct
        hits+=int(correct)
        r=row['retrieval']; ids=r['selected_indices']; scores=r['selected_scores']
        assert r['enabled']==1 and r['actual_backend']=='faiss' and r['prefix_tokens']==16
        assert r['selected_tokens']==min(256,r['total_tokens'])==len(ids)==len(scores)
        assert len(set(ids))==len(ids) and ids==sorted(ids)
        assert all(0<=i<r['total_tokens'] for i in ids)
        assert all(math.isfinite(x) for x in scores)
    report={'count':len(rows),'hit':hits,'score':round(100*hits/len(rows),2),
            'full_evaluation':len(rows)==503,'coverage_scoring_retrieval_checks':'passed',
            'predictions_sha256':hashlib.sha256((args.run/'preds.jsonl').read_bytes()).hexdigest()}
    (args.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
