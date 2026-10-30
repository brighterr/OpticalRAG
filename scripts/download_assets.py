"""Download revision-pinned models and prepare the 503-row evaluation dataset."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, default=ROOT / 'assets')
    ap.add_argument('--only', choices=['data', 'models', 'all'], default='all')
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    config = json.loads((ROOT / 'configs/longbench.json').read_text())
    if args.only in ('data', 'all'):
        from datasets import Dataset, load_from_disk
        raw = hf_hub_download('zai-org/LongBench-v2', 'data.json', repo_type='dataset',
                              revision=config['data_revision'])
        rows = json.loads(Path(raw).read_text())
        if len(rows) != 503 or len({x['_id'] for x in rows}) != 503:
            raise ValueError('Unexpected dataset coverage')
        if any(x['answer'] not in 'ABCD' or len(x['answer']) != 1 or not x['context'] for x in rows):
            raise ValueError('Invalid data schema')
        dest = args.assets / 'data/LongBench-v2/train'
        if dest.exists():
            if list(load_from_disk(str(dest))) != rows:
                raise ValueError('Existing dataset differs from pinned source')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows).save_to_disk(str(dest))
        report = {'count':len(rows), 'revision':config['data_revision'],
                  'raw_sha256':hashlib.sha256(Path(raw).read_bytes()).hexdigest(),
                  'arrow_sha256':hashlib.sha256((dest/'data-00000-of-00001.arrow').read_bytes()).hexdigest()}
        if report['arrow_sha256'] != config['arrow_sha256']:
            raise ValueError('Arrow serialization differs; use the documented datasets version')
        (dest.parent/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.only in ('models', 'all'):
        for name, model in config['models'].items():
            snapshot_download(model['model_id'], revision=model['revision'],
                              local_dir=args.assets/'models'/name)
    print('Requested assets are ready.')

if __name__ == '__main__':
    main()
