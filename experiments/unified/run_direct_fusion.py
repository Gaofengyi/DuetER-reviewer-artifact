"""Apply exactly the new DuetRank calibration/evaluation policy to BM25 controls."""
from pathlib import Path
import time, json
import unified_pipeline as u

def wait_for(path, maximum_seconds=14400):
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > maximum_seconds:
            raise TimeoutError(path)
        time.sleep(5)

if __name__ == '__main__':
    u.torch.set_num_threads(2)
    with u.threadpool_limits(limits=2):
        for dataset, cfg in u.C['datasets'].items():
            direct = u.HERE / 'direct_bm25' / dataset
            wait_for(u.HERE / 'results' / cfg['label'] / 'semantic.json')
            wait_for(direct / 'quality.json')
            while not json.loads((direct / 'quality.json').read_text(encoding='utf-8')).get('verification', {}).get('all_queries_subset_rank_invariant', False):
                time.sleep(5)
            for cap in (300, 1000):
                dest = direct / f'fusion_{cap}'
                if (dest / 'fusion.json').exists():
                    print(dataset, cap, 'already complete', flush=True)
                    continue
                print(dataset, cap, 'FUSION START', flush=True)
                u.run_fusion(dataset, lexical_path=direct / f'direct_{cap}.rankings.npz', destination=dest)
        print('ALL DIRECT FUSIONS COMPLETE', flush=True)
