"""Pair matched query holdouts across the new system and both direct controls."""
from pathlib import Path
import sys, json, hashlib
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(Path(C['original_root'])/'.deps'))
import numpy as np

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def boot(base,system,mask):
    diff=(system[:,mask]-base[:,mask]).mean(axis=0)
    rng=np.random.default_rng(C['fusion']['bootstrap_seed'])
    values=[]
    for _ in range(C['fusion']['bootstrap_samples']//250):
        sampled=rng.integers(0,len(diff),(250,len(diff)))
        values.extend(diff[sampled].mean(axis=1))
    return {'mean_difference':float(diff.mean()),'ci95':np.percentile(values,[2.5,97.5]).tolist(),
            'one_sided_p_nonpositive':float((1+np.count_nonzero(np.asarray(values)<=0))/(1+len(values))),
            'win_fraction':float(np.mean(diff>1e-12)), 'tie_fraction':float(np.mean(abs(diff)<=1e-12)),
            'loss_fraction':float(np.mean(diff< -1e-12)), 'queries':len(diff)}

report={'configuration_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
        'primary_direct_control':'planner first 300, no CCADPE filtering, exact native BM25; independently calibrated identical fusion procedure',
        'secondary_direct_control':'all planner <=1000, more client scoring work; separately calibrated identical fusion procedure',
        'bootstrap_policy':'paired held-out queries, average three noise repeats per query before resampling; conditional on the fixed repeat-zero calibration models',
        'datasets':{}}
for dataset,cfg in C['datasets'].items():
    main=HERE/'results'/cfg['label'];base=read(main/'fusion.json')
    arrays=np.load(main/'holdout_metrics.npz');mask=arrays['outer_holdout']
    modes={'duet':{k:arrays[k] for k in ('semantic','lexical','duetrank','fixed_rrf','standard_rrf')}}
    selections={'duet':base['selection']}
    for cap in (300,1000):
        d=HERE/'direct_bm25'/dataset/f'fusion_{cap}'
        a=np.load(d/'holdout_metrics.npz')
        assert np.array_equal(a['query_ids'],arrays['query_ids'])
        assert np.array_equal(a['outer_holdout'],mask)
        assert np.array_equal(a['semantic'],arrays['semantic'])
        modes[f'direct{cap}']={k:a[k] for k in ('semantic','lexical','duetrank','fixed_rrf','standard_rrf')}
        selections[f'direct{cap}']=read(d/'fusion.json')['selection']
    comparisons={}
    def pair(label,basev,systemv):comparisons[label]=boot(basev,systemv,mask)
    pair('duet_fusion_minus_semantic',arrays['semantic'],arrays['duetrank'])
    for cap in (300,1000):
        direct=modes[f'direct{cap}']
        pair(f'duet_lexical_minus_direct{cap}',direct['lexical'],arrays['lexical'])
        pair(f'duet_fusion_minus_direct{cap}_fusion',direct['duetrank'],arrays['duetrank'])
        pair(f'direct{cap}_fusion_minus_semantic',arrays['semantic'],direct['duetrank'])
    count=np.load(main/'lexical_candidates.npz')['lexical_dpe']
    counts=(count>=0).sum(axis=-1)
    data={'query_count':cfg['queries'],'holdout_count':int(mask.sum()),
          'query_ids_sha256':hashlib.sha256('\n'.join(arrays['query_ids']).encode()).hexdigest(),
          'methods':{mode:{key:{'heldout_ndcg10':float(v[:,mask].mean()),'per_repeat':v[:,mask].mean(axis=1).tolist()} for key,v in vals.items()} for mode,vals in modes.items()},
          'paired_comparisons':comparisons,'selections':selections,
          'candidate_counts':{'ccadpe_exact_bm25_mean':float(counts.mean()),'ccadpe_exact_bm25_max':int(counts.max()),
                              'direct':read(HERE/'direct_bm25'/dataset/'quality.json')['controls']}}
    report['datasets'][dataset]=data
    print(dataset,{k:round(v['duetrank']['heldout_ndcg10'],6) for k,v in data['methods'].items()},flush=True)
out=HERE/'quality_comparison.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(out)
