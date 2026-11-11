"""Assemble the frozen primary analysis and optional path-only explanation."""
from pathlib import Path
import hashlib, importlib.util, json, sys
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('component_original_quality_finish', HERE/'run_quality.py')
q = importlib.util.module_from_spec(spec); sys.modules[spec.name] = q; spec.loader.exec_module(q)
np = q.np
PLAN = HERE/'PATH_DIAGNOSTIC_PLAN.json'
METHODS = {'semantic_native': 'semantic_full', 'semantic_rank_only': 'semantic_no_native',
           'semantic_plain_native': 'semantic_plain', 'lexical_native': 'lexical_full',
           'lexical_planner_order': 'lexical_no_native', 'lexical_plain_native': 'lexical_plain',
           'lexical_planner_quota_native': 'lexical_planner_quota'}


def interval(native, rank_only):
    assert native.shape == rank_only.shape and native.ndim == 2
    n = len(native); rng = np.random.default_rng(20261109)
    diff = np.empty((10000, 2)); ratio = np.empty_like(diff)
    for start in range(0, 10000, 64):
        stop = min(start+64, 10000)
        ix = rng.integers(0, n, size=(stop-start, n))
        a = native[ix].mean(axis=1); b = rank_only[ix].mean(axis=1)
        assert np.all(b > 0)
        diff[start:stop] = a-b; ratio[start:stop] = a/b
    a = native.mean(axis=0); b = rank_only.mean(axis=0)
    ci = np.percentile(diff, [2.5, 97.5], axis=0)
    rci = np.percentile(ratio, [2.5, 97.5], axis=0)
    return {metric: {'native': float(a[j]), 'rank_only': float(b[j]),
                     'native_minus_rank_only': float(a[j]-b[j]),
                     'paired_difference_ci95': ci[:, j].tolist(),
                     'native_over_rank_only': float(a[j]/b[j]), 'paired_ratio_ci95': rci[:, j].tolist()}
            for j, metric in enumerate(('ndcg10', 'recall100'))}


def path_diagnostics():
    output = HERE/'path_diagnostics'
    if (output/'summary.json').exists():
        raise FileExistsError(output/'summary.json')
    results = {}
    for dataset, cfg in q.c.u.C['datasets'].items():
        ds = q.c.prepare(dataset); folder = HERE/'results'/cfg['label']
        report = q.c.read(folder/'verification.json')
        assert report['outputs']['stage_rankings.npz']['sha256'] == q.c.sha(folder/'stage_rankings.npz')
        with np.load(folder/'stage_rankings.npz', allow_pickle=False) as z:
            indices = z['query_indices']; ids = z['query_ids']
            assert np.array_equal(indices, np.flatnonzero(ds['outer_holdout']))
            assert np.array_equal(ids, np.asarray(ds['query_ids'])[indices]) and z['outer_holdout'].all()
            ranks = {key: z[name].copy() for key, name in METHODS.items()}
        docmap = q.c.qrel_map(ds)
        assert ds['qrel_document_map_coverage']['absent_from_corpus'] == 0
        ds.update(ids=ids.tolist(), docmap=docmap)
        means = {}; per_query = {}; arrays = {}
        for name, values in ranks.items():
            if values.ndim == 2:
                values = values[None]
            evaluated = [q.metrics(ds, row[:, :100]) for row in values]
            ndcg = np.stack([x[0] for x in evaluated]); recall = np.stack([x[1] for x in evaluated])
            arrays[name+'_ndcg10'] = ndcg; arrays[name+'_recall100'] = recall
            per_query[name] = np.stack((ndcg.mean(axis=0), recall.mean(axis=0)), axis=1)
            means[name] = {'ndcg10': float(ndcg.mean()), 'recall100': float(recall.mean()),
                           'stored_repetitions': len(values),
                           'effective_repetitions': 1 if name in ('lexical_plain_native', 'lexical_planner_quota_native') else 3}
        stats = {'semantic': interval(per_query['semantic_native'], per_query['semantic_rank_only']),
                 'lexical': interval(per_query['lexical_native'], per_query['lexical_planner_order'])}
        path = output/(cfg['label']+'.metrics.npz')
        q.write_npz(path, **arrays, query_ids=ids, query_indices=indices, outer_holdout=np.ones(len(ids), dtype=bool))
        results[dataset] = {'outer_queries': len(ids), 'means': means,
                            'native_vs_rank_only': stats, 'metrics': q.binding(path),
                            'stage_rankings': q.binding(folder/'stage_rankings.npz'),
                            'primary_verification': q.binding(folder/'verification.json'),
                            'qrel_coverage': ds['qrel_document_map_coverage']}
        print(dataset, 'path-only diagnostics', stats, flush=True)
    q.write_json(output/'summary.json', {'schema': 'matched-component-path-diagnostics-v1',
        'status': 'computed_pending_independent_audit', 'datasets': results,
        'plan': q.binding(PLAN), 'script': q.binding(__file__), 'frozen_quality_runner': q.binding(HERE/'run_quality.py'),
        'qrel_metric_definition': 'Positive judged qrels, graded 2^gain-1 nDCG@10, Recall@100 from path first100.',
        'all_native_gains_measured_before_fusion': True, 'primary_seven_rows_unchanged': True,
        'new_retrieval_or_native_scoring_or_fit_performed': False,
        'interpretation': 'Explanatory additional analysis, reused outer cohort and common admission budgets; pointwise paired intervals, not new independent holdout or protocol timing.'})


if __name__ == '__main__':
    if not (HERE/'paired_results.json').exists():
        q.assemble()
    path_diagnostics()
