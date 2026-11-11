"""Direct native-BM25 controls on the identical frozen FTS5 planner pool.

No CCADPE pruning.  300 candidates matches client scoring work; 1000 is a
larger-compute control.  Timings here are explicitly kernel-only, not protocol.
The query-vocabulary CSR is an offline evaluation acceleration, never a claim
that these unprotected matrices are the online encrypted payload.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['DUETER_FORCE_CPU_DEPS'] = '1'
import sys, json, hashlib, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / 'config.json'
CFG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
ROOT = Path(CFG['original_root'])
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT)]
import numpy as np
from benchmark_exact_bm25_client_rerank import (
    selected_document_rows, query_vocabulary, load_document_frequencies,
    build_candidate_term_matrix, load_document_lengths, exact_bm25_rerank,
)
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from benchmark_dual_compartment_full import metric_mean
from experiment_candidate_rank_fusion import split_mask, paired_bootstrap
from experiment_adaptive_fusion import ndcg_rows

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def run(dataset):
    started = time.perf_counter()
    c = CFG['datasets'][dataset]
    dest = HERE / 'direct_bm25' / dataset
    dest.mkdir(parents=True, exist_ok=True)
    qids, texts, qrels, _ = load_queries_qrels(ROOT / 'data' / dataset, c['split'])
    assert len(qids) == c['queries']
    raw = np.asarray(np.load(ROOT / c['planner_cache'])[f"budget_{c['posting_budget']}"][:, :CFG['lexical']['planner_cap']], dtype=np.int32)
    assert raw.shape == (len(qids), 1000)
    rows = selected_document_rows(raw)
    terms, term_to_index = query_vocabulary(texts)
    cache = dest / 'payload_cache'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': dataset, 'query_ids_sha256': digest(qids),
                'vocabulary_sha256': digest(terms), 'vocabulary_size': len(terms),
                'row_count': len(rows), 'row_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
                'planner_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                'description': 'Offline exact TF CSR for union of all first 1000 planner candidates; no CCADPE selection.'}
    mp = cache / 'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text(encoding='utf-8')) == manifest, 'incompatible cache'
    atomic_json(mp, manifest)
    print(f'[{dataset}] building payload: {len(rows):,} documents, {len(terms):,} terms', flush=True)
    lengths, indptr, indices, counts = build_candidate_term_matrix(ROOT / 'data' / dataset / 'corpus.jsonl', rows, term_to_index, cache)
    assert np.array_equal(np.load(cache / 'rows.npy'), rows)
    df = load_document_frequencies(ROOT / c['fts_database'], terms)
    np.save(cache / 'df.npy', df)
    full_lengths = load_document_lengths(ROOT / c['fts_database'], ROOT / 'cache' / 'exact_bm25_client_rerank' / c['label'] / 'full_doc_lengths_v2.npy')
    assert len(full_lengths) == c['documents']
    average_length = float(np.mean(full_lengths))
    atomic_json(cache / 'ready.json', {**manifest, 'average_length': average_length, 'k1': 1.2, 'b': 0.75, 'query_term_frequency': 'binary'})
    print(f'[{dataset}] payload ready', flush=True)
    ids = read_ids(ROOT / c['semantic_cache'] / 'doc_ids.txt')
    id_to_row = {v: i for i, v in enumerate(ids)}
    holdout = split_mask(qids, 'outer')
    assert int(holdout.sum()) == c['outer_holdout_queries']
    report = {'dataset': dataset, 'config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
              'query_ids_sha256': digest(qids), 'holdout_queries': int(holdout.sum()),
              'queries': len(qids), 'average_document_length': average_length,
              'controls': {}, 'timing_scope': 'Exact BM25 kernel only; excludes planner, payload construction/loading, tokenization, serialization, transport, decryption, and Fetch.'}
    outputs = {}
    for cap in (300, 1000):
        candidates = raw[None, :, :cap]
        print(f'[{dataset}] scoring direct BM25 cap={cap}', flush=True)
        ranking, timing = exact_bm25_rerank(candidates, rows, lengths, indptr, indices, counts, texts, term_to_index, df, c['documents'], average_length)
        outputs[cap] = ranking
        np.savez_compressed(dest / f'direct_{cap}.rankings.npz', lexical_dpe=ranking)
        per_query = ndcg_rows(ranking[0], id_to_row, qids, qrels)
        np.save(dest / f'direct_{cap}.ndcg.npy', per_query)
        report['controls'][str(cap)] = {'candidate_rule': f'First <= {cap} FTS5 planner candidates; no CCADPE filtering; exact native BM25',
            'scored_candidates_mean': float(np.mean((candidates >= 0).sum(axis=-1))),
            'scored_candidates_max': int((candidates >= 0).sum(axis=-1).max()),
            'all_queries': metric_mean(ranking, ids, qids, qrels),
            'heldout_ndcg_at_10': float(per_query[holdout].mean()),
            'calibration_ndcg_at_10': float(per_query[~holdout].mean()), 'bm25_kernel_ms': timing}
        atomic_json(dest / 'quality.json', report)
    # All common documents have identical native scores and deterministic row-ID ties.
    for qi in range(len(qids)):
        small = outputs[300][0, qi]
        small = small[small >= 0]
        large = outputs[1000][0, qi]
        large = large[large >= 0]
        restricted = large[np.isin(large, small, assume_unique=True)]
        assert np.array_equal(small, restricted), f'inconsistent scores at query {qi}'
    report['verification'] = {'all_queries_subset_rank_invariant': True, 'input_shape': list(raw.shape), 'scorer_k1': 1.2, 'scorer_b': .75, 'query_term_frequency': 'binary'}
    report['elapsed_seconds_including_offline_materialization'] = time.perf_counter() - started
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(dest / 'quality.json', report)
    print(f'[{dataset}] complete: {json.dumps(report["controls"])}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(CFG['datasets']), default=list(CFG['datasets']))
    args = parser.parse_args()
    for dataset in args.datasets:
        run(dataset)
