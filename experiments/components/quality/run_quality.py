"""Common-budget cached floating component quality; no protocol timing claims."""
from __future__ import annotations
import argparse, ctypes, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT.parent
PLAIN = BASE/'experiment_plaintext_float_matched_20261109'
RUNTIME = BASE/'experiment_pprag_precision_network_20261108'
PLAN = HERE/'QUALITY_PLAN.json'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


runtime_binding = module('component_quality_runtime', RUNTIME/'client_runtime_binding.py')
RUNTIME_INFO = runtime_binding.preload_client_runtime()
sys.path.insert(0, str(PLAIN/'lexical'))
import common as c
f = module('component_frozen_fusion', PLAIN/'run_fusion.py')
np = c.np
c.enforce_threads()
VARIANTS = ('full', 'late_route', 'plain_semantic', 'plain_lexical',
            'planner_quota_lexical', 'no_native', 'fixed_rrf')


def binding(path):
    path = Path(path).resolve(); st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': c.sha(path)}


def write_json(path, obj):
    path = Path(path).resolve()
    assert HERE in path.parents
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    with temp.open('x', encoding='utf-8') as h:
        json.dump(obj, h, ensure_ascii=False, indent=2, allow_nan=False); h.write('\n')
    temp.replace(path)


def write_npz(path, **values):
    path = Path(path).resolve(); assert HERE in path.parents and not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.npz')
    with temp.open('xb') as h:
        np.savez_compressed(h, **values)
    temp.replace(path)


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def available_memory():
    class M(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(x, ctypes.c_ulonglong) for x in ('total', 'available', 'page_total',
                       'page_available', 'virtual_total', 'virtual_available', 'extended')]]
    m = M(); m.length = ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.available)


def rank_major_unique(emitted):
    """Local rank first, frozen route ordinal second; stable first occurrence."""
    a = np.asarray(emitted).T.reshape(-1)
    a = a[a >= 0]
    _, first = np.unique(a, return_index=True)
    return np.asarray(a[np.sort(first)], dtype=np.int32)


def restore_planner(pool, selected):
    pool = c.native.distinct_valid(np.asarray(pool))
    selected = np.asarray(selected); selected = selected[selected >= 0]
    assert len(selected) == len(np.unique(selected)) and np.isin(selected, pool).all()
    out = pool[np.isin(pool, selected, assume_unique=True)]
    assert len(out) == len(selected)
    return out


def padded(rows, width=100):
    a = np.full(width, -1, dtype=np.int32)
    a[:min(width, len(rows))] = rows[:width]
    return a


def native_rank(rows, values, query):
    assert np.array_equal(rows, np.sort(rows)) and len(rows) == len(np.unique(rows))
    assert values.dtype == query.dtype == np.float32 and values.flags.c_contiguous
    scores = values @ query
    assert scores.dtype == np.float32 and np.isfinite(scores).all()
    order = np.lexsort((rows, -scores))[:100]
    return padded(rows[order]), scores


def synthetic():
    a = np.asarray([[9, 7, 3], [4, 9, -1], [9, 8, 2]], dtype=np.int32)
    assert rank_major_unique(a).tolist() == [9, 4, 7, 8, 3, 2]
    assert restore_planner(np.asarray([7, 3, 9, 2, -1]), np.asarray([2, 7, 9])).tolist() == [7, 9, 2]
    rng = np.random.default_rng(20261109)
    for _ in range(64):
        a = rng.integers(-1, 30, size=(8, 5), dtype=np.int32)
        expected = []; seen = set()
        for rank in range(5):
            for cell in range(8):
                row = int(a[cell, rank])
                if row >= 0 and row not in seen:
                    expected.append(row); seen.add(row)
        assert rank_major_unique(a).tolist() == expected
    rows = np.asarray([2, 7, 9], dtype=np.int32)
    values = np.zeros((3, 384), dtype=np.float32)
    rank, _ = native_rank(rows, values, np.zeros(384, dtype=np.float32))
    assert rank[:3].tolist() == [2, 7, 9]
    return {'rank_major_reference_cases': 65, 'planner_membership_order': True,
            'native_ties_by_global_row': True, 'waterfill': c.native.tests()}


def load_aligned(path, key, ds):
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], ds['query_ids']), str(path)
        if 'outer_holdout' in z:
            assert np.array_equal(z['outer_holdout'], ds['outer_holdout']), str(path)
        return np.asarray(z[key], dtype=np.int32)


def prepare(dataset, smoke, dest):
    ds = c.prepare(dataset); label = ds['cfg']['label']; n = len(ds['query_ids'])
    indices = np.flatnonzero(ds['outer_holdout'])
    if smoke:
        indices = indices[np.linspace(0, len(indices)-1, 4, dtype=int)]
    assert len(indices) > 0
    source_paths = [Path(__file__), PLAN, PLAIN/'run_fusion.py', PLAIN/'lexical/common.py',
                    RUNTIME/'client_runtime_binding.py', RUNTIME/'runtime_binding.json']
    original = c.FROZEN/'results'/label
    plain = PLAIN/'semantic/results'/label
    protected_path = original/'semantic_emitted.npz'
    plain_path = plain/'emitted.npz'
    sv = c.read(plain/'verification.json')
    assert sv['status'] == 'passed' and sv['emitted_sha256'] == c.sha(plain_path)
    with np.load(protected_path, allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], ds['query_ids'])
        route = z['query_cells']
        emitted = z['emitted'][:, indices].copy()
    with np.load(plain_path, allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], ds['query_ids'])
        assert np.array_equal(z['outer_holdout'], ds['outer_holdout'])
        assert np.array_equal(z['query_cells'], route)
        plain_emitted = z['emitted'][indices].copy()
    depth = ds['cfg']['semantic_depth']; qn = len(indices)
    assert emitted.shape == (3, qn, 128, depth) and plain_emitted.shape == (qn, 128, depth)
    assert route.shape == (n, 128)
    raw, trace, planner_bind = c.planner(ds)
    p_root = ds['protected_lexical_root']
    l_root = PLAIN/'lexical/results'/label
    q_root = c.PREVIOUS/'results'/label/'planner_quota'
    lexical = load_aligned(p_root/'rankings.npz', 'lexical_dpe', ds)[:, indices]
    selected = load_aligned(p_root/'candidates.npz', 'lexical_dpe', ds)[:, indices]
    plain_lexical = load_aligned(l_root/'rankings.npz', 'lexical', ds)[indices]
    plain_selected = load_aligned(l_root/'candidates.npz', 'lexical', ds)[indices]
    quota_lexical = load_aligned(q_root/'rankings.npz', 'lexical_dpe', ds)[:, indices]
    quota_selected = load_aligned(q_root/'candidates.npz', 'lexical_dpe', ds)[:, indices]
    native_path = c.FROZEN/'direct_bm25'/dataset/'direct_1000.rankings.npz'
    # The historical complete-pool BM25 archive predates embedded qids. Its
    # companion report binds the full ordered query-ID digest (also checked by
    # c.prepare); never silently infer alignment from shape alone.
    native_quality = c.FROZEN/'direct_bm25'/dataset/'quality.json'
    assert c.read(native_quality)['query_ids_sha256'] == ds['query_ids_sha256']
    with np.load(native_path, allow_pickle=False) as z:
        native_all = z['lexical_dpe']
        assert native_all.ndim == 3 and native_all.shape[1:] == (n, 1000)
        native_order = np.asarray(native_all[0, indices], dtype=np.int32)
    del native_all
    allocation_path = c.PREVIOUS/'results'/label/'allocation.npz'
    with np.load(allocation_path, allow_pickle=False) as z:
        assert np.array_equal(z['query_ids'], ds['query_ids'])
        quotas = z['quotas'][indices]
    geometry = c.GEOMETRY/'cache'/dataset/'r256'
    rows = np.load(geometry/'candidate_rows.npy', mmap_mode='r')
    cells = np.load(geometry/'cells.npy', mmap_mode='r')
    no_lexical = np.full_like(lexical, -1)
    counts = np.zeros(qn, dtype=np.int32)
    for oi, qi in enumerate(indices):
        pool = c.native.distinct_valid(raw[qi]); count = min(300, len(pool)); counts[oi] = count
        pos = np.searchsorted(rows, pool); assert np.array_equal(rows[pos], pool)
        pcells = np.asarray(cells[pos]); expected_quota = c.native.quota(np.bincount(pcells, minlength=16))
        assert np.array_equal(expected_quota, quotas[oi])
        expected_planner = c.native.choose_by_cell(pool, pcells, expected_quota)
        assert np.array_equal(np.sort(native_order[oi][native_order[oi] >= 0]), np.sort(pool))
        for r in range(3):
            variants = ((selected[r, oi], lexical[r, oi]),
                        (plain_selected[oi], plain_lexical[oi]),
                        (quota_selected[r, oi], quota_lexical[r, oi]))
            assert np.array_equal(quota_selected[r, oi][quota_selected[r, oi] >= 0], expected_planner)
            for chosen, ranked in variants:
                chosen = chosen[chosen >= 0]; ranked = ranked[ranked >= 0]
                assert len(chosen) == count == len(np.unique(chosen))
                p = np.searchsorted(rows, chosen); assert np.array_equal(rows[p], chosen)
                assert np.array_equal(np.bincount(cells[p], minlength=16), expected_quota)
                assert np.array_equal(c.native.order_subset(native_order[oi], chosen), ranked)
            no_lexical[r, oi] = padded(restore_planner(pool, selected[r, oi]), 300)
    folder, prior, model, model_binding = f.frozen(ds)
    replay = PLAIN/'protected_replay'/label/'verification.json'
    gate = c.read(replay)
    assert gate['status'] == 'passed' and gate['model'] == model_binding
    assert gate['saved_dual_rankings_exactly_reproduced']
    assert all(v == 0 for v in gate['max_abs_per_query_metric_errors'].values())
    source_paths += [protected_path, plain_path, plain/'verification.json', plain/'input_binding.json',
                    plain/'full_index_validation.json', original/'semantic.json',
                    *[p_root/x for x in ('rankings.npz', 'candidates.npz', 'quality.json', 'verification.json')],
                    *[l_root/x for x in ('rankings.npz', 'candidates.npz', 'verification.json')],
                    *[q_root/x for x in ('rankings.npz', 'candidates.npz', 'quality.json')],
                    native_path, native_quality, allocation_path, geometry/'candidate_rows.npy', geometry/'cells.npy',
                    folder/'fusion.json', folder/'duetrank.joblib', replay,
                    ds['semantic_source']/ds['cfg']['query_embeddings_file'],
                    c.u.SRC/'data'/dataset/'qrels'/(ds['cfg']['split']+'.tsv'),
                    c.u.SRC/'data'/dataset/'queries.jsonl']
    corpus_path = ds['semantic_source']/'corpus_embeddings.npy'
    original_binding = c.read(plain/'input_binding.json')
    corpus = original_binding['corpus_header_stat']
    st = corpus_path.stat()
    assert corpus['path'] == str(corpus_path.resolve()) and st.st_size == corpus['bytes'] and st.st_mtime_ns == corpus['mtime_ns']
    corpus_validation = c.read(plain/'full_index_validation.json')
    docmap = c.qrel_map(ds)
    assert ds['qrel_document_map_coverage']['absent_from_corpus'] == 0
    inputs = {str(p.resolve()): binding(p) for p in source_paths}
    info = {'schema': 'matched-budget-component-quality-input-v1', 'dataset': dataset,
            'evaluation_scope': 'four fixed evenly spaced outer queries' if smoke else 'all reused outer queries',
            'query_count': qn, 'full_query_count': n, 'query_indices': indices.tolist(),
            'query_ids_sha256': digest(np.asarray(ds['query_ids'])[indices]),
            'query_vectors_sha256': digest(ds['queries'][indices]),
            'route_sha256': digest(route[indices]), 'query_repeats': 3,
            'sources': inputs, 'historical_source_sha256': ds['source_hashes'], 'planner_sources': planner_bind,
            'corpus_header_stat': corpus, 'corpus_full_validation': corpus_validation,
            'model': model_binding, 'semantic_backoff_weight': float(prior['selection']['semantic_backoff_weight']),
            'fixed_rrf_semantic_weight': float(prior['selection']['fixed_rrf_semantic_weight']),
            'qrel_document_map_coverage': ds['qrel_document_map_coverage'],
            'runtime': RUNTIME_INFO, 'threadpools': c.enforce_threads(),
            'lexical_pool_quota_membership_native_subset_all_checked': True,
            'no_model_fit_or_selection': True}
    path = dest/'input_binding.json'
    if path.exists():
        previous = c.read(path)
        # Runtime metadata contains no timings; preserve exact immutable scientific bindings.
        assert previous == info
    else:
        write_json(path, info)
    ds.update(indices=indices, emitted=emitted, plain_emitted=plain_emitted, route=route[indices],
              lexical=lexical, plain_lexical=plain_lexical, quota_lexical=quota_lexical,
              no_lexical=no_lexical, lexical_counts=counts, model=model, model_binding=model_binding,
              backoff=info['semantic_backoff_weight'], fixed_weight=info['fixed_rrf_semantic_weight'],
              trace={k: [trace[k][int(i)] for i in indices] for k in ('terms', 'estimated_postings')},
              texts=[ds['query_texts'][int(i)] for i in indices], ids=[ds['query_ids'][int(i)] for i in indices],
              q=ds['queries'][indices].copy(), docmap=docmap, corpus_path=corpus_path,
              input_binding=info, input_binding_sha256=c.sha(path))
    return ds


def admissions(ds):
    n = len(ds['indices']); cap = 128*ds['cfg']['semantic_depth']
    full = np.full((3, n, cap), -1, dtype=np.int32)
    plain = np.full_like(full, -1)
    bstar = np.zeros((3, n), dtype=np.int32)
    full_count = np.zeros_like(bstar); plain_count = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = rank_major_unique(ds['plain_emitted'][i]); plain_count[i] = len(p)
        for r in range(3):
            a = rank_major_unique(ds['emitted'][r, i]); full_count[r, i] = len(a)
            b = min(len(a), len(p)); assert b >= 100
            bstar[r, i] = b; full[r, i, :b] = a[:b]; plain[r, i, :b] = p[:b]
    assert np.array_equal(bstar, np.minimum(full_count, plain_count[None, :]))
    assert np.array_equal((full >= 0).sum(axis=-1), bstar)
    assert np.array_equal((plain >= 0).sum(axis=-1), bstar)
    return full, plain, bstar, full_count, plain_count


def direct_gather(path, rows):
    m = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        assert m.dtype == np.float32 and m.shape[1] == 384 and m.flags.c_contiguous
        return np.array(m[rows], dtype=np.float32, copy=True, order='C')
    finally:
        m._mmap.close()


def compute_semantic(ds, dest, full, plain, bstar, smoke):
    n = full.shape[1]; output_full = np.full((3, n, 100), -1, dtype=np.int32)
    output_plain = np.full_like(output_full, -1)
    blocks = []; start = 0; total_gathered = 0; total_gather_seconds = 0.; total_native_seconds = 0.
    while start < n:
        stop = min(n, start+32)
        while True:
            union = np.unique(np.concatenate((full[:, start:stop].reshape(-1), plain[:, start:stop].reshape(-1))))
            union = union[union >= 0]
            if union.nbytes*384 <= 512*1024**2:
                break
            stop = start+max(1, (stop-start)//2)
        size = len(union)*384*4
        assert size <= 512*1024**2
        blockpath = dest/'native_blocks'/f'queries_{start:05d}_{stop:05d}.npz'
        reportpath = blockpath.with_suffix('.json')
        if blockpath.exists():
            report = c.read(reportpath)
            assert report['input_binding_sha256'] == ds['input_binding_sha256']
            assert report['sha256'] == c.sha(blockpath)
            with np.load(blockpath, allow_pickle=False) as z:
                assert np.array_equal(z['query_indices'], ds['indices'][start:stop])
                assert np.array_equal(z['full_candidates'], full[:, start:stop])
                assert np.array_equal(z['plain_candidates'], plain[:, start:stop])
                output_full[:, start:stop] = z['full_rank100']; output_plain[:, start:stop] = z['plain_rank100']
            blocks.append(report); start = stop; continue
        required = size+4*1024**3+2*65536*384*4
        available = available_memory()
        if available < required:
            raise MemoryError(f'Need {required} available bytes including 4GiB reserve; have {available}')
        cache = np.empty((len(union), 384), dtype=np.float32)
        t = time.perf_counter()
        for first in range(0, len(union), 65536):
            last = min(len(union), first+65536)
            chunk = direct_gather(ds['corpus_path'], union[first:last])
            cache[first:last] = chunk
            assert np.array_equal(cache[first:last].view(np.uint8), chunk.view(np.uint8))
            del chunk
        gather_seconds = time.perf_counter()-t
        t = time.perf_counter(); fixture_cases = 0
        for i in range(start, stop):
            for r in range(3):
                for candidates, output in ((full, output_full), (plain, output_plain)):
                    rows = np.sort(candidates[r, i, :bstar[r, i]])
                    p = np.searchsorted(union, rows); assert np.array_equal(union[p], rows)
                    matrix = np.array(cache[p], dtype=np.float32, order='C', copy=True)
                    rank, score = native_rank(rows, matrix, ds['q'][i])
                    if smoke:
                        raw = direct_gather(ds['corpus_path'], rows)
                        check_rank, check_score = native_rank(rows, raw, ds['q'][i])
                        assert np.array_equal(matrix.view(np.uint8), raw.view(np.uint8))
                        assert np.array_equal(score, check_score) and np.array_equal(rank, check_rank)
                        fixture_cases += 1
                        del raw, check_score, check_rank
                    output[r, i] = rank
                    del matrix, score
        native_seconds = time.perf_counter()-t
        del cache; gc.collect()
        write_npz(blockpath, query_indices=ds['indices'][start:stop],
                  full_candidates=full[:, start:stop], plain_candidates=plain[:, start:stop],
                  B_star=bstar[:, start:stop], full_rank100=output_full[:, start:stop], plain_rank100=output_plain[:, start:stop])
        report = {'status': 'passed', 'input_binding_sha256': ds['input_binding_sha256'],
                  'sha256': c.sha(blockpath), 'start': start, 'stop': stop, 'union_rows': len(union),
                  'union_bank_bytes': size, 'available_memory_before': available,
                  'source_gather_seconds_offline': gather_seconds, 'native_seconds_offline': native_seconds,
                  'raw_vs_cache_bitwise_cases': fixture_cases}
        write_json(reportpath, report); blocks.append(report)
        total_gathered += len(union); total_gather_seconds += gather_seconds; total_native_seconds += native_seconds
        print(ds['dataset'], 'native common-budget', stop, '/', n, 'gather', round(gather_seconds, 2),
              'score', round(native_seconds, 2), flush=True)
        start = stop
    return output_full, output_plain, {'blocks': blocks, 'new_gather_rows': total_gathered,
             'new_gather_seconds_offline': total_gather_seconds, 'new_native_seconds_offline': total_native_seconds}


def metrics(ds, rankings):
    ndcg = c.u.ndcg_rows(rankings, ds['docmap'], ds['ids'], ds['qrels'])
    recall = np.zeros(len(rankings), dtype=np.float64)
    independent = np.zeros_like(recall)
    for i, qid in enumerate(ds['ids']):
        gains = {ds['docmap'][d]: g for d, g in ds['qrels'][qid].items() if g > 0}
        actual = rankings[i]; valid = actual[actual >= 0]
        assert len(valid) == len(np.unique(valid)) and len(valid) <= 100
        recall[i] = len(set(valid.tolist()) & set(gains))/max(1, len(gains))
        ideal = sorted(gains.values(), reverse=True)[:10]
        denom = sum((2.**float(g)-1)/np.log2(j+2) for j, g in enumerate(ideal))
        numerator = sum((2.**float(gains.get(int(row), 0))-1)/np.log2(j+2) for j, row in enumerate(actual[:10]))
        independent[i] = numerator/denom if denom else 0.
    assert np.max(np.abs(ndcg-independent), initial=0) < 1e-14
    return ndcg, recall


def evaluate(ds, dest, sfull, splain, full_candidates, bstar, full_count, plain_count, io, smoke):
    n = len(ds['indices']); rankings = {k: np.full((3, n, 100), -1, dtype=np.int32) for k in VARIANTS}
    features = {}; ndcg = {}; recall = {}
    for r in range(3):
        inputs = {'full': (sfull[r], ds['lexical'][r]),
                  'plain_semantic': (splain[r], ds['lexical'][r]),
                  'plain_lexical': (sfull[r], ds['plain_lexical']),
                  'planner_quota_lexical': (sfull[r], ds['quota_lexical'][r]),
                  'no_native': (full_candidates[r, :, :100], ds['no_lexical'][r])}
        for name, (s, l) in inputs.items():
            ranks, feat = f.fused(s, l, ds['texts'], ds['trace'], ds['model'], ds['backoff'], ds['fixed_weight'])
            rankings[name][r] = ranks['duetrank']
            features.setdefault(name, []).append(feat)
            if name == 'full':
                rankings['fixed_rrf'][r] = ranks['fixed_rrf']
                rankings['late_route'][r] = ranks['duetrank']
        print(ds['dataset'], 'frozen fusion', r+1, '/3', flush=True)
    for name, values in rankings.items():
        pairs = [metrics(ds, row) for row in values]
        ndcg[name] = np.stack([x[0] for x in pairs]); recall[name] = np.stack([x[1] for x in pairs])
    assert np.array_equal(rankings['full'], rankings['late_route'])
    assert f.bind(ds['protected_fusion_root']/'duetrank.joblib') == ds['model_binding']
    shared = {'query_ids': np.asarray(ds['ids']), 'query_indices': ds['indices'], 'outer_holdout': np.ones(n, dtype=bool)}
    write_npz(dest/'rankings.npz', **rankings, **shared)
    write_npz(dest/'metrics.npz', **{k+'_ndcg10': v for k, v in ndcg.items()}, **{k+'_recall100': v for k, v in recall.items()}, **shared)
    write_npz(dest/'query_features.npz', **{k: np.stack(v) for k, v in features.items()}, **shared)
    write_npz(dest/'budgets.npz', B_star=bstar, full_unique_before=full_count, plain_unique_before=plain_count,
              lexical_valid=ds['lexical_counts'], **shared)
    write_npz(dest/'stage_rankings.npz', semantic_full=sfull, semantic_plain=splain,
              semantic_no_native=full_candidates[:, :, :100], lexical_full=ds['lexical'],
              lexical_plain=ds['plain_lexical'], lexical_planner_quota=ds['quota_lexical'],
              lexical_no_native=ds['no_lexical'], **shared)
    result = {'schema': 'matched-budget-component-quality-result-v1', 'status': 'passed' if smoke else 'computed_pending_independent_audit',
              'dataset': ds['dataset'], 'queries': n, 'outer_queries': n, 'noise_repeats': 3,
              'input_binding_sha256': ds['input_binding_sha256'], 'plan_sha256': c.sha(PLAN), 'script_sha256': c.sha(__file__),
              'model': ds['model_binding'], 'semantic_backoff_weight': ds['backoff'], 'fixed_rrf_semantic_weight': ds['fixed_weight'],
              'means': {k: {'ndcg10': float(ndcg[k].mean()), 'recall100': float(recall[k].mean())} for k in VARIANTS},
              'budgets': {'B_star': c.u.stats(bstar.reshape(-1)), 'full_unique_before': c.u.stats(full_count.reshape(-1)),
                          'plain_unique_before': c.u.stats(plain_count), 'lexical_valid': c.u.stats(ds['lexical_counts'])},
              'same_actual_admitted_count_every_variant': True,
              'actual_native_scoring_definition': 'B_star semantic and lexical_valid BM25, except no_native=0 for both; cached lexical ranks are reused, not newly timed.',
              'late_route_quality_exactly_full': True, 'new_common_budget_full_not_old_table_II': True,
              'all_five_fusion_features_recomputed_no_fit': True, 'qrel_coverage': ds['qrel_document_map_coverage'],
              'source_native_precision': 'FP32 original vectors and cached queries; separate from cost protocol returned FP16',
              'io': io, 'no_online_timing_or_privacy_claim': True,
              'outputs': {name: binding(dest/name) for name in ('rankings.npz', 'metrics.npz', 'budgets.npz', 'stage_rankings.npz', 'query_features.npz')}}
    write_json(dest/'verification.json', result)
    print(ds['dataset'], 'quality', result['status'], result['means'], flush=True)


def run(dataset, smoke, release):
    if not smoke:
        gate = c.read(release)
        assert gate['status'] == 'released' and gate['formal_timing_active'] is False
        assert gate['script_sha256'] == c.sha(__file__) and gate['plan_sha256'] == c.sha(PLAN)
        assert dataset in gate['datasets']
        test = HERE/'smoke'/c.u.C['datasets'][dataset]['label']/'verification.json'
        checked = c.read(test)
        assert checked['status'] == 'passed' and checked['script_sha256'] == c.sha(__file__)
        assert gate['smoke_sha256'][dataset] == c.sha(test)
    dest = HERE/('smoke' if smoke else 'results')/c.u.C['datasets'][dataset]['label']
    if (dest/'verification.json').exists():
        raise FileExistsError(dest/'verification.json')
    dest.mkdir(parents=True, exist_ok=True)
    ds = prepare(dataset, smoke, dest)
    full, plain, budgets, full_count, plain_count = admissions(ds)
    # Selection is complete and frozen before opening original document vectors.
    del ds['emitted'], ds['plain_emitted']; gc.collect()
    sf, sp, io = compute_semantic(ds, dest, full, plain, budgets, smoke)
    evaluate(ds, dest, sf, sp, full, budgets, full_count, plain_count, io, smoke)


def assemble():
    datasets = {}; all_diffs = []
    for dataset, cfg in c.u.C['datasets'].items():
        dest = HERE/'results'/cfg['label']; report = c.read(dest/'verification.json')
        assert report['status'] in ('computed_pending_independent_audit', 'passed')
        assert report['script_sha256'] == c.sha(__file__) and report['plan_sha256'] == c.sha(PLAN)
        assert report['outputs']['metrics.npz']['sha256'] == c.sha(dest/'metrics.npz')
        with np.load(dest/'metrics.npz', allow_pickle=False) as z:
            assert z['outer_holdout'].all()
            qids = z['query_ids']; values = np.stack([np.stack([z[k+'_ndcg10'].mean(axis=0), z[k+'_recall100'].mean(axis=0)], axis=1) for k in VARIANTS], axis=1)
        assert len(qids) == cfg['outer_holdout_queries'] and values.shape == (len(qids), 7, 2)
        rng = np.random.default_rng(20261109); means = values.mean(axis=0)
        diffs = np.empty((10000, 7, 2)); ratios = np.empty_like(diffs)
        for start in range(0, 10000, 64):
            stop = min(10000, start+64); ix = rng.integers(0, len(values), size=(stop-start, len(values)))
            m = values[ix].mean(axis=1)
            diffs[start:stop] = m-m[:, :1]
            ratios[start:stop] = m/m[:, :1]
        ci = np.percentile(diffs, [2.5, 97.5], axis=0); rci = np.percentile(ratios, [2.5, 97.5], axis=0)
        datasets[dataset] = {'queries': len(qids), 'metrics': {
            k: {metric: {'mean': float(means[j, mi]), 'difference_from_full': float(means[j, mi]-means[0, mi]),
                        'paired_difference_ci95': ci[:, j, mi].tolist(), 'ratio_to_full': float(means[j, mi]/means[0, mi]),
                        'paired_ratio_ci95': rci[:, j, mi].tolist()}
                for mi, metric in enumerate(('ndcg10', 'recall100'))} for j, k in enumerate(VARIANTS)},
            'budgets': report['budgets'], 'verification': binding(dest/'verification.json')}
        print(dataset, 'paired 10000-query-bootstrap complete', flush=True)
    write_json(HERE/'paired_results.json', {'schema': 'matched-budget-components-paired-quality-v1',
        'status': 'computed_pending_independent_audit', 'datasets': datasets, 'outer_queries': 9005,
        'repeats_averaged_before_bootstrap': 3, 'bootstrap_samples': 10000, 'bootstrap_seed': 20261109,
        'plan': binding(PLAN), 'runner': binding(__file__),
        'scope': 'Reused outer cohort, frozen model/index, new common-budget Full; cached FP32 utility only, no measured protocol costs or equivalence claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('synthetic', 'smoke', 'full', 'assemble'), required=True)
    parser.add_argument('--datasets', nargs='+', choices=list(c.u.C['datasets']), default=list(c.u.C['datasets']))
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.stage == 'synthetic':
        print(json.dumps(synthetic(), indent=2))
    elif args.stage == 'assemble':
        assemble()
    else:
        synthetic()
        if args.stage == 'full' and args.release is None:
            parser.error('--full requires --release')
        for dataset in args.datasets:
            run(dataset, args.stage == 'smoke', args.release)
