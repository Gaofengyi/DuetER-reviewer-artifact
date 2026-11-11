"""Transparent IO-only wrapper around the immutable e9e6659c quality runner.

The native score loop and all scientific functions remain the original source.
Only block grouping and loading completed immutable blocks are replaced.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, inspect, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE/'run_quality.py'
ORIGINAL_SHA = 'e9e6659c533cd241baf4465c91b99a5c9a0c7680b18bc206fbc52a842830e0a9'
sys.dont_write_bytecode = True
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
spec = importlib.util.spec_from_file_location('component_original_quality', ORIGINAL)
q = importlib.util.module_from_spec(spec); sys.modules[spec.name] = q; spec.loader.exec_module(q)
np = q.np
IO = HERE/'io128'


def bytes_digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def load_completed(ds, dest, full, plain, bstar, sf, sp):
    reports = []; start = 0
    for report_path in sorted((dest/'native_blocks').glob('queries_*.json')):
        report = q.c.read(report_path); path = report_path.with_suffix('.npz')
        assert report['status'] == 'passed' and report['input_binding_sha256'] == ds['input_binding_sha256']
        assert report['start'] == start and report['stop'] > start
        stop = report['stop']; assert report['sha256'] == q.c.sha(path)
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['query_indices'], ds['indices'][start:stop])
            assert np.array_equal(z['B_star'], bstar[:, start:stop])
            assert np.array_equal(z['full_candidates'], full[:, start:stop])
            assert np.array_equal(z['plain_candidates'], plain[:, start:stop])
            sf[:, start:stop] = z['full_rank100']; sp[:, start:stop] = z['plain_rank100']
        reports.append(report); start = stop
    print(ds['dataset'], 'loaded complete immutable native blocks through', start, flush=True)
    return reports, start


def amended_source():
    original = inspect.getsource(q.compute_semantic)
    old1 = 'blocks = []; start = 0; total_gathered = 0; total_gather_seconds = 0.; total_native_seconds = 0.'
    new1 = 'blocks, start = _io128_load_completed(ds, dest, full, plain, bstar, output_full, output_plain); total_gathered = 0; total_gather_seconds = 0.; total_native_seconds = 0.'
    old2 = 'stop = min(n, start+32)'
    new2 = 'stop = min(n, start+128)'
    assert original.count(old1) == original.count(old2) == 1
    changed = original.replace(old1, new1).replace(old2, new2)
    assert changed.replace(new1, old1).replace(new2, old2) == original
    return original, changed


def prepare_amendment():
    original, changed = amended_source()
    hot = HERE/'results/hotpotqa/native_blocks'
    reports = sorted(hot.glob('*.json'))
    assert reports
    last = q.c.read(reports[-1]); assert last['stop'] >= 256
    result = {'schema': 'component-quality-prospective-io-amendment-v1', 'status': 'proposed_before_fixture',
        'authorization': 'Root authorized IO-only max_queries128 with unchanged 512MiB bank limit and 4GiB reserve; adopt only if predeclared two-block benchmark improves gather time and exact numerical closure passes.',
        'original_runner': q.binding(ORIGINAL), 'wrapper': q.binding(__file__),
        'original_compute_function_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'amended_compute_function_sha256': hashlib.sha256(changed.encode()).hexdigest(),
        'exact_source_edits': ['Load contiguous existing completed blocks into the output arrays before the loop.', 'Change maximum query group from 32 to 128; retain existing automatic halving until union bank <=512MiB.'],
        'unchanged': ['admitted candidate IDs and B_star', 'per-query sorted rows and C-contiguous FP32 bytes',
                      'query/repeat/selector GEMV call order', 'native dot and global-row tie sorting',
                      'all lexical ranks, frozen model/backoff/RRF', 'all metrics, outer queries and bootstrap'],
        'completed_nq_verification': q.binding(HERE/'results/nq/verification.json'),
        'preserved_original_log': q.binding(HERE/'full.log'),
        'last_completed_hot_query_ordinal': last['stop'], 'last_completed_hot_block': q.binding(reports[-1]),
        'original_worker_stopped': {'pid': 27304, 'reason': 'Root-authorized IO-only restart after complete checkpoint; no numerical failure or quality-based stopping.',
                                    'all_existing_artifacts_retained': True, 'incomplete_block_artifacts_found': False},
        'fixture': {'dataset': 'hotpotqa', 'ranges': [[0, 128], [128, 256]],
                    'orders': [['32', '128'], ['128', '32']],
                    'checks': 'Every query/repeat/selector matrix byte hash, FP32 score byte hash and Top100 rank equals both backends and the already-completed original rank artifact.',
                    'adoption_rule': 'All exact checks pass and total gather_seconds for two 128-query banks is below the total for eight 32-query banks. No outcome-based choice.'},
        'no_online_performance_claim': True}
    q.write_json(IO/'amendment.json', result)
    q.write_json(IO/'source_transform.json', {'original_function': original, 'amended_function': changed})
    print('IO amendment frozen', q.c.sha(IO/'amendment.json'), flush=True)


def read_fixture_data():
    cfg = q.c.u.C['datasets']['hotpotqa']
    folder = HERE/'results/hotpotqa'
    parts = []
    for start in range(0, 256, 32):
        path = folder/'native_blocks'/f'queries_{start:05d}_{start+32:05d}.npz'
        info = q.c.read(path.with_suffix('.json'))
        assert info['status'] == 'passed' and info['sha256'] == q.c.sha(path)
        with np.load(path, allow_pickle=False) as z:
            parts.append({k: z[k].copy() for k in z.files})
    data = {k: np.concatenate([p[k] for p in parts], axis=0 if k == 'query_indices' else 1) for k in parts[0]}
    source = q.c.u.SRC/cfg['semantic_cache']
    queries = np.load(source/cfg['query_embeddings_file'], mmap_mode='r')[data['query_indices']].copy()
    return data, queries, source/'corpus_embeddings.npy'


def fixture_backend(data, queries, path, first, last, width):
    all_hashes = {}; gather_seconds = 0.; logical_rows = 0; peak_bank = 0
    for start in range(first, last, width):
        stop = min(last, start+width)
        union = np.unique(np.concatenate([data[k][:, start:stop].reshape(-1) for k in ('full_candidates', 'plain_candidates')]))
        union = union[union >= 0]
        size = len(union)*384*4
        assert size <= 512*1024**2
        assert q.available_memory() >= size+4*1024**3+2*65536*384*4
        cache = np.empty((len(union), 384), dtype=np.float32)
        t = time.perf_counter()
        for a in range(0, len(union), 65536):
            b = min(len(union), a+65536)
            chunk = q.direct_gather(path, union[a:b]); cache[a:b] = chunk
            assert np.array_equal(cache[a:b].view(np.uint8), chunk.view(np.uint8))
            del chunk
        elapsed = time.perf_counter()-t; gather_seconds += elapsed
        logical_rows += len(union); peak_bank = max(peak_bank, size)
        for i in range(start, stop):
            for r in range(3):
                for which, ref in (('full_candidates', 'full_rank100'), ('plain_candidates', 'plain_rank100')):
                    rows = np.sort(data[which][r, i, :data['B_star'][r, i]])
                    positions = np.searchsorted(union, rows); assert np.array_equal(union[positions], rows)
                    values = np.array(cache[positions], dtype=np.float32, order='C', copy=True)
                    rank, score = q.native_rank(rows, values, queries[i])
                    assert np.array_equal(rank, data[ref][r, i]), (i, r, which)
                    all_hashes[f'{i}:{r}:{which}'] = [bytes_digest(values), bytes_digest(score), bytes_digest(rank)]
                    del values, score
        del cache; gc.collect()
        print('IO fixture', width, start, stop, 'gather', round(elapsed, 3), flush=True)
    return all_hashes, {'query_start': first, 'query_stop': last, 'max_queries': width,
                       'gather_seconds': gather_seconds, 'source_gather_rows': logical_rows,
                       'peak_union_bank_bytes': peak_bank, 'cases': len(all_hashes)}


def fixture():
    amendment = q.c.read(IO/'amendment.json')
    assert amendment['wrapper']['sha256'] == q.c.sha(__file__)
    assert amendment['original_runner']['sha256'] == ORIGINAL_SHA
    data, queries, path = read_fixture_data()
    reports = []; total_cases = 0
    for first, last, order in ((0, 128, (32, 128)), (128, 256, (128, 32))):
        results = {}
        for width in order:
            results[width] = fixture_backend(data, queries, path, first, last, width)
            reports.append(results[width][1])
        assert results[32][0] == results[128][0]
        total_cases += len(results[32][0])
    old = sum(x['gather_seconds'] for x in reports if x['max_queries'] == 32)
    new = sum(x['gather_seconds'] for x in reports if x['max_queries'] == 128)
    result = {'schema': 'component-quality-io128-fixture-v1', 'status': 'passed',
        'amendment_sha256': q.c.sha(IO/'amendment.json'), 'wrapper_sha256': q.c.sha(__file__),
        'original_runner_sha256': ORIGINAL_SHA, 'cases': total_cases,
        'matrix_bytes_and_FP32_score_bytes_and_rank_all_identical': True,
        'all_ranks_equal_original_completed_checkpoints': True, 'benchmark_blocks': reports,
        'old_32_gather_seconds': old, 'new_128_gather_seconds': new, 'gather_speedup': old/new,
        'adopt_io128': new < old, 'scientific_parameters_unchanged': True,
        'scope': 'Fixed operational IO fixture with AB/BA order, not protocol timing, not a reported scientific speedup.'}
    q.write_json(IO/'fixture.json', result)
    print(json.dumps(result), flush=True)


def resume(datasets):
    amendment = q.c.read(IO/'amendment.json'); gate = q.c.read(IO/'fixture.json')
    assert gate['status'] == 'passed' and gate['adopt_io128']
    assert gate['wrapper_sha256'] == q.c.sha(__file__) == amendment['wrapper']['sha256']
    assert gate['original_runner_sha256'] == ORIGINAL_SHA
    assert gate['amendment_sha256'] == q.c.sha(IO/'amendment.json')
    original, changed = amended_source()
    assert hashlib.sha256(changed.encode()).hexdigest() == amendment['amended_compute_function_sha256']
    q._io128_load_completed = load_completed
    exec(compile(changed, str(__file__)+'::compute_semantic_io128', 'exec'), q.__dict__)
    for dataset in datasets:
        q.run(dataset, False, HERE/'release.json')
        dest = HERE/'results'/q.c.u.C['datasets'][dataset]['label']
        q.write_json(dest/'execution_backend.json', {'schema': 'component-quality-io-execution-v1',
            'status': 'passed', 'dataset': dataset, 'original_runner': q.binding(ORIGINAL),
            'io_wrapper': q.binding(__file__), 'amendment': q.binding(IO/'amendment.json'),
            'fixture': q.binding(IO/'fixture.json'), 'verification': q.binding(dest/'verification.json'),
            'max_new_queries_per_block': 128, 'max_union_bank_bytes': 512*1024**2,
            'original_complete_native_blocks_loaded_and_validated': True,
            'scientific_computation': 'Original function source except max query group and immutable-block loader; original score loop and all fusion/metrics source remain unchanged.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('prepare', 'fixture', 'resume'), required=True)
    p.add_argument('--datasets', nargs='+', choices=('hotpotqa', 'msmarco'), default=['hotpotqa', 'msmarco'])
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare_amendment()
    elif a.stage == 'fixture':
        fixture()
    else:
        resume(a.datasets)
