"""One-path offline SSD bridge outside the frozen scientific/IO128 runners."""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.util, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SSD_DIR = HERE/'ssd_cache'
PLAN = SSD_DIR/'PLAN.json'
BRIDGE = ROOT/'native_cache_bridge.py'
REVIEW = ROOT/'audits/quality_native_cache_wrapper_review.json'
ORIGINAL = HERE/'run_quality.py'
IO128 = HERE/'native_io128.py'
ORIGINAL_SHA = 'e9e6659c533cd241baf4465c91b99a5c9a0c7680b18bc206fbc52a842830e0a9'
IO128_SHA = 'a16b9850c9e84677c54025ed0cb95a87e6677ae967b457212d23dc271c12d448'
RUNTIME = ROOT.parent/'experiment_pprag_precision_network_20261108'
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def spec(path):
    path = Path(path).resolve(); st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha(path)}


def write(path, obj, update=False):
    path = Path(path).resolve(); assert HERE in path.parents
    if path.exists() and not update:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    temp.replace(path)


def load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = value; module_spec.loader.exec_module(value)
    return value


def preflight():
    assert sha(ORIGINAL) == ORIGINAL_SHA and sha(IO128) == IO128_SHA
    audit = read(REVIEW)
    assert audit['status'] == 'passed' and not audit.get('open_findings')
    assert Path(audit['wrapper']['path']).resolve() == Path(__file__).resolve()
    assert audit['wrapper']['sha256'] == sha(__file__)
    bridge = load_module('component_quality_shared_cache_bridge', BRIDGE)
    bridge.enforce_review()
    binding = bridge.checked_binding(); plan = read(PLAN)
    assert Path(binding['source']['path']).resolve() == Path(plan['source']).resolve()
    assert Path(binding['cache']['path']).resolve() == Path(plan['cache']).resolve()
    assert binding['source']['bytes'] == binding['cache']['bytes'] == plan['expected_bytes']
    assert binding['source']['sha256'] == binding['cache']['sha256'] == plan['expected_sha256']
    runtime = load_module('component_quality_SSD_runtime', RUNTIME/'client_runtime_binding.py')
    runtime_info = runtime.preload_client_runtime()
    import numpy as np
    assert np.__version__ == '2.5.2'
    evidence = {'wrapper': spec(__file__), 'plan': spec(PLAN), 'original_runner': spec(ORIGINAL),
        'io128_wrapper': spec(IO128), 'shared_bridge': spec(BRIDGE), 'native_cache_binding': spec(bridge.BINDING),
        'independent_wrapper_review': spec(REVIEW), 'independent_bridge_review': spec(bridge.REVIEW),
        'source_native': binding['source'], 'actual_native_read_file': binding['cache'], 'runtime': runtime_info}
    return bridge, binding, np, evidence


def check_observations(observations, binding):
    assert observations
    for event in observations:
        assert Path(event['requested_path']).resolve() == Path(binding['source']['path']).resolve()
        assert Path(event['actual_read_path']).resolve() == Path(binding['cache']['path']).resolve()
        assert event['mmap_mode'] == 'r' and event['other_arguments_preserved']


def fixture():
    bridge, binding, np, evidence = preflight()
    path = SSD_DIR/'fixture.json'
    if path.exists():
        raise FileExistsError(path)
    old_load = np.load; events = []
    blockpath = HERE/'results/msmarco_dev/native_blocks/queries_00000_00064.npz'
    blockreport = read(blockpath.with_suffix('.json'))
    assert blockreport['status'] == 'passed' and blockreport['sha256'] == sha(blockpath)
    with old_load(blockpath, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in z.files}
    cases = 0; observations = []
    try:
        with bridge.redirected_load(np, binding['source']['path'], binding['cache']['path'], events, guard=bridge.checked_binding):
            # The context is active before either frozen runner is imported.
            io = load_module('component_quality_SSD_IO128_fixture', IO128)
            query_path = io.q.c.u.SRC/io.q.c.u.C['datasets']['msmarco']['semantic_cache']/'query_embeddings_dev.npy'
            input_path = HERE/'results/msmarco_dev/input_binding.json'
            prior_input = read(input_path)
            assert sha(input_path) == blockreport['input_binding_sha256']
            assert sha(query_path) == prior_input['sources'][str(query_path.resolve())]['sha256']
            query_bank = np.load(query_path, mmap_mode='r', allow_pickle=False)
            try:
                queries = np.asarray(query_bank[data['query_indices'][[0, 63]]], dtype=np.float32).copy()
            finally:
                query_bank._mmap.close()
            for j, qi in enumerate((0, 63)):
                for repeat in range(3):
                    for candidate_key, rank_key in (('full_candidates', 'full_rank100'), ('plain_candidates', 'plain_rank100')):
                        rows = np.sort(data[candidate_key][repeat, qi, :data['B_star'][repeat, qi]])
                        native = old_load(binding['source']['path'], mmap_mode='r', allow_pickle=False)
                        try:
                            expected = np.array(native[rows], dtype=np.float32, copy=True, order='C')
                        finally:
                            native._mmap.close()
                        actual = io.q.direct_gather(Path(binding['source']['path']), rows)
                        assert np.array_equal(expected.view(np.uint8), actual.view(np.uint8))
                        rank_e, score_e = io.q.native_rank(rows, expected, queries[j])
                        rank_c, score_c = io.q.native_rank(rows, actual, queries[j])
                        assert np.array_equal(score_e.view(np.uint8), score_c.view(np.uint8))
                        assert np.array_equal(rank_e, rank_c) and np.array_equal(rank_c, data[rank_key][repeat, qi])
                        cases += 1
                        observations.append({'original_full_query_index': int(data['query_indices'][qi]), 'repeat': repeat,
                            'candidates': candidate_key, 'candidate_count': len(rows), 'bytes_score_rank_exact': True})
            assert cases == 12 and len(events) == cases
    finally:
        assert np.load is old_load, 'Shared NumPy loader was not restored'
    check_observations(events, binding); bridge.checked_binding()
    assert sha(bridge.BINDING) == evidence['native_cache_binding']['sha256']
    write(path, {'schema': 'component-quality-SSD-fixture-v1', 'status': 'passed', 'cases': cases,
        'all_matrix_bytes_FP32_scores_and_old_checkpoint_Top100_exact': True, 'numpy_loader_restored': True,
        'redirected_numpy_loads': events, 'case_checks': observations,
        'existing_native_block': spec(blockpath), 'existing_native_block_report': spec(blockpath.with_suffix('.json')),
        'original_input_binding': spec(input_path), 'query_embeddings': spec(query_path),
        'bindings': evidence, 'scope': 'Only two already-completed MS queries; no new selection, fitting or quality outcome.'})
    print(json.dumps({'status': 'passed', 'cases': cases, 'fixture_sha256': sha(path)}), flush=True)


def resume(attempt=1):
    assert isinstance(attempt, int) and attempt >= 1
    bridge, binding, np, evidence = preflight()
    fixture_path = SSD_DIR/'fixture.json'; gate = read(fixture_path)
    assert gate['status'] == 'passed' and gate['cases'] == 12
    assert gate['bindings']['wrapper'] == evidence['wrapper']
    assert gate['bindings']['native_cache_binding'] == evidence['native_cache_binding']
    output = SSD_DIR/f'execution_attempt{attempt}.json'
    if output.exists():
        raise FileExistsError(output)
    records = sorted((HERE/'results/msmarco_dev/native_blocks').glob('queries_*.json'))
    start = read(records[-1])['stop'] if records else 0
    events = []; old_load = np.load
    runtime = {'schema': 'component-quality-SSD-execution-v1', 'status': 'running', 'dataset': 'msmarco',
        'attempt': attempt, 'process_id': os.getpid(), 'actual_argv': sys.argv, 'bindings': evidence, 'fixture': spec(fixture_path),
        'start_complete_query_count': start, 'memory_guard_changed': False,
        'scientific_function_or_model_or_input_changed': False, 'formal_timing': False,
        'redirected_numpy_loads': events, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(output, runtime)
    try:
        with bridge.redirected_load(np, binding['source']['path'], binding['cache']['path'], events, guard=bridge.checked_binding):
            io = load_module('component_quality_SSD_IO128_resume', IO128)
            io.resume(['msmarco'])
        assert np.load is old_load
        check_observations(events, binding); bridge.checked_binding()
        assert sha(bridge.BINDING) == evidence['native_cache_binding']['sha256']
        assert sha(__file__) == evidence['wrapper']['sha256']
        dest = HERE/'results/msmarco_dev'
        runtime.update(status='complete', end_complete_query_count=3526, numpy_loader_restored=True,
            redirect_count=len(events), completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            verification=spec(dest/'verification.json'), underlying_IO128_backend=spec(dest/'execution_backend.json'))
        write(output, runtime, update=True)
        write(dest/'execution_ssd_backend.json', {'status': 'passed', 'scope': 'Additional offline read-path provenance; original scientific results and IO128 backend sidecar remain unchanged.',
            'SSD_execution': spec(output), 'source_native': binding['source'], 'actual_native_read_file': binding['cache'],
            'underlying_IO128_backend': runtime['underlying_IO128_backend'], 'verification': runtime['verification']})
        print(json.dumps({'status': 'complete', 'redirect_count': len(events), 'execution_sha256': sha(output)}), flush=True)
    except BaseException as error:
        runtime.update(status='failed', error_type=type(error).__name__, error=str(error),
                       numpy_loader_restored=np.load is old_load, redirect_count=len(events))
        write(output, runtime, update=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('fixture', 'resume'), required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.stage == 'fixture':
        fixture()
    else:
        resume(args.attempt)
