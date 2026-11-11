"""Bind completed quality outputs without rereading large native vector banks."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(p):
    p = Path(p).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def main():
    dest = HERE/'execution_closure.json'
    if dest.exists():
        raise FileExistsError(dest)
    release = read(HERE/'release.json')
    assert release['script_sha256'] == sha(HERE/'run_quality.py')
    assert release['plan_sha256'] == sha(HERE/'QUALITY_PLAN.json')
    source_checks = {}; outputs = {}; corpora = {}; backends = {}; query_total = 0
    for ds, label in (('nq', 'nq'), ('hotpotqa', 'hotpotqa'), ('msmarco', 'msmarco_dev')):
        folder = HERE/'results'/label
        report = read(folder/'verification.json')
        input_path = folder/'input_binding.json'; info = read(input_path)
        assert report['input_binding_sha256'] == sha(input_path)
        assert report['script_sha256'] == release['script_sha256']
        assert report['plan_sha256'] == release['plan_sha256']
        assert report['status'] in ('computed_pending_independent_audit', 'passed')
        assert report['outer_queries'] == report['queries'] == info['query_count']
        query_total += report['queries']
        for name, spec in info['sources'].items():
            path = Path(name); st = path.stat()
            assert st.st_size == spec['bytes'] and st.st_mtime_ns == spec['mtime_ns'], name
            if name not in source_checks:
                source_checks[name] = sha(path)
            assert source_checks[name] == spec['sha256'], name
        for name, digest in info['historical_source_sha256'].items():
            if name not in source_checks:
                source_checks[name] = sha(name)
            assert source_checks[name] == digest
        bank = info['corpus_header_stat']; st = Path(bank['path']).stat()
        assert st.st_size == bank['bytes'] and st.st_mtime_ns == bank['mtime_ns']
        corpora[ds] = {'header_stat_unchanged_since_preflight': bank,
            'inherited_prior_full_content_validation': info['corpus_full_validation'],
            'new_full_corpus_hash_pass_performed': False}
        items = [bind(folder/'verification.json'), bind(input_path)]
        for name, spec in report['outputs'].items():
            assert sha(folder/name) == spec['sha256']
            items.append(bind(folder/name))
        for block in sorted((folder/'native_blocks').glob('*.json')):
            meta = read(block); artifact = block.with_suffix('.npz')
            assert meta['status'] == 'passed' and meta['input_binding_sha256'] == sha(input_path)
            assert meta['sha256'] == sha(artifact)
            items.extend((bind(block), bind(artifact)))
        backend_path = folder/'execution_backend.json'
        if backend_path.exists():
            backend = read(backend_path)
            assert backend['status'] == 'passed' and backend['dataset'] == ds
            assert backend['verification']['sha256'] == sha(folder/'verification.json')
            for field in ('original_runner', 'io_wrapper', 'amendment', 'fixture'):
                spec = backend[field]
                assert sha(spec['path']) == spec['sha256']
            backends[ds] = {'evidence': bind(backend_path), 'details': backend}
            items.append(bind(backend_path))
        else:
            backends[ds] = {'implementation': 'Original frozen 32-query runner, no IO wrapper.'}
        ssd_path = folder/'execution_ssd_backend.json'
        if ssd_path.exists():
            ssd = read(ssd_path)
            assert ssd['status'] == 'passed' and ssd['verification']['sha256'] == sha(folder/'verification.json')
            execution = ssd['SSD_execution']; assert sha(execution['path']) == execution['sha256']
            runtime = read(execution['path'])
            assert runtime['status'] == 'complete' and runtime['numpy_loader_restored'] and runtime['redirect_count'] > 0
            assert runtime['memory_guard_changed'] is False and runtime['scientific_function_or_model_or_input_changed'] is False
            assert ssd['source_native']['sha256'] == ssd['actual_native_read_file']['sha256'] == info['corpus_full_validation']['corpus_embeddings_sha256']
            for spec in runtime['bindings'].values():
                if isinstance(spec, dict) and 'path' in spec and 'sha256' in spec:
                    # The 13.6GB native files were fully hashed by the independent bridge.
                    # Here only current stats are rechecked for those two files.
                    if spec['path'] in (ssd['source_native']['path'], ssd['actual_native_read_file']['path']):
                        st = Path(spec['path']).stat()
                        assert st.st_size == spec['bytes'] and st.st_mtime_ns == spec['mtime_ns']
                    else:
                        assert sha(spec['path']) == spec['sha256']
            backends[ds]['SSD_additional_execution_layer'] = bind(ssd_path)
            items.extend((bind(ssd_path), bind(execution['path'])))
        outputs[ds] = items
    assert query_total == 9005
    paired = read(HERE/'paired_results.json')
    assert paired['outer_queries'] == 9005 and paired['runner']['sha256'] == release['script_sha256']
    diagnostic_path = HERE/'path_diagnostics/summary.json'
    diagnostic = read(diagnostic_path)
    assert diagnostic['primary_seven_rows_unchanged']
    assert diagnostic['new_retrieval_or_native_scoring_or_fit_performed'] is False
    for key in ('plan', 'script', 'frozen_quality_runner'):
        spec = diagnostic[key]; assert sha(spec['path']) == spec['sha256']
    for values in diagnostic['datasets'].values():
        for key in ('metrics', 'stage_rankings', 'primary_verification'):
            spec = values[key]; assert sha(spec['path']) == spec['sha256']
    matched_path = HERE/'matched_pool/paired_results.json'; matched = read(matched_path)
    assert matched['outer_queries'] == 9005 and matched['all_original_inputs_and_reports_unchanged']
    assert matched['bindings']['original_paired']['sha256'] == sha(HERE/'paired_results.json')
    for values in matched['datasets'].values():
        spec = values['verification']; assert sha(spec['path']) == spec['sha256']
        verified = read(spec['path'])
        assert verified['original_RRF100_rankings_and_metrics_exactly_replayed']
        assert verified['six_other_rows_rankings_metrics_and_paired_stats_byte_value_identical']
        for spec in verified['outputs'].values():
            assert sha(spec['path']) == spec['sha256']
    result = {'schema': 'matched-components-quality-execution-closure-v1', 'status': 'passed',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'outer_queries': query_total,
        'release': bind(HERE/'release.json'), 'plan': bind(HERE/'QUALITY_PLAN.json'),
        'runner': bind(HERE/'run_quality.py'), 'closure_script': bind(__file__),
        'all_bound_small_sources_hash_and_stat_unchanged': source_checks,
        'native_corpus_bindings': corpora, 'outputs': outputs,
        'paired_results': bind(HERE/'paired_results.json'),
        'path_only_diagnostics': bind(diagnostic_path),
        'matched_fusion_pool_primary_report': bind(matched_path),
        'execution_backends': backends,
        'io_optimization': 'Global-union resident cache was rejected because all three banks exceed 2GiB. Adopted 128-query grouping and the later byte-identical MS SSD read bridge are separate audited execution layers, bound by per-dataset backend sidecars and exact fixtures; original scientific runner and memory guard remain unchanged.',
        'scientific_status': 'Numerical outputs retain computed_pending_independent_audit; this closure verifies provenance/completion, not independent scientific acceptance.',
        'no_protocol_timing_or_privacy_claim': True}
    with dest.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False); f.write('\n')
    print(json.dumps({'status': 'passed', 'outer_queries': query_total, 'closure_sha256': sha(dest)}))


if __name__ == '__main__':
    main()
