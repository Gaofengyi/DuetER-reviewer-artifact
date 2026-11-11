"""Stream cached admission arrays only; never open original native vectors."""
import argparse, hashlib, json, os, sys, time, zipfile
from pathlib import Path
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '2'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path.insert(0, str(BASE/'duetdpe_experiments/.deps'))
import numpy as np


def rows(path, expected_shape):
    with zipfile.ZipFile(path) as z, z.open('emitted.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        assert tuple(shape) == tuple(expected_shape) and not fortran and dtype == np.int32
        count = shape[-2]*shape[-1]; size = count*4
        for _ in range(int(np.prod(shape[:-2]))):
            raw = stream.read(size); assert len(raw) == size
            yield np.frombuffer(raw, dtype=dtype).reshape(shape[-2:])
        assert not stream.read(1)


def unique(a):
    a = a.T.reshape(-1); a = a[a >= 0]
    _, first = np.unique(a, return_index=True)
    return a[np.sort(first)]


def run(dataset):
    start = time.perf_counter()
    config = json.loads((BASE/'experiment_revision_20261107/config.json').read_text())['datasets'][dataset]
    old = BASE/'experiment_revision_20261107/results'/config['label']
    new = BASE/'experiment_plaintext_float_matched_20261109/semantic/results'/config['label']
    with np.load(old/'semantic.rankings.npz') as z:
        outer = z['outer_holdout']; qids = z['query_ids']
    with np.load(new/'emitted.npz') as z:
        assert np.array_equal(z['query_ids'], qids) and np.array_equal(z['outer_holdout'], outer)
    with np.load(old/'semantic_emitted.npz') as z:
        assert np.array_equal(z['query_ids'], qids)
    n = len(qids); depth = config['semantic_depth']
    plain_count = np.zeros(n, dtype=np.int32); max_budget = np.zeros(n, dtype=np.int32)
    membership = np.zeros(config['documents'], dtype=bool)
    for qi, a in enumerate(rows(new/'emitted.npz', (n, 128, depth))):
        if outer[qi]:
            plain_count[qi] = len(unique(a))
    for ordinal, a in enumerate(rows(old/'semantic_emitted.npz', (3, n, 128, depth))):
        qi = ordinal % n
        if outer[qi]:
            full = unique(a); b = min(len(full), int(plain_count[qi]))
            membership[full[:b]] = True; max_budget[qi] = max(max_budget[qi], b)
    for qi, a in enumerate(rows(new/'emitted.npz', (n, 128, depth))):
        if outer[qi]:
            membership[unique(a)[:max_budget[qi]]] = True
    selected = np.flatnonzero(membership).astype(np.int32)
    result = {'dataset': dataset, 'outer_queries': int(outer.sum()), 'exact_global_unique_admitted_rows': len(selected),
              'FP32_native_384_bank_bytes': len(selected)*384*4,
              'within_2GiB': len(selected)*384*4 <= 2*1024**3,
              'row_array_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
              'scope': 'Exact union of both current rank-major B_star prefixes over three repeats; no native bank opened.',
              'offline_seconds': time.perf_counter()-start}
    dest = HERE/'io_feasibility'; dest.mkdir(exist_ok=True)
    with (dest/(dataset+'.json')).open('x', encoding='utf-8') as h:
        json.dump(result, h, indent=2); h.write('\n')
    np.save(dest/(dataset+'_rows.npy'), selected, allow_pickle=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--datasets', nargs='+', default=['nq', 'hotpotqa', 'msmarco'])
    for ds in p.parse_args().datasets:
        run(ds)
