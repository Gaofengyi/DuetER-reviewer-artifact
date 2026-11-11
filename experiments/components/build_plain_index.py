"""Build an immutable, cell-contiguous native-FP32 semantic control index.

Old scientific arrays are read-only. Only this experiment's incomplete output
may be resumed; completed outputs are never overwritten. No online timing.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'experiment_pprag_precision_network_20261108'
LEGACY = ROOT.parent / 'experiment_projectlift_protocol_20261107' / 'protocol'
CONFIG = json.loads((LEGACY/'config.json').read_text(encoding='utf-8'))
SRC = Path(CONFIG['original_root'])
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
sys.path[:0] = [str(SRC/'.gpu_deps'), str(SRC/'.deps')]
import numpy as np
from threadpoolctl import threadpool_limits
LIMIT = threadpool_limits(limits=2)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def stat(path):
    path=Path(path).resolve();s=path.stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def sorted_gather(native, rows):
    order=np.argsort(rows,kind='stable')
    sorted_values=np.asarray(native[rows[order]],dtype='<f4')
    result=np.empty_like(sorted_values)
    result[order]=sorted_values
    return result


def build(dataset, block_rows=65536, resume=False):
    if block_rows<1 or block_rows>262144:raise ValueError('Bounded block size required')
    cfg=CONFIG['datasets'][dataset];src=SRC/cfg['semantic_cache']
    destination=ROOT/'prepared'/dataset/'indexes';destination.mkdir(parents=True,exist_ok=True)
    target=destination/'plain_semantic_cell_fp32.npy'
    partial=destination/'plain_semantic_cell_fp32.partial.npy'
    checkpoint=destination/'plain_semantic_build_checkpoint.json'
    manifest=destination/'plain_semantic_manifest.json'
    if target.exists() or manifest.exists():raise FileExistsError('Completed native index is immutable')
    native_path=src/'corpus_embeddings.npy';posting_path=src/'semantic_postings.npy';offset_path=src/'semantic_offsets.npy'
    inputs={name:stat(p) for name,p in [('native',native_path),('postings',posting_path),('offsets',offset_path)]}
    sources={str(p):sha(p) for p in (Path(__file__),LEGACY/'config.json')}
    native=np.load(native_path,mmap_mode='r');postings=np.load(posting_path,mmap_mode='r');offsets=np.load(offset_path)
    assert native.shape==(cfg['documents'],384) and native.dtype==np.float32
    assert postings.ndim==1 and np.issubdtype(postings.dtype,np.integer)
    assert offsets.shape==(2049,) and int(offsets[0])==0 and int(offsets[-1])==len(postings)
    assert np.all(np.diff(offsets)>=0) and len(postings)==2*cfg['documents']
    started=time.perf_counter();first=0
    if partial.exists() or checkpoint.exists():
        if not resume or not(partial.exists() and checkpoint.exists()):raise FileExistsError('Incomplete index requires explicit matching resume')
        state=read(checkpoint)
        assert state['inputs']==inputs and state['source_sha256']==sources and state['dataset']==dataset
        first=int(state['rows_written']);header=int(state['header_bytes'])
        assert 0<=first<=len(postings) and partial.stat().st_size>=header+first*384*4
        with partial.open('r+b') as stream:stream.truncate(header+first*384*4)
    else:
        with partial.open('xb') as stream:
            np.lib.format.write_array_header_2_0(stream,{'descr':np.dtype('<f4').str,'fortran_order':False,'shape':(len(postings),384)})
            header=stream.tell()
        state={'schema':'component-native-index-build-v1','dataset':dataset,'status':'building','inputs':inputs,
               'source_sha256':sources,'header_bytes':header,'rows_written':0,'block_rows':block_rows}
        write(checkpoint,state)
    print(json.dumps({'dataset':dataset,'stage':'native_contiguous_index','rows':len(postings),'resume_row':first,'data_bytes':len(postings)*384*4}),flush=True)
    with partial.open('ab') as output:
        for start in range(first,len(postings),block_rows):
            stop=min(start+block_rows,len(postings));rows=np.asarray(postings[start:stop],dtype=np.int64)
            assert len(rows) and int(rows.min())>=0 and int(rows.max())<len(native)
            values=sorted_gather(native,rows)
            if not np.isfinite(values).all():raise ValueError('Nonfinite source native coordinate')
            output.write(values.tobytes(order='C'));output.flush()
            state.update(rows_written=stop,elapsed_this_invocation_seconds=time.perf_counter()-started)
            write(checkpoint,state)
            if (start//block_rows)%8==0 or stop==len(postings):
                print(json.dumps({'dataset':dataset,'stage':'native_contiguous_index','rows_written':stop,'total_rows':len(postings),'elapsed_seconds':time.perf_counter()-started}),flush=True)
    assert partial.stat().st_size==header+len(postings)*384*4
    stored=np.load(partial,mmap_mode='r');assert stored.shape==(len(postings),384) and stored.dtype==np.float32
    positions=np.unique(np.concatenate((np.linspace(0,len(postings)-1,128,dtype=np.int64),offsets[:-1][np.diff(offsets)>0].astype(np.int64))))
    for start in range(0,len(positions),128):
        chosen=positions[start:start+128]
        assert np.array_equal(np.asarray(stored[chosen]),np.asarray(native[np.asarray(postings[chosen],np.int64)]))
    del stored,native,postings
    assert {name:stat(value['path']) for name,value in inputs.items()}==inputs
    print(json.dumps({'dataset':dataset,'stage':'full_index_and_input_hash_binding'}),flush=True)
    input_hashes={name:sha(spec['path']) for name,spec in inputs.items()}
    output_hash=sha(partial);partial.rename(target)
    result={'schema':'component-plain-semantic-index-v1','status':'complete','dataset':dataset,
        'index':stat(target)|{'sha256':output_hash},'inputs':{k:v|{'sha256':input_hashes[k]} for k,v in inputs.items()},
        'source_sha256':sources,'shape':[int(offsets[-1]),384],'dtype':'float32',
        'record_order':'Exactly original semantic_postings order; cell slices use unchanged semantic_offsets.',
        'construction':'Every posting row is copied from its original native FP32 vector, preserving all 384 coordinates and two replicas.',
        'sampled_persisted_rows_bitwise_equal':len(positions),'whole_output_sha256_read_from_persisted_file':True,
        'full_corpus_native_source_sha256_read':True,'formal_timing':False,'elapsed_seconds':time.perf_counter()-started}
    write(manifest,result);state.update(status='complete',output_manifest=str(manifest));write(checkpoint,state)
    print(json.dumps({'dataset':dataset,'status':'complete','manifest':str(manifest),'bytes':target.stat().st_size}),flush=True)
    return result


def selftest():
    bank=np.arange(120,dtype=np.float32).reshape(10,12)
    rows=np.asarray([9,0,8,2,2,7,1],np.int64)
    assert np.array_equal(sorted_gather(bank,rows),bank[rows])
    print(json.dumps({'status':'passed','scope':'Sorted-gather reconstructs original posting order including duplicates; no corpus read.'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=tuple(CONFIG['datasets']));p.add_argument('--block-rows',type=int,default=65536);p.add_argument('--resume',action='store_true');p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest()
    elif a.datasets:
        for dataset in a.datasets:build(dataset,a.block_rows,a.resume)
    else:p.error('Supply --datasets or --selftest')
