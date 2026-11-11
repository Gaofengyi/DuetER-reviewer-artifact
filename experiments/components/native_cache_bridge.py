"""Audited, offline-only native read-path bridge for the frozen index builder.

The already existing C-drive cache is opened read-only after current full-file
hash equivalence. No scientific coordinate, posting order, output path, native
dtype, or frozen builder source is changed. This module is never imported by
the online protocol server or client.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse, datetime, hashlib, importlib.util, json, os, sys, time

ROOT=Path(__file__).resolve().parent
BUILDER=ROOT/'build_plain_index.py'
BUILDER_SHA256='a8a74afdca846c4da8e520fd6f28061cf8e9d474f410d185c789496a66916868'
SOURCE=ROOT.parent/'duetdpe_experiments'/'cache/full_semantic/msmarco_8841823/corpus_embeddings.npy'
CACHE=Path('<HOME>/.codex/tmp/matched_pprag_20261107/msmarco_native.npy')
HISTORICAL=ROOT.parent/'experiment_pprag_matched_20261107'/'local_native_cache.json'
BINDING=ROOT/'audits/native_cache_binding.json'
REVIEW=ROOT/'audits/native_cache_bridge_review.json'
HISTORY=ROOT/'audits/ms_index_build_operational_history.json'
EXPECTED_BYTES=13581040256
sys.dont_write_bytecode=True


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()


def stat(path):
    path=Path(path).resolve();s=path.stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def specification(path):return stat(path)|{'sha256':sha(path)}


def enforce_review():
    audit=read(REVIEW)
    if audit.get('status')!='passed' or audit.get('open_findings'):raise ValueError('Independent bridge review has not passed')
    item=audit.get('wrapper',{})
    if Path(item.get('path','')).resolve()!=Path(__file__).resolve() or item.get('sha256')!=sha(__file__):raise ValueError('Independent review does not bind this exact wrapper')
    if sha(BUILDER)!=BUILDER_SHA256:raise ValueError('Frozen index builder changed')


def verify():
    enforce_review()
    if BINDING.exists():raise FileExistsError('Existing cache verification is immutable')
    previous=read(HISTORICAL)
    if (previous.get('passed') is not True or Path(previous['source']).resolve()!=SOURCE.resolve()
            or Path(previous['destination']).resolve()!=CACHE.resolve()
            or previous['bytes']!=EXPECTED_BYTES or previous['source_sha256']!=previous['destination_sha256']):
        raise ValueError('Historical cache evidence differs from the intended identical native source')
    before_source,before_cache=stat(SOURCE),stat(CACHE)
    if before_source['bytes']!=EXPECTED_BYTES or before_cache['bytes']!=EXPECTED_BYTES:raise ValueError('Native size differs')
    started=time.perf_counter();print(json.dumps({'stage':'verify_native_source_full_sha','bytes':EXPECTED_BYTES}),flush=True)
    source_hash=sha(SOURCE)
    print(json.dumps({'stage':'verify_existing_C_cache_full_sha','bytes':EXPECTED_BYTES}),flush=True)
    cache_hash=sha(CACHE)
    if source_hash!=cache_hash or source_hash!=previous['source_sha256']:raise ValueError('Current E/C native bytes differ from each other or historical source')
    if stat(SOURCE)!=before_source or stat(CACHE)!=before_cache:raise ValueError('Native/cache file changed during verification')
    result={'schema':'component-native-read-cache-v1','status':'passed','readonly':True,
        'source':before_source|{'sha256':source_hash},'cache':before_cache|{'sha256':cache_hash},
        'byte_identical':True,'original_source_unchanged':True,'no_file_copy_needed':True,
        'historical_binding':specification(HISTORICAL),'verifier_source':specification(__file__),
        'independent_review':specification(REVIEW),'builder_source':specification(BUILDER),
        'verification':'Current independent complete SHA256 reads of E source and existing C cache, with before/after stat equality.',
        'purpose':'Offline-only identical byte source for contiguous index construction and separately reviewed quality replay; never an online index-path change.',
        'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started}
    write(BINDING,result);print(json.dumps({'status':'passed','binding':str(BINDING),'sha256':source_hash,'elapsed_seconds':result['elapsed_seconds']}),flush=True)


def checked_binding():
    binding=read(BINDING)
    if (binding.get('schema')!='component-native-read-cache-v1' or binding.get('status')!='passed'
            or binding.get('readonly') is not True or binding.get('byte_identical') is not True):raise ValueError('Invalid native cache binding')
    if (Path(binding['source']['path']).resolve()!=SOURCE.resolve() or Path(binding['cache']['path']).resolve()!=CACHE.resolve()
            or binding['source']['sha256']!=binding['cache']['sha256']
            or binding['source']['bytes']!=EXPECTED_BYTES or binding['cache']['bytes']!=EXPECTED_BYTES):raise ValueError('Wrong exact source/cache/size/hash')
    for name in ('source','cache'):
        item=binding[name]
        if stat(item['path'])!={k:item[k] for k in ('path','bytes','mtime_ns')}:raise ValueError('Verified native/cache stat changed')
    if binding['verifier_source']['sha256']!=sha(__file__) or binding['builder_source']['sha256']!=sha(BUILDER):raise ValueError('Cache verification source differs')
    return binding


@contextmanager
def redirected_load(numpy_module,source,cache,observations,guard=None):
    """Redirect this exact one path, preserve every other argument and call."""
    original=numpy_module.load;resolved_source=Path(source).resolve();resolved_cache=Path(cache).resolve()
    def load(file,*args,**kwargs):
        path=Path(file).resolve() if isinstance(file,(str,os.PathLike)) else None
        if path==resolved_source:
            mmap=kwargs.get('mmap_mode',args[0] if args else None)
            if mmap!='r':raise ValueError('Cache redirect requires immutable read-only mmap')
            if guard is not None:guard()
            observations.append({'requested_path':str(resolved_source),'actual_read_path':str(resolved_cache),
                'mmap_mode':mmap,'other_arguments_preserved':True})
            return original(resolved_cache,*args,**kwargs)
        return original(file,*args,**kwargs)
    numpy_module.load=load
    try:yield
    finally:numpy_module.load=original


def build(resume,block_rows):
    enforce_review();binding=checked_binding();before_binding_sha=sha(BINDING)
    if block_rows!=262144 or not resume:raise ValueError('This approved transition requires existing MS checkpoint and262144 rows')
    initial=read(ROOT/'prepared/msmarco/indexes/plain_semantic_build_checkpoint.json')
    if initial['dataset']!='msmarco' or initial['status']!='building':raise ValueError('MS checkpoint not resumable')
    current=specification(__file__);history=read(HISTORY)
    if history['segments'][-1]['end_row_exclusive']!=initial['rows_written']:raise ValueError('Operational history/checkpoint prefix mismatch')
    runtime_path=ROOT/'audits/ms_cached_index_runtime_binding.json'
    if runtime_path.exists():raise FileExistsError('Existing cached index execution is immutable')
    observations=[];runtime={'schema':'component-cached-index-execution-v1','status':'running','dataset':'msmarco',
        'process_id':os.getpid(),'actual_argv':sys.argv,'wrapper':current,'frozen_builder':specification(BUILDER),
        'native_cache_binding':specification(BINDING),'independent_review':specification(REVIEW),
        'source_native':binding['source'],'actual_native_read_file':binding['cache'],
        'output_path':str(ROOT/'prepared/msmarco/indexes/plain_semantic_cell_fp32.npy'),
        'start_row_inclusive':initial['rows_written'],'actual_block_rows':block_rows,'redirected_numpy_loads':observations,
        'formal_timing':False,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(runtime_path,runtime)
    module_spec=importlib.util.spec_from_file_location('_component_frozen_index_builder',BUILDER)
    module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
    original_load=module.np.load
    try:
        with redirected_load(module.np,SOURCE,CACHE,observations,guard=checked_binding):result=module.build('msmarco',block_rows=block_rows,resume=True)
        if module.np.load is not original_load or len(observations)!=1:raise AssertionError('Unexpected native load count or un-restored loader')
        checked_binding()
        if sha(BINDING)!=before_binding_sha or specification(__file__)!=current:raise ValueError('Cache binding/wrapper changed during execution')
        runtime.update(status='complete',end_row_exclusive=result['shape'][0],output_index=result['index'],numpy_loader_restored=True,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(runtime_path,runtime)
        history['segments'].append({'start_row_inclusive':initial['rows_written'],'end_row_exclusive':result['shape'][0],
            'actual_block_rows':block_rows,'cli':'native_cache_bridge.py build --resume --block-rows 262144',
            'log':'logs/plain_index_msmarco_cached_v1.log','status':'complete','actual_native_read_path':str(CACHE),
            'runtime_binding':specification(runtime_path)})
        history.update(status='complete',native_cache_binding=specification(BINDING),
            native_cache_transition={'status':'completed','checkpoint_row':initial['rows_written'],'source_unchanged':True})
        write(HISTORY,history)
        manifest_path=ROOT/'prepared/msmarco/indexes/plain_semantic_manifest.json'
        raw_manifest=manifest_path.with_name('plain_semantic_manifest.builder_raw.json')
        with raw_manifest.open('xb') as handle:handle.write(manifest_path.read_bytes())
        result=read(manifest_path);result['offline_read_execution_binding']={
            'scope':'Additional audited offline I/O provenance; original index/input/source bindings below remain unchanged.',
            'original_builder_manifest':specification(raw_manifest),'operational_history':specification(HISTORY),
            'runtime_binding':specification(runtime_path),'native_cache_binding':specification(BINDING)}
        write(manifest_path,result)
        print(json.dumps({'status':'complete','offline_read_binding_added':True,'manifest':str(manifest_path)}),flush=True)
    except BaseException as error:
        runtime.update(status='failed',error_type=type(error).__name__,numpy_loader_restored=module.np.load is original_load)
        write(runtime_path,runtime);raise


def selftest():
    class Fake:
        def __init__(self):self.calls=[]
        def load(self,file,*args,**kwargs):self.calls.append((file,args,kwargs));return 'loaded'
    fake=Fake();original=fake.load;events=[];other=ROOT/'audits/untouched.npy'
    with redirected_load(fake,SOURCE,CACHE,events):
        assert fake.load(SOURCE,mmap_mode='r',allow_pickle=False)=='loaded'
        assert fake.load(other,'r',allow_pickle=False)=='loaded'
        sentinel=object();assert fake.load(sentinel)=='loaded'
        try:fake.load(SOURCE,mmap_mode='r+')
        except ValueError:pass
        else:raise AssertionError('Writable cache mapping was allowed')
    assert fake.load==original and len(events)==1
    assert fake.calls[0]==(CACHE.resolve(),(),{'mmap_mode':'r','allow_pickle':False})
    assert fake.calls[1]==(other,('r',),{'allow_pickle':False}) and fake.calls[2][0] is sentinel
    try:
        with redirected_load(fake,SOURCE,CACHE,[]):raise RuntimeError('fixture')
    except RuntimeError:pass
    assert fake.load==original
    report={'status':'passed','wrapper':specification(__file__),'scope':'Fake-loader fixture; no NumPy import, corpus reads or cache modifications.',
        'checks':['exact target only','all args/kwargs preserved','unrelated/file-object calls untouched','read-only mmap enforced','loader restored on normal and exceptional exits']}
    write(ROOT/'audits/native_cache_bridge_fixture.json',report);print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('verify','build','selftest'));parser.add_argument('--resume',action='store_true');parser.add_argument('--block-rows',type=int,default=262144);args=parser.parse_args()
    if args.stage=='selftest':selftest()
    elif args.stage=='verify':verify()
    else:build(args.resume,args.block_rows)
