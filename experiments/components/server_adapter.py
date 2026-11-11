"""Real seven-mode component Search/Fetch server; no owner keys or rankings.

Returned lookup handles remain cell-local semantic ordinals. The Plain
semantic control uses a newly materialized contiguous FP32 posting index.
"""
from pathlib import Path
import argparse, hashlib, json, os, socket, socketserver, sqlite3, sys, threading, time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'experiment_pprag_precision_network_20261108'
LEGACY=ROOT.parent/'experiment_projectlift_protocol_20261107'/'protocol'
CONFIG=json.loads((LEGACY/'config.json').read_text(encoding='utf-8'))
SRC=Path(CONFIG['original_root'])
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
sys.path[:0]=[str(SRC/'.gpu_deps'),str(SRC/'.deps'),str(LEGACY)]
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from selection import waterfill,local_order
from wire import encode,decode,send,receive
LIMIT=threadpool_limits(limits=2)
MODES=('full','late_route','plain_semantic','plain_lexical','planner_quota_lexical','no_native','fixed_rrf')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def spec(path,known_hash=None):
    path=Path(path).resolve();s=path.stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':known_hash or sha(path)}


def check_stat(binding):
    path=Path(binding['path']);s=path.stat()
    if s.st_size!=binding['bytes'] or s.st_mtime_ns!=binding['mtime_ns']:raise ValueError('Frozen public source changed: '+str(path))
    return path


def top(scores,k,rows=None):
    """Exact original D selection; optional explicit row tie for Plain."""
    if k<=0:return np.empty(0,dtype=np.int64)
    k=min(k,len(scores))
    if rows is None:
        if k==len(scores):return np.argsort(-scores,kind='stable')
        a=np.argpartition(-scores,k-1)[:k];return a[np.argsort(-scores[a],kind='stable')]
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    a=np.flatnonzero(scores>=threshold)
    return a[np.lexsort((rows[a],-scores[a]))[:k]]


def expected_public_inputs(dataset,mode):
    if mode not in MODES:raise ValueError('Unregistered component mode')
    cfg=CONFIG['datasets'][dataset];b=CONFIG['protocol']['datasets'][dataset];s=b['semantic_files'];g=b['geometry_files']
    paths={'route_labels':Path(s['labels']['path']),'semantic_offsets':Path(s['offsets']['path']),
           'lexical_candidate_rows':Path(g['candidate_rows.npy']['path']),'lexical_cells':Path(g['cells.npy']['path']),
           'fts_index':Path(b['server_index']['path'])}
    if mode=='plain_semantic':
        paths['semantic_coordinates']=ROOT/'prepared'/dataset/'indexes'/'plain_semantic_cell_fp32.npy'
    else:
        paths.update(semantic_coordinates=Path(s['coordinates']['path']),semantic_norms=Path(s['norms']['path']))
    if mode=='plain_lexical':paths['lexical_coordinates']=SRC/cfg['lexical_source']/'lexical_sketch_fp16.npy'
    elif mode!='planner_quota_lexical':
        paths.update(lexical_coordinates=Path(g['noisy_local_fp16.npy']['path']),lexical_norms=Path(g['noisy_fp16_norms_fp32.npy']['path']))
    return {key:value.resolve() for key,value in paths.items()}


def source_index_specs(dataset,mode):
    """Preparation-only binding helper; hashes only previously unbound arrays."""
    cfg=CONFIG['datasets'][dataset];b=CONFIG['protocol']['datasets'][dataset]
    known={str(Path(v['path']).resolve()):v for v in [*b['semantic_files'].values(),*b['geometry_files'].values(),b['server_index']]}
    if mode=='plain_semantic':
        built=read(ROOT/'prepared'/dataset/'indexes'/'plain_semantic_manifest.json')
        if built['status']!='complete':raise ValueError('Incomplete native index')
        known[str(Path(built['index']['path']).resolve())]=built['index']
    result={}
    for key,path in expected_public_inputs(dataset,mode).items():
        if str(path) in known:
            value=known[str(path)];check_stat(value);result[key]=dict(value)
        else:result[key]=spec(path)
    return result


class Index:
    def __init__(self,dataset,mode,*,require_prepared=True):
        if mode not in MODES:raise ValueError('Unregistered component mode')
        self.dataset,self.mode=dataset,mode;self.cfg=CONFIG['datasets'][dataset]
        expected=expected_public_inputs(dataset,mode)
        if require_prepared:
            manifest=read(ROOT/'prepared'/dataset/'public'/f'server_manifest_{mode}.json')
            if (manifest['mode'],manifest['dataset'],manifest['role'])!=(mode,dataset,'server-public'):raise ValueError('Wrong public manifest')
            bindings=manifest['index_files']
            if set(bindings)!=set(expected):raise ValueError('Public index allowlist differs')
            for key,value in bindings.items():
                if check_stat(value).resolve()!=expected[key]:raise ValueError('Unexpected public index path')
        else:
            # Owner-only offline coverage may use the immutable bound sources
            # before public record-store manifests have been created.
            b=CONFIG['protocol']['datasets'][dataset]
            for value in [*b['semantic_files'].values(),*b['geometry_files'].values(),b['server_index']]:check_stat(value)
            if mode=='plain_semantic':
                native=read(ROOT/'prepared'/dataset/'indexes'/'plain_semantic_manifest.json')
                if native['status']!='complete':raise ValueError('Native index not complete')
                check_stat(native['index'])
        self.labels=np.load(expected['route_labels']);self.token_to_cell={bytes(v):i for i,v in enumerate(self.labels)}
        self.offsets=np.load(expected['semantic_offsets'],mmap_mode='r')
        self.coordinates=np.load(expected['semantic_coordinates'],mmap_mode='r')
        self.norms=None if mode=='plain_semantic' else np.load(expected['semantic_norms'],mmap_mode='r')
        dimension=384 if mode=='plain_semantic' else 256
        if self.coordinates.shape!=(int(self.offsets[-1]),dimension):raise ValueError('Bad semantic index shape')
        if self.coordinates.dtype!=(np.float32 if mode=='plain_semantic' else np.float16):raise ValueError('Bad semantic index dtype')
        self.candidate_rows=np.load(expected['lexical_candidate_rows'],mmap_mode='r')
        self.assignments=np.load(expected['lexical_cells'],mmap_mode='r')
        self.lexical=self.lexical_norms=None
        if 'lexical_coordinates' in expected:
            self.lexical=np.load(expected['lexical_coordinates'],mmap_mode='r')
            if self.lexical.shape!=(len(self.candidate_rows),1024 if mode=='plain_lexical' else 257):raise ValueError('Bad lexical index shape')
        if 'lexical_norms' in expected:self.lexical_norms=np.load(expected['lexical_norms'],mmap_mode='r')
        if mode=='plain_lexical':
            rawrows=np.load(SRC/self.cfg['lexical_source']/'candidate_rows.npy',mmap_mode='r')
            if not np.array_equal(rawrows,self.candidate_rows):raise ValueError('Original sketch row order differs')
        self.db=sqlite3.connect(f"file:{expected['fts_index'].as_posix()}?mode=ro",uri=True,check_same_thread=False)
        self.db.execute('PRAGMA query_only=ON');self.db.execute('PRAGMA cache_size=-262144')
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='docmap'").fetchone():raise ValueError('Owner map exposed by server FTS')

    def close(self):
        self.db.close()
        for name in ('coordinates','norms','candidate_rows','assignments','lexical','lexical_norms','offsets'):
            value=getattr(self,name,None)
            if isinstance(value,np.memmap):value._mmap.close()
            setattr(self,name,None)

    def planner(self,termbytes,term_slots):
        if len(termbytes)!=term_slots*16:raise ValueError('Bad lexical label width')
        terms=[termbytes[i:i+16].hex() for i in range(0,len(termbytes),16)]
        fetched=self.db.execute('SELECT rowid FROM postings WHERE postings MATCH ? ORDER BY bm25(postings) LIMIT ?',(' OR '.join(terms),1000)).fetchall()
        candidates=np.asarray([int(r[0])-1 for r in fetched],np.int64)
        positions=np.searchsorted(self.candidate_rows,candidates)
        if len(positions) and (np.any(positions>=len(self.candidate_rows)) or np.any(self.candidate_rows[positions]!=candidates)):raise ValueError('Unmaterialized lexical row')
        return candidates,positions,np.asarray(self.assignments[positions],np.int32)

    def select(self,meta,blocks):
        started=time.perf_counter_ns()
        if meta['dataset']!=self.dataset or meta['component_mode']!=self.mode:raise ValueError('Request mode/dataset mismatch')
        if len(blocks)!=4:raise ValueError('Four request blocks required')
        labels,sembytes,termbytes,lexbytes=blocks
        if len(labels)!=128*16:raise ValueError('Exactly128 eligible routing labels required')
        cells=[self.token_to_cell[labels[i:i+16]] for i in range(0,len(labels),16)]
        if len(set(cells))!=128:raise ValueError('Repeated eligible semantic cell')
        plain=self.mode=='plain_semantic';late=self.mode=='late_route'
        q=np.frombuffer(sembytes,dtype='<f4' if plain else '<f2').astype(np.float32)
        shape=(384,) if plain else (2048,256) if late else (128,256)
        q=q.reshape(shape)
        if not np.isfinite(q).all():raise ValueError('Nonfinite semantic query')
        decoded=time.perf_counter_ns();selected_by_cell={};reads=0;depth=self.cfg['semantic_depth']
        eligible=set(cells)
        for ordinal,cell in enumerate(range(2048) if late else cells):
            a,b=int(self.offsets[cell]),int(self.offsets[cell+1])
            d=np.asarray(self.coordinates[a:b],dtype=np.float32)
            if plain:score=d@q
            else:score=2*(d@(q[cell] if late else q[ordinal]))-self.norms[a:b]
            reads+=b-a
            if cell in eligible:
                selected_by_cell[cell]=top(score,min(depth,len(d)),np.arange(len(d),dtype=np.int64) if plain else None)
        selection=[]
        for ordinal,cell in enumerate(cells):
            chosen=selected_by_cell[cell]
            selection.extend((1,cell,int(p),-1,ordinal) for p in chosen)
            selection.extend((1,cell,-1,-1,ordinal) for _ in range(depth-len(chosen)))
        semantic_done=time.perf_counter_ns()
        candidates,positions,assignments=self.planner(termbytes,int(meta['term_slots']))
        planner_done=time.perf_counter_ns();quotas=waterfill(np.bincount(assignments,minlength=16),300)
        if self.mode=='planner_quota_lexical':
            if lexbytes:raise ValueError('Planner-quota mode has no lexical numerical token')
            lq=None
        elif self.mode=='plain_lexical':
            lq=np.frombuffer(lexbytes,dtype='<f4');assert lq.shape==(1024,) and np.isfinite(lq).all()
        else:
            lq=np.frombuffer(lexbytes,dtype='<f2').reshape(16,257).astype(np.float32)
            if not np.isfinite(lq).all():raise ValueError('Nonfinite lexical query')
        lexical_scored=0
        for cell in range(16):
            ordinal=np.flatnonzero(assignments==cell);positions_cell=positions[ordinal]
            if self.mode=='planner_quota_lexical':chosen=np.arange(min(int(quotas[cell]),len(ordinal)),dtype=np.int64)
            else:
                d=np.asarray(self.lexical[positions_cell],np.float32)
                score=d@lq if self.mode=='plain_lexical' else 2*(d@lq[cell])-self.lexical_norms[positions_cell]
                lexical_scored+=len(d);chosen=local_order(score,candidates[ordinal],int(quotas[cell]))
            selection.extend((2,cell,int(candidates[o]),int(o),cell) for o in ordinal[chosen])
        used=int(quotas.sum());selection.extend((2,0,-1,-1,0) for _ in range(300-used))
        finished=time.perf_counter_ns()
        return selection,{'request_decode_vectors_ms':(decoded-started)/1e6,
            'semantic_lookup_refine_ms':(semantic_done-decoded)/1e6,'lexical_fts5_planner_ms':(planner_done-semantic_done)/1e6,
            'lexical_partition_refine_ms':(finished-planner_done)/1e6,'semantic_posting_reads':reads,
            'semantic_cells_scored':2048 if late else 128,'semantic_eligible_cells':128,'semantic_wire_slots':128*depth,
            'semantic_unique_documents_scored':None,'planner_candidates':len(candidates),
            'planner_rows_sha256':hashlib.sha256(candidates.astype('<i8').tobytes()).hexdigest(),
            'lexical_proxy_scored_candidates':lexical_scored,'lexical_selected_valid':used,'lexical_wire_slots':300,
            'lexical_quotas':quotas.tolist()}


class ProtectedStore:
    def __init__(self,path):
        self.db=sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro',uri=True,check_same_thread=False)
        self.db.execute('PRAGMA query_only=ON');self.db.execute('PRAGMA cache_size=-262144')
    def record(self,p,c,r):
        item=self.db.execute('SELECT value FROM records WHERE path=? AND cell=? AND row=?',(p,c,r)).fetchone()
        if item is None:raise KeyError(f'Unprepared protected record {p},{c},{r}')
        return item[0]
    def document(self,locator):
        item=self.db.execute('SELECT value FROM documents WHERE locator=?',(locator,)).fetchone()
        if item is None:raise KeyError('Unprepared protected document')
        return item[0]


def run(args):
    public=ROOT/'prepared'/args.dataset/'public'/f'server_manifest_{args.mode}.json';manifest=read(public)
    if set(manifest)!={'schema','role','dataset','mode','parameters','index_files','record_store'}:raise ValueError('Unexpected public manifest fields')
    if manifest['schema']!='component-server-public-v1':raise ValueError('Unexpected public schema')
    index=Index(args.dataset,args.mode);binding=manifest['record_store']
    if Path(binding['path']).resolve()!=(ROOT/'prepared'/args.dataset/'public'/'records.sqlite3').resolve():raise ValueError('Unexpected record store')
    store=ProtectedStore(check_stat(binding));lock=threading.Lock();logs=[]
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            while True:
                try:body,received=receive(self.request)
                except EOFError:return
                started=time.perf_counter_ns();meta={}
                try:
                    meta,blocks=decode(body);kind=meta['kind']
                    if kind=='shutdown':
                        send(self.request,encode({'kind':'stopped'},[]));threading.Thread(target=server.shutdown,daemon=True).start();return
                    with lock:
                        entered=time.perf_counter_ns()
                        if kind=='search':
                            selected,counters=index.select(meta,blocks);lookup=time.perf_counter_ns()
                            records=[store.record(p,c,r) for p,c,r,_,_ in selected]
                            out={'kind':'search_result','slots':[[p,g,o] for p,_,_,o,g in selected],
                                 'server':counters|{'record_lookup_ms':(time.perf_counter_ns()-lookup)/1e6}}
                        elif kind=='fetch':
                            if len(blocks)!=1 or len(blocks[0])!=8*meta['count'] or not 0<=meta['count']<=20:raise ValueError('Invalid Fetch cap/frame')
                            lookup=time.perf_counter_ns();records=[store.document(blocks[0][i:i+8]) for i in range(0,len(blocks[0]),8)]
                            out={'kind':'fetch_result','count':len(records),'server':{'document_lookup_ms':(time.perf_counter_ns()-lookup)/1e6}}
                        else:raise ValueError('Unsupported RPC')
                        out['server'].update(queue_wait_ms=(entered-started)/1e6,request_application_bytes=received)
                        encoded_started=time.perf_counter_ns();response=encode(out,records);encoded=time.perf_counter_ns()
                    sent=send(self.request,response);ended=time.perf_counter_ns()
                    logs.append({'kind':kind,'request_id':meta.get('request_id'),'received_application_bytes':received,'response_application_bytes':sent,
                        'server_wall_after_receive_ms':(ended-started)/1e6,'serialize_ms':(encoded-encoded_started)/1e6,'send_ms':(ended-encoded)/1e6})
                except Exception as exc:
                    logs.append({'kind':'error','request_id':meta.get('request_id'),'error':f'{type(exc).__name__}: {exc}'})
                    send(self.request,encode({'error':f'{type(exc).__name__}: {exc}'},[]))
    class Server(socketserver.ThreadingTCPServer):allow_reuse_address=True;daemon_threads=True
    with Server(('127.0.0.1',0),Handler) as server:
        write(args.ready,{'host':'127.0.0.1','port':server.server_address[1],'pid':os.getpid(),'dataset':args.dataset,'comparison_mode':args.mode,
            'server_scan_workers':1,'actual_threadpools':threadpool_info(),'server_reads_owner_keys':False,'global_posting_map_loaded':False,
            'semantic_coordinate_dimension':384 if args.mode=='plain_semantic' else 256,'semantic_coordinate_dtype':'float32' if args.mode=='plain_semantic' else 'float16',
            'record_store_path':binding['path'],'record_store_sha256':binding['sha256'],'public_manifest_path':str(public),'public_manifest_fields':sorted(manifest)})
        server.serve_forever(poll_interval=.1)
    index.close();store.db.close();write(args.log,{'requests':logs,'source':'Actual Search/Fetch TCP frames, real selection and protected store reads.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=tuple(CONFIG['datasets']));parser.add_argument('--mode',required=True,choices=MODES)
    parser.add_argument('--ready',type=Path,required=True);parser.add_argument('--log',type=Path,required=True);run(parser.parse_args())
