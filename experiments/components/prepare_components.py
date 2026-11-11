"""Owner-only preparation for the seven matched-budget component protocols.

Fresh keys and a new sample protected store. No old source is modified. Record
coverage is prepared offline, never synthesized during a measured request.
"""
from pathlib import Path
import argparse, copy, gc, hashlib, importlib.util, json, os, sqlite3, sys, time
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'experiment_pprag_precision_network_20261108'
sys.dont_write_bytecode=True
# The actual fixed client runtime must be established before NumPy/Torch.
client_spec=importlib.util.spec_from_file_location('_components_preparation_client',ROOT/'components_client.py')
client=importlib.util.module_from_spec(client_spec);sys.modules[client_spec.name]=client;client_spec.loader.exec_module(client)
server_spec=importlib.util.spec_from_file_location('_components_preparation_server',ROOT/'server_adapter.py')
server=importlib.util.module_from_spec(server_spec);sys.modules[server_spec.name]=server;server_spec.loader.exec_module(server)
np=client.np;CONFIG=server.CONFIG;SRC=server.SRC;LEGACY=server.LEGACY
read,write,sha,spec=server.read,server.write,server.sha,server.spec
records=client.parent.Owner.__module__
records=sys.modules[records]


def source_hashes():
    return {str(p.resolve()):sha(p) for p in (Path(__file__),ROOT/'server_adapter.py',ROOT/'components_client.py',
        ROOT/'build_plain_index.py',BASE/'client_runtime_binding.py',BASE/'runtime_binding.json',LEGACY/'config.json',LEGACY/'records.py')}


def fresh_private(dataset):
    old_path=BASE/'.owner'/dataset/'client_private.json';original=read(old_path)
    private=copy.deepcopy(original)
    base=client.parent.Owner.fresh().exported()
    precision=dict(base)
    for domain in ('payload','join'):
        value=os.urandom(32).hex()
        while value in base.values() or value in precision.values():value=os.urandom(32).hex()
        precision[domain]=value
    private.update(keys=base,dueter_precision_keys=precision,document_ids={},component_budgets={},
        component_origin={'owner_private_path':str(old_path),'owner_private_sha256':sha(old_path),
                          'scope':'Same immutable48-query cohort; new component budget and new secret record keys.'})
    return original,private


def global_groups(selected,offsets,postings):
    groups=[[] for _ in range(128)]
    for path,cell,row,_,ordinal in selected:
        if path==1 and row>=0:groups[ordinal].append(int(postings[int(offsets[cell])+row]))
    return groups


def coverage(dataset):
    own=ROOT/'.owner'/dataset;dest=ROOT/'prepared'/dataset
    if (own/'client_private.json').exists() or (own/'coverage_report.json').exists():raise FileExistsError('Owner coverage already exists')
    own.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    initial=source_hashes();started=time.perf_counter();old,private=fresh_private(dataset)
    queries=private['queries'];assert len(queries)==16 and len({str(q['id']) for q in queries})==16
    owners={mode:client.QueryOwnerAdapter(CONFIG,dataset,private,mode) for mode in server.MODES}
    first=owners['full'];first.load_encoder()
    for owner in owners.values():
        owner.encoder=first.encoder;owner.encoder_parameter_dtype=first.encoder_parameter_dtype
    runtime=client.verify_loaded_runtime(first.encoder)
    oldowner=client.base.QueryOwnerAdapter(CONFIG,dataset,old,'dueter_projectlift')
    cfg=CONFIG['datasets'][dataset];source=SRC/cfg['semantic_cache']
    offsets=np.load(source/'semantic_offsets.npy',mmap_mode='r');postings=np.load(source/'semantic_postings.npy',mmap_mode='r')
    with np.load(SRC/cfg['planner_cache'],allow_pickle=False) as z:frozen=np.asarray(z[f"budget_{cfg['posting_budget']}"][:,:1000],np.int64)
    # These four selectors span every candidate set. The other three modes
    # have identical candidates by definition and are independently exercised
    # through the real server in the later functional gate.
    modes=('full','plain_semantic','plain_lexical','planner_quota_lexical')
    fresh=[];requests={};checks={};arrays={};keys=set();budgets={};selection_counts={}
    for qi,query in enumerate(queries):
        embedding=first.encode_text(query['text'])
        if not np.array_equal(embedding,np.asarray(query['fresh_embedding'],np.float32)):raise ValueError('Fixed-runtime fresh embedding differs from preceding protocol')
        fresh.append(embedding);qid=str(query['id'])
        oldmeta,oldblocks,_=oldowner.request(query['text'],qid,embedding,f'old-coverage-{qi}')
        mode_requests={}
        for mode,owner in owners.items():
            meta,blocks,trace=owner.request(query['text'],qid,embedding,f'component-coverage-{qi}-{mode}')
            mode_requests[mode]=(meta,blocks)
            if mode=='full' and blocks!=oldblocks:raise ValueError('Full request blocks differ from old frozen protocol')
            if blocks[0]!=oldblocks[0] or blocks[2]!=oldblocks[2]:raise ValueError('Routing or term labels changed across components')
            if mode=='late_route':
                natural=np.frombuffer(blocks[1],dtype='<f2').reshape(2048,256)
                label_map={bytes(label):i for i,label in enumerate(first.labels)}
                route=[label_map[oldblocks[0][i:i+16]] for i in range(0,len(oldblocks[0]),16)]
                if natural[route].tobytes()!=oldblocks[1]:raise ValueError('Late eligible views differ from Full')
            requests[f'{mode}_{qi}']={'block_sha256':[hashlib.sha256(b).hexdigest() for b in blocks],
                'block_bytes':list(map(len,blocks))}
        checks[str(qi)]={'fresh_embedding_bitwise':True,'full_request_blocks_unchanged':True,'route_and_term_labels_shared':True,'late_eligible_views_bitwise':True}
        arrays[f'fresh_embedding_{qi}']=embedding
        # Store request bytes only in the owner-only preparation memory until
        # this phase ends; public deployment never receives these snapshots.
        for mode in modes:
            for number,block in enumerate(mode_requests[mode][1]):arrays[f'block_{mode}_{qi}_{number}']=np.frombuffer(block,dtype=np.uint8).copy()
    for mode in modes:
        index=server.Index(dataset,mode,require_prepared=False)
        try:
            for qi,query in enumerate(queries):
                meta={'dataset':dataset,'component_mode':mode,'term_slots':private['term_slots']}
                blocks=[arrays[f'block_{mode}_{qi}_{number}'].tobytes() for number in range(4)]
                selected,counters=index.select(meta,blocks)
                expected=np.asarray(frozen[query['index']],np.int64);expected=expected[expected>=0]
                if counters['planner_rows_sha256']!=hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest():raise ValueError('Planner differs from frozen workload')
                positions=np.searchsorted(index.candidate_rows,expected)
                for row,cell in zip(expected,index.assignments[positions]):keys.add((2,int(cell),int(row)))
                keys.update((p,c,r) for p,c,r,_,_ in selected if r>=0)
                groups=global_groups(selected,offsets,postings);admission=client.rank_major_unique(groups)
                arrays[f'selection_{mode}_{qi}']=np.asarray(selected,np.int64)
                arrays[f'admission_{mode}_{qi}']=np.asarray(admission,np.int64)
                selection_counts[f'{mode}_{qi}']={'semantic_unique':len(admission),'semantic_return_slots':128*cfg['semantic_depth'],
                    'lexical_selected_valid':counters['lexical_selected_valid'],'planner_candidates':len(expected)}
                if mode=='full':arrays[f'planner_{qi}']=expected
                elif mode!='plain_semantic':
                    full=arrays[f'selection_full_{qi}'];sem=np.asarray([r for r in selected if r[0]==1],np.int64)
                    if not np.array_equal(sem,full[full[:,0]==1]):raise ValueError('Lexical-only component changed semantic candidates')
            print(json.dumps({'dataset':dataset,'stage':'candidate_coverage','selector_complete':mode,'queries':16}),flush=True)
        finally:index.close()
    for qi,query in enumerate(queries):
        full=arrays[f'admission_full_{qi}'];plain=arrays[f'admission_plain_semantic_{qi}']
        budget=min(len(full),len(plain));assert budget>=100
        budgets[str(query['id'])]=budget
        checks[str(qi)].update(component_budget=budget,full_unique=len(full),plain_semantic_unique=len(plain),
            selection_basis='Rank-major round robin over ordered128 route cells; stable global-row first occurrence, then common prefix.')
        for mode in ('late_route','no_native','fixed_rrf'):
            arrays[f'selection_{mode}_{qi}']=arrays[f'selection_full_{qi}']
            arrays[f'admission_{mode}_{qi}']=full
    # Tokens are regenerated in every actual online request, not read from
    # these owner-only diagnostic snapshots by the server or timed client.
    arrays.update(query_ids=np.asarray([str(q['id']) for q in queries]),query_indices=np.asarray([q['index'] for q in queries],np.int64),
                  fresh_embeddings=np.stack(fresh),component_budgets=np.asarray(list(budgets.values()),np.int32))
    private['component_budgets']=budgets
    np.save(own/'coverage_keys.npy',np.asarray(sorted(keys),np.int64),allow_pickle=False)
    np.savez_compressed(own/'coverage_reference.npz',**arrays)
    if source_hashes()!=initial:raise ValueError('Preparation source changed during coverage')
    write(own/'client_private.json',private)
    report={'schema':'component-owner-coverage-v1','role':'owner-only','status':'complete','dataset':dataset,'source_sha256':initial,
        'queries':16,'fresh_runtime':runtime,'query_checks':checks,'selection_counts':selection_counts,'requests':requests,
        'covered_keys':len(keys),'coverage_keys':spec(own/'coverage_keys.npy'),'coverage_reference':spec(own/'coverage_reference.npz'),
        'actual_selectors_executed':list(modes),'same_candidate_modes_assigned_offline':['late_route','no_native','fixed_rrf'],
        'same_candidate_modes_need_real_online_functional_gate':True,'formal_timing':False,'elapsed_seconds':time.perf_counter()-started}
    write(own/'coverage_report.json',report)
    print(json.dumps({'dataset':dataset,'stage':'coverage_complete','protected_keys':len(keys),'budget_min':min(budgets.values()),'budget_max':max(budgets.values())}),flush=True)
    return report


def build_store(dataset):
    own=ROOT/'.owner'/dataset;dest=ROOT/'prepared'/dataset;public=dest/'public';public.mkdir(parents=True,exist_ok=True)
    final=public/'records.sqlite3';work=dest/'records.building.sqlite3'
    if final.exists() or work.exists() or (dest/'preparation_complete.json').exists():raise FileExistsError('Existing store/completion will not be overwritten')
    coverage_report=read(own/'coverage_report.json');initial=source_hashes()
    if coverage_report['status']!='complete' or coverage_report['source_sha256']!=initial:raise ValueError('Stale/incomplete coverage source binding')
    private=read(own/'client_private.json');crypto=client.parent.Owner(private['dueter_precision_keys'])
    cfg=CONFIG['datasets'][dataset];source=SRC/cfg['semantic_cache'];started=time.perf_counter()
    built=read(dest/'indexes'/'plain_semantic_manifest.json')
    if built['status']!='complete':raise ValueError('Native index not complete')
    for binding in built['inputs'].values():server.check_stat(binding)
    offsets=np.load(source/'semantic_offsets.npy',mmap_mode='r');postings=np.load(source/'semantic_postings.npy',mmap_mode='r')
    keys=np.load(own/'coverage_keys.npy',allow_pickle=False)
    byrow=defaultdict(list)
    for p,c,r in keys:
        p,c,r=int(p),int(c),int(r);globalrow=int(postings[int(offsets[c])+r]) if p==1 else r
        byrow[globalrow].append((p,c,r))
    del offsets,postings,keys
    writer=records.StoreWriter(work);native=np.load(source/'corpus_embeddings.npy',mmap_mode='r')
    lengths=np.load(SRC/'cache/exact_bm25_client_rerank'/cfg['label']/'full_doc_lengths_v2.npy',mmap_mode='r')
    if lengths.shape!=(cfg['documents'],) or not np.issubdtype(lengths.dtype,np.integer):raise ValueError('Bad native length array')
    tokens=private['token_ids'];seen=total_bytes=total_rows=0;corpus=SRC/'data'/dataset/'corpus.jsonl'
    with corpus.open('rb') as stream,(source/'doc_ids.txt').open(encoding='utf-8') as idstream:
        for row,(line,docid) in enumerate(zip(stream,idstream)):
            total_rows+=1;total_bytes+=len(line)
            if row not in byrow:continue
            document=json.loads(line)
            if str(document['_id'])!=docid.strip():raise ValueError('Native/corpus document alignment differs')
            private['document_ids'][str(row)]=str(document['_id']);payloads={};paths={p for p,_,_ in byrow[row]}
            if 1 in paths:
                vector=np.asarray(native[row],dtype='<f2')
                if not np.isfinite(vector).all():raise ValueError('Nonfinite FP16 native record')
                payloads[1]=vector.tobytes()
            if 2 in paths:
                words=client.parent.tokenize(f"{document.get('title','')} {document.get('text','')}".strip())
                if len(words)!=int(lengths[row]):raise ValueError('Tokenization differs from native BM25 length')
                for word in words:
                    if word not in tokens:tokens[word]=len(tokens)
                payloads[2]=records.pad_lexical_payload(words,tokens)
            for path,cell,lookup in byrow[row]:
                protected=crypto.make_record(path,cell,row,payloads[path])
                if crypto.open_record(protected)!=(row,crypto.locator(row),1,payloads[path]):raise ValueError('Record AEAD roundtrip failed')
                writer.record(path,cell,lookup,protected)
            locator=crypto.locator(row);ciphertext=crypto.protect('document',line,locator)
            if crypto.open('document',ciphertext,locator)!=line:raise ValueError('Document AEAD roundtrip failed')
            writer.document(locator,ciphertext);seen+=1
            if seen%10000==0:
                writer.db.commit();print(json.dumps({'dataset':dataset,'stage':'protect_candidate_records','documents':seen,'target':len(byrow)}),flush=True)
        if stream.readline()!=b'' or idstream.readline()!='':raise ValueError('Paired corpus/ID stream length mismatch')
    if seen!=len(byrow) or total_rows!=cfg['documents'] or total_bytes!=corpus.stat().st_size:raise ValueError('Full source/coverage count mismatch')
    for cell in range(2048):writer.record(1,cell,-1,crypto.make_record(1,cell,records.DUMMY_ROW,bytes(768)))
    for cell in range(16):writer.record(2,cell,-1,crypto.make_record(2,cell,records.DUMMY_ROW,bytes(256)))
    metadata={'dataset':dataset,'method':'component-controls','format':'native-FP16-semantic-and-lexical-AEAD-v1',
        'scope':'Same sampled-workload union store for seven modes; owner preparation metadata excluded.'}
    payload_bytes,document_bytes=writer.payload_bytes,writer.document_bytes
    writer.finish(metadata)
    del native,lengths;gc.collect()
    # Independent persisted-byte authentication and complete Fetch closure.
    db=sqlite3.connect(f'file:{work.as_posix()}?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity check failed')
    if {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}!={'records','documents','metadata'}:raise ValueError('Unexpected public tables')
    if json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])!=metadata:raise ValueError('Public metadata mismatch')
    count=0;required={};nonces=set();logical=hashlib.sha256()
    import struct
    for path,cell,lookup,value in db.execute('SELECT path,cell,row,value FROM records ORDER BY path,cell,row'):
        row,locator,version,payload=crypto.open_record(value);count+=1
        if version==1:required[locator]=row
        if path==1 and (len(payload)!=768 or not np.isfinite(np.frombuffer(payload,dtype='<f2')).all()):raise ValueError('Persisted native precision invalid')
        if path==2 and version==1:records.open_lexical_payload(payload)
        _,plen,_=records.RECORD.unpack_from(value)
        for at in (24,24+plen):
            nonce=value[at:at+12]
            if nonce in nonces:raise ValueError('Nonce reused in fresh record namespace')
            nonces.add(nonce)
        logical.update(struct.pack('!iiqQ8sI',path,cell,lookup,row,locator,version));logical.update(payload)
    documents=0;covered=set()
    for locator,value in db.execute('SELECT locator,value FROM documents ORDER BY locator'):
        plaintext=crypto.open('document',value,locator);document=json.loads(plaintext);documents+=1
        if locator not in required or str(document['_id'])!=private['document_ids'][str(required[locator])]:raise ValueError('Fetch identity/closure mismatch')
        covered.add(locator)
        nonce=value[:12]
        if nonce in nonces:raise ValueError('Nonce reused across fresh domains')
        nonces.add(nonce)
    if covered!=set(required) or documents!=len(byrow):raise ValueError('Missing candidate Fetch document')
    db.close();work.rename(final);store_binding=spec(final)
    private['record_store_sha256']=store_binding['sha256'];write(own/'client_private.json',private)
    public_manifests={};index_cache={}
    for mode in server.MODES:
        bindings=server.source_index_specs(dataset,mode)
        # source_index_specs hashes the raw1024 bank once, only for PlainL.
        manifest={'schema':'component-server-public-v1','role':'server-public','dataset':dataset,'mode':mode,
            'parameters':{'semantic_eligible_probes':128,'semantic_cells_scored':2048 if mode=='late_route' else 128,
                'semantic_local_depth':cfg['semantic_depth'],'semantic_return_slots':128*cfg['semantic_depth'],
                'planner_cap':1000,'lexical_return_slots':300,'fetch_cap':20,'returned_semantic_dtype':'float16',
                'returned_semantic_dimension':384,'returned_semantic_payload_bytes':768},
            'index_files':bindings,'record_store':store_binding}
        path=public/f'server_manifest_{mode}.json';write(path,manifest);public_manifests[mode]=spec(path)
    if source_hashes()!=initial:raise ValueError('Preparation source changed during store construction')
    report={'schema':'component-preparation-v1','role':'coordinator-only','status':'complete','dataset':dataset,'formal_timing':False,
        'source_sha256':initial,'owner_private_binding':spec(own/'client_private.json'),'coverage_report':spec(own/'coverage_report.json'),
        'record_store':store_binding,'server_manifests':public_manifests,'protected_records':count,'protected_documents':documents,
        'protected_record_value_bytes':payload_bytes,'protected_document_value_bytes':document_bytes,
        'all_persisted_records_authenticated':True,'all_persisted_documents_authenticated':True,'all_candidates_have_Fetch_documents':True,
        'fresh_secret_keys':True,'unique_nonces_checked':len(nonces),'public_metadata_exact_keys':sorted(metadata),
        'logical_record_payload_sha256':logical.hexdigest(),'source_corpus_rows':total_rows,'source_corpus_bytes':total_bytes,
        'scope':'Full-corpus scientific indexes, newly prepared sampled-query union payload/document store; not full-corpus protected-store materialization.',
        'elapsed_seconds':time.perf_counter()-started}
    write(dest/'preparation_complete.json',report)
    print(json.dumps({'dataset':dataset,'stage':'preparation_complete','records':count,'documents':documents,'store_bytes':store_binding['bytes']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=tuple(CONFIG['datasets']));parser.add_argument('--stage',choices=('coverage','store','all'),default='all');args=parser.parse_args()
    if args.stage in ('coverage','all'):coverage(args.dataset)
    if args.stage in ('store','all'):build_store(args.dataset)
