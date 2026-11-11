"""Independent component checks. No numerical import before the live bootstrap.

install_numeric_audit is FUNCTIONAL ONLY: extra decrypt/recompute work must
never enter a formal timed run. CLI reconciliation is standard-library-only.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODES=('full','late_route','plain_semantic','plain_lexical','planner_quota_lexical','no_native','fixed_rrf')

def require(condition,message):
    if not condition: raise AssertionError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def spec(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def write(path,value):
    p=Path(path).resolve();require(ROOT in p.parents,'Audit output outside experiment')
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def stable_admission(groups):
    seen=set();out=[]
    for rank in range(max((len(g) for g in groups),default=0)):
        for group in groups:
            if rank<len(group) and int(group[rank])>=0 and int(group[rank]) not in seen:
                row=int(group[rank]);seen.add(row);out.append(row)
    return out

def fixed_rrf(semantic,lexical,weight):
    scores={}
    for w,ranking in ((weight,semantic),(1-weight,lexical)):
        if w<=0:continue
        for i,row in enumerate(ranking):
            if row>=0:scores[row]=scores.get(row,0.)+w/(61+i)
    return sorted(scores,key=lambda row:(-scores[row],row))[:100]

def sample_metrics(document_ids,relevant):
    positive={str(k):float(v) for k,v in relevant.items() if float(v)>0}
    ideal=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(positive.values(),reverse=True)[:10]))
    dcg=sum((2**positive.get(str(d),0)-1)/math.log2(i+2) for i,d in enumerate(document_ids[:10]))
    return {'ndcg_at_10':dcg/ideal if ideal else 0.,
            'recall_at_100':len(set(map(str,document_ids[:100]))&positive.keys())/max(1,len(positive))}

def install_numeric_audit(owner):
    """Called only after actual pinned runtime/owner initialization."""
    require(not getattr(owner,'_component_audit_installed',False),'Duplicate audit hook')
    np=sys.modules.get('numpy');require(np is not None,'Runtime must be bootstrapped first')
    original=owner.rerank
    def checked(text,embedding,response_meta,records,trace,*args,**kwargs):
        result=original(text,embedding,response_meta,records,trace,*args,**kwargs)
        top20,locators,m=result;slots=response_meta['slots'];groups=[[] for _ in range(128)]
        sem={};lex={};locator_map={};dummy=0;wire_sem=0;wire_lex=0
        for (path,group,ordinal),record in zip(slots,records):
            row,locator,version,payload=owner.owner.open_record(record)
            require(path in (1,2),'Unexpected path in independent decoder')
            wire_sem+=path==1;wire_lex+=path==2
            if version==0:dummy+=1;continue
            require(version==1,'Unknown authenticated version')
            if row in locator_map:require(locator_map[row]==locator,'Independent join mismatch')
            locator_map[row]=locator
            if path==1:
                require(0<=int(group)<128 and len(payload)==768,'Semantic group/FP16 payload')
                vector=np.frombuffer(payload,dtype='<f2').astype(np.float32)
                require(bool(np.isfinite(vector).all()),'Nonfinite semantic payload')
                if row in sem:require(np.array_equal(sem[row],vector),'Replicas disagree on native bytes')
                sem[row]=vector;groups[int(group)].append(int(row))
            else:
                require(row not in lex,'Duplicate lexical candidate')
                lex[row]=(int(ordinal),payload)
        qid=str(trace['query_id']);budget=int(owner.private['component_budgets'][qid])
        prefix=stable_admission(groups);admitted=prefix[:budget]
        require(len(admitted)==budget and admitted==m['semantic_admitted_rows'],'Independent admission differs')
        require(len(prefix)==m['semantic_unique_candidates'],'Raw unique counter differs')
        no_native=owner.component_mode=='no_native'
        if no_native:expected_sem=admitted[:100]
        else:
            rows=np.asarray(sorted(admitted),dtype=np.int64)
            matrix=np.asarray([sem[int(row)] for row in rows],dtype=np.float32,order='C')
            score=matrix@np.asarray(embedding,dtype=np.float32)
            expected_sem=rows[np.lexsort((rows,-score))[:100]].tolist()
        actual_sem=[int(r) for r in m['semantic_rank100'] if r>=0]
        require(expected_sem==actual_sem,'Independent native semantic rank differs')
        planner_order=sorted(lex,key=lambda row:lex[row][0]);expected_lex=planner_order
        if not no_native:
            # Independently decode the wire's uint32 count/token array and use
            # a scalar BM25 formula, rather than the client's CSR/vector code.
            import struct
            parent=sys.modules['_components_frozen_fp16_client'].parent
            terms=list(dict.fromkeys(parent.tokenize(text)))
            pairs=[]
            for term in terms:
                token=owner.private['token_ids'].get(term)
                df=owner.private['document_frequencies'].get(parent.keyed_term(term),0)
                if token is not None and df>0:pairs.append((int(token),int(df)))
            values={};n=owner.cfg['documents'];average=owner.private['average_document_length']
            for row in planner_order:
                payload=lex[row][1];count=struct.unpack_from('!I',payload)[0]
                require(4+4*count<=len(payload),'Lexical payload count exceeds bytes')
                ids=[x[0] for x in struct.iter_unpack('!I',payload[4:4+4*count])]
                counts=collections.Counter(ids);norm=1.2*(1-.75+.75*count/average)
                values[row]=sum(math.log1p((n-df+.5)/(df+.5))*counts[token]*2.2/max(counts[token]+norm,1e-12) for token,df in pairs)
            expected_lex=sorted(planner_order,key=lambda row:(-values[row],row))
        actual_lex=[int(r) for r in m['lexical_rank'] if r>=0]
        require(expected_lex==actual_lex,'Independent native BM25/planner rank differs')
        require(m['semantic_scored_candidates']==(0 if no_native else budget),'Semantic score count')
        require(m['lexical_exact_scored']==(0 if no_native else len(lex)),'Lexical score count')
        require(m['returned_record_open_count']==len(records),'All record authentication count')
        require(m['semantic_wire_slots']==wire_sem and m['lexical_wire_slots']==wire_lex,'Wire slot counters')
        require(m['dummy_slots']==dummy,'Dummy decode count')
        require(top20==m['fused_rank100'][:20] and locators==[locator_map[r] for r in top20],'Independent Fetch locators')
        if owner.component_mode=='fixed_rrf':
            require(fixed_rrf(actual_sem,actual_lex,owner.fixed_rrf_weight)==m['fused_rank100'],'Independent calibrated RRF')
        m['independent_numeric_audit']={'status':'passed','script_sha256':sha(__file__),
            'authenticated_record_count':len(records),'semantic_fp16_vectors_verified':sum(map(len,groups)),
            'admission_exact':True,'native_semantic_rank_exact':not no_native,
            'native_bm25_rank_exact':not no_native,'no_native_rank_policy_exact':no_native,
            'fixed_rrf_exact':owner.component_mode=='fixed_rrf','fetch_locator_binding_exact':True,
            'scope':'Functional-only extra decrypt/recompute; these timings are never used as formal costs.'}
        return result
    owner.rerank=checked;owner._component_audit_installed=True

def context(dataset,metadata):
    config=read(metadata['config']['path']);private=read(metadata['private']['path'])
    require(sha(metadata['config']['path'])==metadata['config']['sha256'],'Config changed')
    require(sha(metadata['private']['path'])==metadata['private']['sha256'],'Private metadata changed')
    origin=private['component_origin']
    require(sha(origin['owner_private_path'])==origin['owner_private_sha256'],'Original cohort binding changed')
    old=read(origin['owner_private_path'])
    require([(str(q['id']),q['index'],q['text']) for q in private['queries']]==[(str(q['id']),q['index'],q['text']) for q in old['queries']],'Original live query cohort changed')
    cfg=config['datasets'][dataset];qids={str(q['id']) for q in private['queries']}
    path=Path(config['original_root'])/'data'/dataset/'qrels'/(cfg['split']+'.tsv')
    qrels={q:{} for q in qids}
    with path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.reader(handle,delimiter='\t');next(reader)
        for q,d,g,*_ in reader:
            if q in qrels:qrels[q][d]=float(g)
    return config,private,qrels,spec(path)

def check_runtime(runtime):
    binding=read(runtime['binding_path'])
    require(sha(runtime['binding_path'])==runtime['binding_sha256'],'Runtime spec changed')
    require(runtime['execution_device']=='cpu' and runtime['encoder_parameter_devices']==['cpu'],'Wrong model device')
    require(runtime['torch_threads']==runtime['torch_interop_threads']==2,'Wrong CPU thread count')
    for name,expected in binding['packages'].items():
        actual=runtime['packages'][name]
        require(actual['version']==expected['version'] and Path(actual['file']).resolve()==Path(expected['file']).resolve(),'Runtime provider mismatch: '+name)
    require(runtime['packages']['torch']['version']=='2.12.1+cu126','Unregistered Torch version')

def audit_dataset(directory,functional,source_hashes):
    dataset=directory.name;summary=read(directory/'summary.json');metadata=read(directory/'metadata.json')
    require(summary['status']=='complete','Incomplete dataset')
    require(summary['source_sha256']==metadata['source_sha256']==source_hashes,'Run source binding mismatch')
    profile=metadata['profile']
    require(profile['upload_mbps']==profile['download_mbps']==100 and profile['rtt_ms']==20,'Unregistered link profile')
    check_runtime(metadata['runtime_binding']);config,private,qrels,qrel_binding=context(dataset,metadata)
    setup=read(directory/'client_setup.json');expected_model=config['protocol']['datasets'][dataset]['modes']['projectlift']
    for key in ('fusion_model','fusion_report'):
        binding=expected_model[key];require(sha(binding['path'])==binding['sha256'],'Frozen final257 fusion artifact changed')
    for mode in MODES:
        binding=setup['ranker_bindings'][mode]
        require(binding['model']==expected_model['fusion_model'] and binding['report']==expected_model['fusion_report'],'Wrong mode-specific frozen model')
    queries=private['queries'];full_qids=[str(q['id']) for q in queries]
    require(len(full_qids)==len(set(full_qids))==16,'Query cohort changed')
    expected_qids={full_qids[0],full_qids[-1]} if functional else set(full_qids)
    phases={'functional':1} if functional else {'warmup':2,'measurement':3}
    observations=[];batch_records=[];seen=set();bykey={};files=[]
    for file in sorted((directory/'batches').glob('*.json')):
        b=read(file);key=(b['phase'],b['repeat'],b['mode'])
        require(key not in seen,'Duplicate batch');seen.add(key)
        require(b['phase'] in phases and 0<=b['repeat']<phases[b['phase']] and b['mode'] in MODES,'Unexpected batch')
        obs=b['observations'];require({str(o['query_id']) for o in obs}==expected_qids and len(obs)==len(expected_qids),'Missing/duplicate query')
        require(b['batch_wall_seconds']>0,'Invalid observed batch wall time')
        require(b['batch_wall_seconds']*1000+1>=sum(o['end_to_end_ms'] for o in obs),'Sequential wall time below summed endpoint intervals')
        for direction,field in [('upload','sent_application_bytes'),('download','received_application_bytes')]:
            total=sum(o[k+'_rpc'][field] for o in obs for k in ('search','fetch'))
            proxy=b['proxy_reconciliation']['directions'][direction]
            require(proxy['bytes']==total and proxy['frames']==2*len(obs),'Proxy bytes/frame mismatch')
            require(abs(proxy['serialization_ns_total']-total*80)<=proxy['chunks']+2,'100Mbps serialization charge mismatch')
        for o in obs:
            qid=str(o['query_id']);mode=b['mode'];budget=int(private['component_budgets'][qid]);cap=128*config['datasets'][dataset]['semantic_depth']
            require(o['component_mode']==mode and o['mode']==mode and o['phase']==b['phase'] and o['repeat']==b['repeat'],'Observation identity mismatch')
            require(o['semantic_admitted_candidates']==o['matched_semantic_budget']==budget,'Common budget mismatch')
            admitted=o['semantic_admitted_rows'];require(len(admitted)==len(set(admitted))==budget,'Admitted IDs not distinct/budgeted')
            require(o['semantic_unique_candidates']-budget==o['semantic_discarded_unique_before_scoring']>=0,'Discard count mismatch')
            require(o['semantic_wire_slots']==cap and o['lexical_wire_slots']==300 and o['returned_record_open_count']==cap+300,'Authentication/wire slots mismatch')
            require(o['semantic_return_element_bits']==16 and o['semantic_return_payload_bytes']==768,'FP16 return precision mismatch')
            expected_sem_bytes=384*4 if mode=='plain_semantic' else (2048 if mode=='late_route' else 128)*256*2
            expected_lex_bytes=0 if mode=='planner_quota_lexical' else 1024*4 if mode=='plain_lexical' else 16*257*2
            require(o['semantic_wire_coordinate_bytes']==expected_sem_bytes and o['lexical_wire_coordinate_bytes']==expected_lex_bytes,'Variant query representation bytes differ')
            no_native=mode=='no_native';valid=o['lexical_survivors']
            require(o['semantic_scored_candidates']==(0 if no_native else budget) and o['lexical_exact_scored']==(0 if no_native else valid),'Native work mismatch')
            server=o['search_server'];require(valid==min(300,server['planner_candidates'])==server['lexical_selected_valid'],'Lexical admission mismatch')
            require(sum(server['lexical_quotas'])==valid and len(server['lexical_quotas'])==16,'Lexical quota count')
            require(server['semantic_cells_scored']==(2048 if mode=='late_route' else 128) and server['semantic_eligible_cells']==128,'Pruning execution count')
            require(server['lexical_proxy_scored_candidates']==(0 if mode=='planner_quota_lexical' else server['planner_candidates']),'Lexical proxy work')
            sem=[r for r in o['semantic_rank100'] if r>=0];lex=[r for r in o['lexical_rank'] if r>=0];fused=o['fused_rank100']
            require(len(sem)==len(set(sem))==100 and set(sem)<=set(admitted),'Semantic top100 membership')
            require(len(lex)==len(set(lex))==valid and set(lex)==set(o['lexical_selected_rows']),'Lexical ranking membership')
            require(len(fused)==len(set(fused))==100 and set(fused)<=set(sem+lex),'Fused top100 membership')
            require(o['top20']==fused[:20] and o['fetched_documents']==20,'Fetch does not match fused top100')
            require(o['top20_sha256']==hashlib.sha256(json.dumps(o['top20']).encode()).hexdigest(),'Top20 digest mismatch')
            if no_native:require(sem==admitted[:100] and lex==o['lexical_selected_rows'],'No-native rank order')
            if mode=='fixed_rrf':require(fixed_rrf(sem,lex,{'nq':1.,'hotpotqa':.9,'msmarco':.9}[dataset])==fused,'Frozen RRF differs')
            if functional:
                audit=o.get('independent_numeric_audit',{});require(audit.get('status')=='passed' and audit['script_sha256']==sha(__file__),'Missing current independent numeric audit')
            else:require('independent_numeric_audit' not in o,'Functional hook contaminates formal timing')
            for side,field in [('upload','sent_application_bytes'),('download','received_application_bytes')]:
                require(o['actual_application_'+side+'_bytes']==sum(o[k+'_rpc'][field] for k in ('search','fetch')),'Endpoint byte total mismatch')
            docids=[private['document_ids'][str(row)] for row in fused]
            o=dict(o);o['joint_quality']=sample_metrics(docids,qrels[qid])
            o['outer_holdout']=bool(hashlib.sha256(('outer:'+qid).encode()).digest()[0]&1)
            observations.append(o);bykey[(b['phase'],b['repeat'],mode,qid)]=o
        batch_records.append({k:v for k,v in b.items() if k!='observations'});files.append(spec(file))
    expected={(phase,r,m) for phase,number in phases.items() for r in range(number) for m in MODES}
    require(seen==expected,'Missing or extra batch')
    stable_fields=('semantic_admitted_rows','semantic_rank100','lexical_rank','lexical_selected_rows','fused_rank100','top20','fetched_plaintext_bytes')
    for phase,number in phases.items():
        for r in range(number):
            for qid in expected_qids:
                full=bykey[(phase,r,'full',qid)]
                late=bykey[(phase,r,'late_route',qid)]
                plain=bykey[(phase,r,'plain_semantic',qid)]
                require(full['matched_semantic_budget']==min(full['semantic_unique_candidates'],plain['semantic_unique_candidates']),'Owner common budget is not min of actual raw uniques')
                require(late['search_server']['semantic_posting_reads']==2*config['datasets'][dataset]['documents'],'Late route did not score all replicated index entries')
                for field in stable_fields:require(full[field]==late[field],'Late-route selection/ranking differs: '+field)
                for mode in MODES:
                    row=bykey[(phase,r,mode,qid)]
                    if mode!='plain_semantic':require(row['semantic_admitted_rows']==full['semantic_admitted_rows'],'Unchanged semantic path differs')
                    if mode not in ('plain_lexical','planner_quota_lexical'):require(row['lexical_selected_rows']==full['lexical_selected_rows'],'Unchanged lexical selection differs')
    for mode in MODES:
        actual={}
        for record in read(directory/f'server_{mode}_requests.json')['requests']:
            key=(record['request_id'],record['kind']);require(key not in actual,'Duplicate server RPC');actual[key]=record
        items=[o for o in observations if o['mode']==mode];expected=set()
        for o in items:
            for kind in ('search','fetch'):
                key=(o['request_id'],kind);expected.add(key);server=actual[key];rpc=o[kind+'_rpc']
                require(rpc['sent_application_bytes']==server['received_application_bytes'] and rpc['received_application_bytes']==server['response_application_bytes'],'Independent server/client bytes differ')
        require(expected==set(actual),'Unexpected server RPC')
        for qid in expected_qids:
            rows=[o for o in items if str(o['query_id'])==qid]
            for row in rows[1:]:
                for field in stable_fields:require(row[field]==rows[0][field],'Repeated query scientific output changed')
    return {'dataset':dataset,'status':'passed','query_units':len(expected_qids),'observations':len(observations),
            'formal_measurements':sum(o['phase']=='measurement' for o in observations),'batches':batch_records,
            'files':files,'metadata':spec(directory/'metadata.json'),'qrels':qrel_binding,
            'outer_queries':sum(bool(hashlib.sha256(('outer:'+q).encode()).digest()[0]&1) for q in expected_qids)},observations

def audit_run(run):
    run=Path(run).resolve();suite=read(run/'suite.json')
    require(suite['status']=='complete','Suite incomplete');sources=suite['source_sha256']
    for path,value in sources.items():require(sha(path)==value,'Bound source changed: '+path)
    require(sources.get(str(Path(__file__).resolve()))==sha(__file__),'Validator absent from frozen execution sources')
    datasets={};allobs={}
    for dataset in suite['datasets']:
        result,obs=audit_dataset(run/dataset,suite['functional'],sources);datasets[dataset]=result;allobs[dataset]=obs
    if not suite['functional'] and set(datasets)=={'nq','hotpotqa','msmarco'}:
        require(sum(d['outer_queries'] for d in datasets.values())==22,'Registered mixed live cohort split changed')
    result={'schema':'matched-budget-component-independent-audit-v1','status':'passed',
            'accepted_for_timing':bool(suite['functional']),'datasets':list(datasets),
            'source_sha256':sources,'suite':spec(run/'suite.json'),'dataset_checks':datasets,
            'numeric_audit_required':bool(suite['functional']),'all_observations_retained':True,
            'scope':'Independent small-file protocol/numeric/bytes audit; no full-index scan or claims of statistical equivalence.'}
    return result,allobs

def self_test():
    require(stable_admission([[9,7,3],[4,9],[9,8,2]])==[9,4,7,8,3,2],'Admission self-test')
    require(fixed_rrf([4,2],[2,5],1.)==[4,2],'RRF endpoint self-test')
    q=sample_metrics(['a','b'],{'b':1});require(q['recall_at_100']==1 and abs(q['ndcg_at_10']-1/math.log2(3))<1e-15,'Metric self-test')
    return {'status':'passed','tests':3,'scope':'Standard library synthetic checks, no model/index I/O'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--acceptance',type=Path);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        if args.run is None:p.error('--run required')
        result,_=audit_run(args.run);dest=args.acceptance or args.run/'independent_audit.json';write(dest,result)
        print(json.dumps({'status':'passed','output':str(dest),'datasets':result['datasets'],'accepted_for_timing':result['accepted_for_timing']}))
