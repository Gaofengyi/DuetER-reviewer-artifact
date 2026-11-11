"""New retrieval-quality run; historical data and code are read-only.

Shared query transformations and candidate rule for the timing/control runners.
No attack module is imported or executed. No historical ranking is returned as
a fresh semantic/CCADPE result. Cached query embeddings and deterministic raw
planner candidates are explicitly treated as fixed experimental inputs.
"""
from __future__ import annotations
import os, sys, json, hashlib, math, time, sqlite3, hmac, gc, argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
CONFIG_PATH=HERE/'config.json'
def load_config(path=CONFIG_PATH):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    n=c['noise']; assert math.isclose(n['query_radius'],n['scale']*n['beta']/8)
    assert math.isclose(n['database_radius'],3*n['scale']*n['beta']/8)
    return c
C=load_config(); SRC=Path(C['original_root'])
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]=str(C['runtime']['blas_threads'])
sys.path[:0]=[str(SRC/'.gpu_deps'),str(SRC/'.deps'),str(SRC)]
import torch
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits,threadpool_info
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from benchmark_dual_compartment_full import cell_transform,apply_transform,top_indices
from benchmark_million_semantic_hybrid import load_queries_qrels,read_ids,dpe_transform,ball_noise,evaluate
from benchmark_million_lexical_dpe import query_mips
from benchmark_exact_bm25_client_rerank import query_vocabulary,load_document_frequencies
from experiment_candidate_rank_fusion import split_mask,training_matrix,learned_rankings,mix_rankings
from experiment_adaptive_fusion import query_features,ndcg_rows
from benchmark_budgeted_lexical_candidates import fuse_all
from run_experiment import tokenize

torch.set_num_threads(C['runtime']['blas_threads'])
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()

def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    assert HERE.resolve() in path.resolve().parents
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)

def stats(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95)),'min':float(x.min()),'max':float(x.max())}

def query_seed(query_id,path,repeat=0,config=None):
    c=config or C; r=c['seed']+c['query_seed_offset']+repeat*c['repeat_seed_stride']
    return int.from_bytes(hashlib.sha256(f'{r}:{path}:{query_id}'.encode()).digest()[:8],'big')

def prepare_dataset(dataset,config=None,query_limit=None):
    c=config or C; d=c['datasets'][dataset]
    ids,texts,qrels,_=load_queries_qrels(SRC/'data'/dataset,d['split'])
    assert len(ids)==d['queries']
    source=SRC/d['semantic_cache']
    q=np.load(source/d['query_embeddings_file'],mmap_mode='r')
    assert len(q)==len(ids)
    holdout=split_mask(ids,'outer')
    assert int(holdout.sum())==d['outer_holdout_queries']
    if query_limit is not None:
        ids=ids[:query_limit];texts=texts[:query_limit];q=q[:query_limit];holdout=holdout[:query_limit]
    return {'dataset':dataset,'cfg':d,'query_ids':ids,'query_texts':texts,'qrels':qrels,
            'queries':np.asarray(q,dtype=np.float32),'outer_holdout':holdout,'semantic_source':source,
            'source':SRC/d['lexical_source'],'lexical_cache':SRC/d['lexical_local_cache'],
            'semantic_cache':SRC/d['semantic_local_cache']}

def semantic_global_queries(queries,query_ids,key_path,repeat=0,config=None):
    c=config or C; key=np.load(key_path)
    base=dpe_transform(np.asarray(queries,dtype=np.float32),key['sign1'],key['sign2'],key['permutation'])
    noise=np.concatenate([ball_noise(np.random.default_rng(query_seed(qid,'semantic',repeat,c)),1,base.shape[1],c['noise']['query_radius']) for qid in query_ids])
    return (c['noise']['scale']*base+noise).astype(np.float32)

def semantic_query_views(global_query,cells,config=None):
    c=config or C; p=c['semantic'];key=p['key_label'].encode()
    return np.stack([apply_transform(global_query,cell_transform(key,int(cell),p['work_dimension'],p['local_dimension'])).astype(np.float16) for cell in cells])

def lexical_key_arrays(config=None):
    c=config or C;dim=c['lexical']['work_dimension'];rng=np.random.default_rng(c['seed']+2)
    return (rng.choice(np.asarray([-1.,1.],dtype=np.float32),dim),
            rng.choice(np.asarray([-1.,1.],dtype=np.float32),dim),rng.permutation(dim))

def selected_matrix(coords,input_dimension,config=None):
    c=config or C;dim=c['lexical']['work_dimension'];s1,s2,perm=lexical_key_arrays(c)
    rows=perm[np.asarray(coords,dtype=np.int64)];cols=np.arange(input_dimension,dtype=np.int64)
    parity=np.asarray([int(i).bit_count()&1 for i in range(dim)],dtype=np.int8)
    h=(1.-2.*parity[np.bitwise_and(rows[:,None],cols[None,:])].astype(np.float32))/math.sqrt(dim)
    return (h*s1[:input_dimension][None,:]*s2[rows][:,None]).astype(np.float32)

def selected_noise(qid,repeat,selected_dimension,config=None):
    c=config or C;dim=c['lexical']['work_dimension'];rng=np.random.default_rng(query_seed(qid,'lexical',repeat,c))
    g=rng.normal(size=(1,selected_dimension)).astype(np.float32)
    omitted=rng.chisquare(dim-selected_dimension,size=1).astype(np.float32)
    norm=np.sqrt(np.sum(g*g,axis=1)+omitted).astype(np.float32)
    radius=c['noise']['query_radius']*np.power(rng.random(1),1./dim).astype(np.float32)
    return g[0]*(radius/np.maximum(norm,1e-12))[0]

def lexical_query_views(query_plain,query_ids,repeat=0,config=None):
    c=config or C;p=c['lexical'];transforms=[cell_transform(p['key_label'].encode(),cell,p['work_dimension'],p['local_dimension']) for cell in range(p['partitions'])]
    union=np.unique(np.concatenate([t.coordinates for t in transforms])).astype(np.int32)
    # Fixed GEMV per query prevents batch-dependent GEMM accumulation from
    # crossing FP16 rounding boundaries in online-vs-offline query tokens.
    matrix=selected_matrix(union,query_plain.shape[1],c)
    base=c['noise']['scale']*np.stack([np.asarray(row,dtype=np.float32)@matrix.T for row in query_plain])
    base+=np.stack([selected_noise(qid,repeat,len(union),c) for qid in query_ids])
    result=np.empty((len(query_ids),p['partitions'],p['local_dimension']),dtype=np.float16)
    for cell,t in enumerate(transforms):
        positions=np.searchsorted(union,t.coordinates)
        result[:,cell]=(base[:,positions]*t.signs*t.multiplier+t.translation).astype(np.float16)
    return result

def keyed_term(term):
    from benchmark_million_semantic_hybrid import LEXICAL_KEY
    return hmac.new(LEXICAL_KEY,term.encode(),hashlib.sha256).hexdigest()[:32]

def lexical_plain_queries(query_texts,dataset_cfg,config=None):
    c=config or C; tokens=sorted({keyed_term(t) for q in query_texts for t in tokenize(q)})
    conn=sqlite3.connect(f"file:{(SRC/dataset_cfg['fts_database']).as_posix()}?mode=ro",uri=True)
    df={}
    for start in range(0,len(tokens),800):
        batch=tokens[start:start+800];marks=','.join('?' for _ in batch)
        df.update({str(t):int(n) for t,n in conn.execute(f'SELECT term, doc FROM vocab WHERE term IN ({marks})',batch)})
    conn.close()
    return np.stack([query_mips(q,df,dataset_cfg['documents'],c['lexical']['hash_dimension']) for q in query_texts])

def lexical_select(raw,query_views,candidate_rows,assignments,local,norms,config=None):
    c=config or C;p=c['lexical'];candidates=np.asarray(raw[raw>=0],dtype=np.int32)
    pos=np.searchsorted(candidate_rows,candidates)
    if len(pos) and (int(pos.max())>=len(candidate_rows) or np.any(candidate_rows[pos]!=candidates)):raise ValueError('planner candidate absent from local index scope')
    cells=np.asarray(assignments[pos]);selected=[]
    for cell in np.unique(cells):
        ordinals=np.flatnonzero(cells==cell);lp=pos[ordinals]
        scores=2.*(np.asarray(local[lp],dtype=np.float32)@np.asarray(query_views[int(cell)],dtype=np.float32))-np.asarray(norms[lp])
        order=top_indices(scores,min(p['local_depth'],len(scores)))
        selected.extend(map(int,ordinals[order]))
    survivor_ordinals=np.unique(selected).astype(np.int64)
    out=np.full(p['exact_scoring_cap'],-1,dtype=np.int32)
    keep=candidates[survivor_ordinals][:p['exact_scoring_cap']]
    out[:len(keep)]=keep
    return out,int(len(survivor_ordinals))

def run_lexical(dataset,query_limit=None):
    ds=prepare_dataset(dataset,query_limit=query_limit);cfg=ds['cfg'];dest=HERE/('smoke' if query_limit else 'results')/cfg['label'];dest.mkdir(parents=True,exist_ok=True)
    output_path=dest/'lexical_candidates.npz'
    if output_path.exists():raise FileExistsError(output_path)
    t=time.perf_counter();plain=lexical_plain_queries(ds['query_texts'],cfg)
    raw=np.asarray(np.load(SRC/cfg['planner_cache'])[f"budget_{cfg['posting_budget']}"][:len(plain),:C['lexical']['planner_cap']],dtype=np.int32)
    rows=np.load(ds['source']/'candidate_rows.npy',mmap_mode='r')
    assignments=np.load(ds['lexical_cache']/'lexical_compartment_cells.npy',mmap_mode='r')
    local=np.load(ds['lexical_cache']/'lexical_compartment_fp16.npy',mmap_mode='r')
    norms=np.load(ds['lexical_cache']/'lexical_compartment_norms.npy',mmap_mode='r')
    allout=[];counts=[];times=[]
    for repeat in range(C['repeats']):
        views=lexical_query_views(plain,ds['query_ids'],repeat)
        out=np.full((len(plain),C['lexical']['exact_scoring_cap']),-1,dtype=np.int32)
        for qi in range(len(plain)):
            start=time.perf_counter();out[qi],n=lexical_select(raw[qi],views[qi],rows,assignments,local,norms)
            times.append((time.perf_counter()-start)*1000);counts.append(n)
        allout.append(out)
        print(dataset,'lexical repeat',repeat,'done',flush=True)
    arr=np.stack(allout);np.savez_compressed(output_path,lexical_dpe=arr,lexical=arr,query_ids=np.asarray(ds['query_ids']),outer_holdout=ds['outer_holdout'])
    write_json(dest/'lexical.json',{'dataset':dataset,'query_count':len(plain),'repeats':C['repeats'],'query_radius':C['noise']['query_radius'],'elapsed_seconds':time.perf_counter()-t,
        'pre_cap_candidates':stats(counts),'exact_input_candidates':stats((arr>=0).sum(axis=-1).reshape(-1)),'refinement_and_cap_ms':stats(times),
        'planner_input':'cached deterministic FTS5 planner IDs; not counted as fresh planner execution or lookup timing',
        'config_sha256':sha(CONFIG_PATH),'script_sha256':sha(__file__)})
    return output_path

def run_semantic(dataset,query_limit=None):
    ds=prepare_dataset(dataset,query_limit=query_limit);cfg=ds['cfg'];p=C['semantic'];dest=HERE/('smoke' if query_limit else 'results')/cfg['label'];dest.mkdir(parents=True,exist_ok=True)
    final_path=dest/'semantic.rankings.npz'
    if final_path.exists():raise FileExistsError(final_path)
    start=time.perf_counter();source=ds['semantic_source'];q=ds['queries'];nq=len(q);depth=cfg['semantic_depth'];repeats=C['repeats']
    mean=np.load(source/'semantic_mean.npy',mmap_mode='r');centroids=np.load(source/'semantic_centroids.npy',mmap_mode='r')
    offsets=np.load(source/'semantic_offsets.npy',mmap_mode='r');postings=np.load(source/'semantic_postings.npy',mmap_mode='r')
    local=np.load(ds['semantic_cache']/'semantic_compartment_fp16.npy',mmap_mode='r');norms=np.load(ds['semantic_cache']/'semantic_compartment_norms.npy',mmap_mode='r')
    residual=q-np.asarray(mean);residual/=np.maximum(np.linalg.norm(residual,axis=1,keepdims=True),1e-12)
    with torch.inference_mode():
        route=torch.from_numpy(residual).cuda()@torch.from_numpy(np.array(centroids)).cuda().T
        cells=torch.topk(route,k=p['probes'],dim=1).indices.cpu().numpy().astype(np.int32);del route
    global_q=np.stack([semantic_global_queries(q,ds['query_ids'],source/'semantic_dpe_key.npz',r) for r in range(repeats)])
    emitted=np.full((repeats,nq,p['probes'],depth),-1,dtype=np.int32)
    bycell=[[] for _ in range(p['cells'])]
    for qi,cc in enumerate(cells):
        for ordinal,cell in enumerate(cc):bycell[int(cell)].append((qi,ordinal))
    cloud_start=time.perf_counter();read_counts=np.zeros(nq,dtype=np.int64)
    with torch.inference_mode():
        for cell,locations in enumerate(bycell):
            if not locations:continue
            first,last=int(offsets[cell]),int(offsets[cell+1])
            if first==last:continue
            qi=np.asarray([a for a,b in locations]);ords=np.asarray([b for a,b in locations]);read_counts[qi]+=last-first
            docs=np.asarray(postings[first:last],dtype=np.int32)
            transform=cell_transform(p['key_label'].encode(),cell,p['work_dimension'],p['local_dimension'])
            rows_gpu=torch.from_numpy(np.array(local[first:last],dtype=np.float32)).cuda()
            norms_gpu=torch.from_numpy(np.array(norms[first:last],dtype=np.float32)).cuda()
            for repeat in range(repeats):
                views=apply_transform(global_q[repeat,qi],transform).astype(np.float16).astype(np.float32)
                for b in range(0,len(qi),C['runtime']['query_batch_size']):
                    end=min(b+C['runtime']['query_batch_size'],len(qi))
                    qgpu=torch.from_numpy(views[b:end]).cuda()
                    scores=2.*(qgpu@rows_gpu.T)-norms_gpu[None,:]
                    selected=torch.topk(scores,k=min(depth,len(docs)),dim=1).indices.cpu().numpy()
                    emitted[repeat,qi[b:end],ords[b:end],:selected.shape[1]]=docs[selected]
                    del qgpu,scores
            del rows_gpu,norms_gpu
            if cell==0 or (cell+1)%128==0:print(dataset,'semantic cell',cell+1,'/',p['cells'],'seconds',round(time.perf_counter()-cloud_start,1),flush=True)
    torch.cuda.synchronize();cloud_seconds=time.perf_counter()-cloud_start
    np.savez_compressed(dest/'semantic_emitted.npz',emitted=emitted,query_cells=cells,query_ids=np.asarray(ds['query_ids']))
    embeddings=np.load(source/'corpus_embeddings.npy',mmap_mode='r')
    ranked=np.full((repeats,nq,p['output_depth']),-1,dtype=np.int32);counts=[];rrtime=[]
    for repeat in range(repeats):
        for qi in range(nq):
            ts=time.perf_counter();ids=np.unique(emitted[repeat,qi]);ids=ids[ids>=0]
            scores=np.asarray(embeddings[ids],dtype=np.float32)@q[qi]
            order=np.lexsort((ids,-scores))[:p['output_depth']]
            ranked[repeat,qi,:len(order)]=ids[order];counts.append(len(ids));rrtime.append((time.perf_counter()-ts)*1000)
            if (qi+1)%1000==0:print(dataset,'semantic rerank',repeat,qi+1,'/',nq,flush=True)
    np.savez_compressed(final_path,semantic=ranked,query_ids=np.asarray(ds['query_ids']),outer_holdout=ds['outer_holdout'])
    write_json(dest/'semantic.json',{'dataset':dataset,'query_count':nq,'repeats':repeats,'local_depth':depth,'query_radius':C['noise']['query_radius'],
      'coordinate_dtype':'float16','arithmetic':'float32 CUDA mm, TF32 disabled; float32 CPU cosine; final ties by doc row',
      'query_seed_policy':C['query_seed_policy'],'cloud_batch_seconds':cloud_seconds,'rerank_ms':stats(rrtime),'candidates':stats(counts),'posting_reads':stats(read_counts),
      'elapsed_seconds':time.perf_counter()-start,'config_sha256':sha(CONFIG_PATH),'script_sha256':sha(__file__),'not_end_to_end':True,
      'gpu_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'inputs_reused':'fixed indexes, embeddings and query embeddings; all query-noise/refinement/reranking recomputed'})
    print(dataset,'semantic complete',round(time.perf_counter()-start,2),'seconds',flush=True)
    return final_path

def run_bm25(dataset):
    ds=prepare_dataset(dataset);cfg=ds['cfg'];dest=HERE/'results'/cfg['label'];outpath=dest/'exact_bm25.rankings.npz'
    if outpath.exists():raise FileExistsError(outpath)
    cache=HERE/'direct_bm25'/dataset/'payload_cache'
    required=['rows.npy','token_lengths.npy','indptr.npy','indices.npy','counts.npy']
    if not all((cache/f).exists() for f in required):raise FileNotFoundError('Waiting for shared direct-BM25 payload cache: '+str(cache))
    arrays={k[:-4]:np.load(cache/k,mmap_mode='r') for k in required};rows=arrays['rows']
    terms,t2i=query_vocabulary(ds['query_texts']);df=load_document_frequencies(SRC/cfg['fts_database'],terms)
    full_lengths=np.load(SRC/'cache/exact_bm25_client_rerank'/cfg['label']/'full_doc_lengths_v2.npy',mmap_mode='r')
    matrix=sparse.csr_matrix((arrays['counts'].astype(np.float32),arrays['indices'].astype(np.int32),arrays['indptr'].astype(np.int64)),shape=(len(rows),len(terms)))
    incoming=np.load(dest/'lexical_candidates.npz')['lexical_dpe'];output=np.full_like(incoming,-1);times=[]
    avg=float(np.mean(full_lengths));N=cfg['documents'];started=time.perf_counter()
    for repeat in range(C['repeats']):
        for qi,text in enumerate(ds['query_texts']):
            ts=time.perf_counter();ids=incoming[repeat,qi];ids=ids[ids>=0];pos=np.searchsorted(rows,ids)
            assert np.array_equal(rows[pos],ids)
            columns=[t2i[t] for t in dict.fromkeys(tokenize(text)) if t in t2i and df[t2i[t]]>0]
            if columns:
                tf=matrix[pos][:,columns].toarray().astype(np.float64);dl=np.asarray(arrays['token_lengths'][pos],dtype=np.float64)[:,None]
                dfs=df[columns].astype(np.float64);idf=np.log1p((N-dfs+.5)/(dfs+.5))[None,:]
                norm=1.2*(1.-.75+.75*dl/avg);score=np.sum(idf*tf*2.2/np.maximum(tf+norm,1e-12),axis=1)
            else:score=np.zeros(len(ids))
            order=np.lexsort((ids,-score));output[repeat,qi,:len(ids)]=ids[order];times.append((time.perf_counter()-ts)*1000)
        print(dataset,'exact BM25 repeat',repeat,'done',flush=True)
    np.savez_compressed(outpath,lexical_dpe=output,lexical=output,query_ids=np.asarray(ds['query_ids']),outer_holdout=ds['outer_holdout'])
    write_json(dest/'bm25.json',{'elapsed_seconds':time.perf_counter()-started,'per_query_repeat_ms':stats(times),'candidate_mean':float(np.mean((incoming>=0).sum(axis=-1))),
      'scope':'native BM25 from existing shared query-vocabulary CSR; scoring once per actual repeat subset, no cross-repeat scoring union; not network/payload-decryption timing',
      'payload_cache':str(cache),'term_vocabulary_sha256':hashlib.sha256('\n'.join(terms).encode()).hexdigest(),'config_sha256':sha(CONFIG_PATH)})
    return outpath

def bootstrap(base,system,mask):
    f=C['fusion'];diff=np.mean(system[:,mask]-base[:,mask],axis=0);rng=np.random.default_rng(f['bootstrap_seed']);n=f['bootstrap_samples'];v=np.empty(n)
    for start in range(0,n,250):
        end=min(start+250,n);idx=rng.integers(0,len(diff),size=(end-start,len(diff)));v[start:end]=diff[idx].mean(axis=1)
    return {'mean_absolute_gain':float(diff.mean()),'ci95_low':float(np.percentile(v,2.5)),'ci95_high':float(np.percentile(v,97.5)),
            'one_sided_bootstrap_p':float((1+np.sum(v<=0))/(n+1)),'query_win_fraction':float(np.mean(diff>1e-12)),'query_tie_fraction':float(np.mean(np.abs(diff)<=1e-12)),'query_loss_fraction':float(np.mean(diff< -1e-12))}

def run_fusion(dataset,lexical_path=None,destination=None):
    ds=prepare_dataset(dataset);cfg=ds['cfg'];semantic_dest=HERE/'results'/cfg['label'];dest=Path(destination) if destination is not None else semantic_dest;dest.mkdir(parents=True,exist_ok=True);f=C['fusion'];out=dest/'fusion.json'
    assert HERE.resolve() in dest.resolve().parents
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter();ids=ds['query_ids'];texts=ds['query_texts'];truth=ds['qrels'];hold=ds['outer_holdout'];cal=~hold;val=split_mask(ids,'inner')&cal;train=cal&~val
    s=np.load(semantic_dest/'semantic.rankings.npz')['semantic'];l=np.load(Path(lexical_path) if lexical_path is not None else semantic_dest/'exact_bm25.rankings.npz')['lexical_dpe']
    if len(l)==1:l=np.repeat(l,C['repeats'],axis=0)
    assert len(s)==len(l)==C['repeats']
    docids=read_ids(ds['semantic_source']/'doc_ids.txt');docmap={v:i for i,v in enumerate(docids)}
    trace=json.loads((SRC/cfg['planner_trace']).read_text())[str(cfg['posting_budget'])]
    # Train/select on repeat zero only; repeat-specific features are recomputed for evaluation.
    features=query_features(texts,s[0:1],l[0,:,:100],trace)
    def model(leaves):return HistGradientBoostingClassifier(max_iter=f['max_iter'],learning_rate=f['learning_rate'],max_leaf_nodes=leaves,min_samples_leaf=f['min_samples_leaf'],l2_regularization=f['l2_regularization'],random_state=f['model_seed'])
    x,y,w=training_matrix(s[0],l[0],features,train,ids,truth,docmap);candidates=[]
    for leaves in f['max_leaf_nodes_grid']:
        m=model(leaves);m.fit(x,y,sample_weight=w);learned=learned_rankings(m,s[0],l[0],features,mask=val)
        for weight in f['semantic_backoff_grid']:
            ranking=learned if weight==0 else mix_rankings(s[0],learned,weight)
            score=float(ndcg_rows(ranking,docmap,ids,truth)[val].mean());candidates.append((score,leaves,weight))
        print(dataset,'fusion inner leaves',leaves,'done',flush=True)
    bestscore,leaves,backoff=max(candidates);del x,y,w
    x,y,w=training_matrix(s[0],l[0],features,cal,ids,truth,docmap);m=model(leaves);m.fit(x,y,sample_weight=w)
    joblib.dump(m,dest/'duetrank.joblib');training_rows=len(x);del x,y,w
    # Fixed RRF selected using calibration queries of repeat zero only.
    fixed_scores={weight:float(ndcg_rows(fuse_all(s[0],l[0,:,:100],100,weight),docmap,ids,truth)[cal].mean()) for weight in f['fixed_rrf_semantic_weight_grid']}
    fixed_weight=max(fixed_scores,key=fixed_scores.get)
    values={key:[] for key in ('semantic','lexical','duetrank','fixed_rrf','standard_rrf')};fused=[];full={key:[] for key in ('semantic','lexical')}
    for repeat in range(C['repeats']):
        features=query_features(texts,s[repeat:repeat+1],l[repeat,:,:100],trace)
        learned=learned_rankings(m,s[repeat],l[repeat],features,mask=np.ones(len(ids),dtype=bool))
        mixed=learned if backoff==0 else mix_rankings(s[repeat],learned,backoff);fused.append(mixed)
        ranks={'semantic':s[repeat],'lexical':l[repeat],'duetrank':mixed,
               'fixed_rrf':fuse_all(s[repeat],l[repeat,:,:100],100,fixed_weight),'standard_rrf':fuse_all(s[repeat],l[repeat,:,:100],100,.5)}
        for key,arr in ranks.items():values[key].append(ndcg_rows(arr,docmap,ids,truth))
        for key in full:full[key].append(evaluate(ranks[key],docids,ids,truth))
    values={k:np.stack(v) for k,v in values.items()}
    np.savez_compressed(dest/'holdout_metrics.npz',**values,outer_holdout=hold,query_ids=np.asarray(ids))
    np.savez_compressed(dest/'duetrank.rankings.npz',dual=np.stack(fused),outer_holdout=hold,query_ids=np.asarray(ids))
    report={'dataset':dataset,'split':cfg['split'],'query_count':len(ids),'repeats':C['repeats'],'config_sha256':sha(CONFIG_PATH),
      'calibration_queries':int(cal.sum()),'outer_holdout_queries':int(hold.sum()),'inner_train_queries':int(train.sum()),'inner_validation_queries':int(val.sum()),
      'model_policy':'Select hyperparameters/backoff and fixed RRF on repeat-zero calibration only; fit once on full repeat-zero calibration; freeze for all three outer-holdout repeats. Saved predictions cover all queries for protocol consistency; reported fusion quality uses only outer holdout.',
      'lexical_input_path':str(Path(lexical_path) if lexical_path is not None else semantic_dest/'exact_bm25.rankings.npz'),
      'selection':{'max_leaf_nodes':leaves,'semantic_backoff_weight':backoff,'inner_validation_ndcg':bestscore,'fixed_rrf_semantic_weight':fixed_weight,'training_rows':training_rows,'model_path':str(dest/'duetrank.joblib')},
      'held_out':{key:{'mean_ndcg_at_10':float(v[:,hold].mean()),'per_repeat_ndcg_at_10':v[:,hold].mean(axis=1).tolist()} for key,v in values.items()},
      'duetrank_vs_semantic':bootstrap(values['semantic'],values['duetrank'],hold),'duetrank_vs_fixed_rrf':bootstrap(values['fixed_rrf'],values['duetrank'],hold),
      'full_query_path_metrics':full,'elapsed_seconds':time.perf_counter()-start,'bootstrap_scope':'paired queries, differences first averaged over repeats; conditional on one trained calibration model'}
    write_json(out,report);print(dataset,'FUSION COMPLETE',json.dumps(report['held_out']),flush=True)
    return out

def manifest():
    imported={Path(getattr(m,'__file__',None) or '').name for m in list(sys.modules.values())}
    files=[p for p in SRC.glob('*.py') if p.name in imported]
    report={'run_id':C['run_id'],'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,
      'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0),
      'threads':threadpool_info(),'config_sha256':sha(CONFIG_PATH),'pipeline_sha256':sha(__file__),
      'historical_source_sha256':{str(p):sha(p) for p in files},'query_seed_policy':C['query_seed_policy'],
      'read_only_reuse':'document indexes/embeddings, query embeddings, deterministic raw planner candidate inputs, corpus statistics; no prior post-refinement or fused rankings used as new outputs'}
    write_json(HERE/'quality_manifest.json',report)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['semantic','lexical','bm25','fusion','manifest'],required=True);parser.add_argument('--dataset',choices=list(C['datasets']));parser.add_argument('--query-limit',type=int)
    args=parser.parse_args()
    with threadpool_limits(limits=C['runtime']['blas_threads']):
        if args.stage=='manifest':manifest()
        else:globals()['run_'+args.stage](args.dataset,**({'query_limit':args.query_limit} if args.stage in ('semantic','lexical') else {}))

if __name__=='__main__':main()
