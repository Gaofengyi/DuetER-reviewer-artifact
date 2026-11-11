"""Meaningful bounded checks before the full new query/ranking run."""
import json, hashlib, sys, time
from pathlib import Path
import unified_pipeline as u
import numpy as np

out=u.HERE/'quality_prerequisites.json'
if out.exists():raise SystemExit('Existing prerequisite record is immutable')
c=u.C; record={}
# Selected-coordinate transform must equal the explicitly materialized 8192-D
# keyed transform, independent of noise. This checks selector/permutation order.
rng=np.random.default_rng(77); x=rng.normal(size=(4,1025)).astype(np.float32)
x/=np.linalg.norm(x,axis=1,keepdims=True)
coords=np.sort(rng.choice(8192,100,replace=False));s1,s2,perm=u.lexical_key_arrays()
reference=u.dpe_transform(x,s1,s2,perm)[:,coords]
selected=x@u.selected_matrix(coords,x.shape[1]).T
np.testing.assert_allclose(reference,selected,atol=2e-7,rtol=2e-5)
record['selected_projection_max_abs_error']=float(np.max(np.abs(reference-selected)))

# Per-query deterministic noise must be invariant to batch boundaries/order.
plain=x[:2];ids=['audit_query_a','audit_query_b']
v=u.lexical_query_views(plain,ids,0)
for i in range(2):np.testing.assert_array_equal(v[i],u.lexical_query_views(plain[i:i+1],[ids[i]],0)[0])
assert not np.array_equal(v,u.lexical_query_views(plain,ids,1))
noise=np.stack([u.selected_noise(str(i),0,501) for i in range(100)])
assert np.all(np.linalg.norm(noise,axis=1)<=c['noise']['query_radius']*(1+1e-6))
record['lexical_single_query_batch_equal']=True;record['query_noise_radius']=c['noise']['query_radius']

# Planner-order cap is checked with deliberately reversed local scores.
raw=np.arange(1000,dtype=np.int32);cache_rows=raw.copy();assign=np.arange(1000)%16
local=np.zeros((1000,32),dtype=np.float16);local[:,0]=np.arange(1000)/1000
norm=np.zeros(1000,dtype=np.float32);q=np.zeros((16,32),dtype=np.float16);q[:,0]=1
selected,count=u.lexical_select(raw,q,cache_rows,assign,local,norm)
assert count==1000 and np.array_equal(selected,np.arange(300,dtype=np.int32))
record['cap_occurs_before_bm25_and_preserves_planner_order']=True

# Read only small metadata/header information from every required index.
record['datasets']={}
for dataset in c['datasets']:
 ds=u.prepare_dataset(dataset);cfg=ds['cfg'];entry={'queries':len(ds['query_ids']),'outer_holdout':int(ds['outer_holdout'].sum()),'inputs':{}}
 paths=[ds['semantic_source']/p for p in ['corpus_embeddings.npy','semantic_mean.npy','semantic_centroids.npy','semantic_offsets.npy','semantic_postings.npy',cfg['query_embeddings_file'],'semantic_dpe_key.npz']]
 paths += [ds['semantic_cache']/p for p in ['semantic_compartment_fp16.npy','semantic_compartment_norms.npy']]
 paths += [ds['lexical_cache']/p for p in ['lexical_compartment_fp16.npy','lexical_compartment_norms.npy','lexical_compartment_cells.npy','metadata.json']]
 paths += [ds['source']/'candidate_rows.npy',u.SRC/cfg['planner_cache'],u.SRC/cfg['planner_trace']]
 for p in paths:
  assert p.exists(),str(p);s=p.stat();m={'bytes':s.st_size}
  with p.open('rb') as f:m['header_4096_sha256']=hashlib.sha256(f.read(4096)).hexdigest()
  if s.st_size<4*1024*1024:m['full_sha256']=u.sha(p)
  if p.suffix=='.npy':
   arr=np.load(p,mmap_mode='r');m.update(shape=list(arr.shape),dtype=str(arr.dtype))
  entry['inputs'][str(p)]=m
 one=u.semantic_global_queries(ds['queries'][:2],ds['query_ids'][:2],ds['semantic_source']/'semantic_dpe_key.npz',0)
 single=u.semantic_global_queries(ds['queries'][1:2],ds['query_ids'][1:2],ds['semantic_source']/'semantic_dpe_key.npz',0)
 np.testing.assert_array_equal(one[1],single[0]);entry['semantic_single_query_batch_equal']=True
 entry['query_order_sha256']=hashlib.sha256('\n'.join(ds['query_ids']).encode()).hexdigest()
 record['datasets'][dataset]=entry
 print(dataset,'metadata/query-order checks passed',flush=True)
record['config_sha256']=u.sha(u.CONFIG_PATH);record['pipeline_sha256']=u.sha(u.__file__);record['status']='passed'
u.write_json(out,record)
(u.HERE/'config.sha256').write_text(record['config_sha256']+'  config.json\n',encoding='ascii')
u.manifest()
print(json.dumps({'status':'passed','config_sha256':record['config_sha256'],'checks':list(record)},indent=2),flush=True)
