import unified_pipeline as u
import numpy as np
ds=u.prepare_dataset('nq',query_limit=16)
folder=u.HERE/'smoke/nq'
em=np.load(folder/'semantic_emitted.npz');ranks=np.load(folder/'semantic.rankings.npz')['semantic'];lex=np.load(folder/'lexical_candidates.npz')['lexical_dpe']
assert ranks.shape==(3,16,100) and lex.shape==(3,16,300)
local=np.load(ds['semantic_cache']/'semantic_compartment_fp16.npy',mmap_mode='r');norms=np.load(ds['semantic_cache']/'semantic_compartment_norms.npy',mmap_mode='r')
offsets=np.load(ds['semantic_source']/'semantic_offsets.npy',mmap_mode='r');postings=np.load(ds['semantic_source']/'semantic_postings.npy',mmap_mode='r')
g=u.semantic_global_queries(ds['queries'],ds['query_ids'],ds['semantic_source']/'semantic_dpe_key.npz',0)
worst=0.;overlaps=[]
for qi in range(3):
 cells=em['query_cells'][qi,:4];views=u.semantic_query_views(g[qi],cells)
 for ordinal,cell in enumerate(cells):
  a,b=int(offsets[cell]),int(offsets[cell+1]);doc=np.asarray(postings[a:b],dtype=np.int32)
  scores=2.*(np.asarray(local[a:b],dtype=np.float32)@views[ordinal].astype(np.float32))-np.asarray(norms[a:b])
  cpu=doc[u.top_indices(scores,16)];gpu=em['emitted'][0,qi,ordinal];overlaps.append(len(set(map(int,cpu))&set(map(int,gpu)))/16)
  positions={int(d):j for j,d in enumerate(doc)}
  actual_min=min(float(scores[positions[int(d)]]) for d in gpu if d>=0)
  expected_min=float(np.partition(scores,-16)[-16]);gap=expected_min-actual_min;worst=max(worst,gap)
  assert gap<=2e-4,(qi,int(cell),gap)
raw=np.load(u.SRC/ds['cfg']['planner_cache'])['budget_50000'][:16]
for rep in range(3):
 for qi,row in enumerate(lex[rep]):
  valid=row[row>=0];mapping={int(d):i for i,d in enumerate(raw[qi]) if d>=0};positions=[mapping[int(d)] for d in valid]
  assert len(valid)<=300 and len(set(map(int,valid)))==len(valid)
  assert positions==sorted(positions)
emb=np.load(ds['semantic_source']/'corpus_embeddings.npy',mmap_mode='r')
for rep in range(3):
 for qi in range(3):
  selected=np.unique(em['emitted'][rep,qi]);selected=selected[selected>=0]
  exact=np.asarray(emb[selected],dtype=np.float64)@ds['queries'][qi].astype(np.float64)
  expected=selected[np.lexsort((selected,-exact))[:100]]
  assert set(map(int,expected))==set(map(int,ranks[rep,qi]))
report={'status':'passed','semantic_gpu_cpu_cells_checked':12,'gpu_cpu_top16_mean_set_overlap':float(np.mean(overlaps)),
        'maximum_cpu_score_threshold_shortfall':worst,'semantic_rerank_float64_reference_queries':9,
        'lexical_candidate_order_checks':48,'config_sha256':u.sha(u.CONFIG_PATH),'pipeline_sha256':u.sha(u.__file__)}
u.write_json(u.HERE/'quality_smoke_verification.json',report);u.manifest();print(report,flush=True)
