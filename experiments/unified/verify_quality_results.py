"""Final shape, provenance, candidate, split and sampled exact-score checks."""
import json,hashlib,gc
import unified_pipeline as u
import numpy as np

out=u.HERE/'quality_final_verification.json'
if out.exists():raise FileExistsError(out)
frozen=(u.HERE/'config.sha256').read_text().split()[0]
assert frozen==u.sha(u.CONFIG_PATH)
pre=json.loads((u.HERE/'quality_prerequisites.json').read_text())
manifest=json.loads((u.HERE/'quality_manifest.json').read_text())
report={'status':'in_progress','config_sha256':frozen,'datasets':{},'source_checks':{},'cache_checks':{}}
for path,h in manifest['historical_source_sha256'].items():
 actual=u.sha(path);assert actual==h,path;report['source_checks'][path]=True
for name,entry in pre['datasets'].items():
 for path,meta in entry['inputs'].items():
  p=u.Path(path);assert p.stat().st_size==meta['bytes'],path
  with p.open('rb') as f:assert hashlib.sha256(f.read(4096)).hexdigest()==meta['header_4096_sha256'],path
  if 'full_sha256' in meta:assert u.sha(p)==meta['full_sha256'],path
  report['cache_checks'][path]={'size_and_header_unchanged':True,'full_sha256_checked':'full_sha256' in meta}

for dataset,cfg in u.C['datasets'].items():
 ds=u.prepare_dataset(dataset);dest=u.HERE/'results'/cfg['label'];ids=np.asarray(ds['query_ids']);hold=ds['outer_holdout']
 payload=json.loads((u.HERE/'direct_bm25'/dataset/'payload_cache/ready.json').read_text())
 terms,_=u.query_vocabulary(ds['query_texts'])
 digest=lambda value:hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 assert digest(ds['query_ids'])==payload['query_ids_sha256'];assert digest(terms)==payload['vocabulary_sha256']
 paths={'semantic':('semantic.rankings.npz','semantic',100),'candidate':('lexical_candidates.npz','lexical_dpe',300),
        'exact_bm25':('exact_bm25.rankings.npz','lexical_dpe',300),'duetrank':('duetrank.rankings.npz','dual',100)}
 arrays={};artifact={}
 for name,(filename,key,depth) in paths.items():
  p=dest/filename;a=np.load(p);arr=a[key];assert arr.shape==(3,cfg['queries'],depth)
  assert np.array_equal(a['query_ids'],ids) and np.array_equal(a['outer_holdout'],hold)
  assert np.all((arr>=-1)&(arr<cfg['documents']))
  sorted_rows=np.sort(arr,axis=-1)
  assert not np.any((sorted_rows[...,1:]==sorted_rows[...,:-1])&(sorted_rows[...,1:]>=0))
  arrays[name]=arr;artifact[filename]={'shape':list(arr.shape),'sha256':u.sha(p),'valid_candidate_counts':u.stats((arr>=0).sum(axis=-1).reshape(-1))}
 assert np.array_equal(np.sort(arrays['candidate'],axis=-1),np.sort(arrays['exact_bm25'],axis=-1))
 raw=np.load(u.SRC/cfg['planner_cache'])[f"budget_{cfg['posting_budget']}"][:,:1000]
 for qi in range(cfg['queries']):
  lookup={int(d):j for j,d in enumerate(raw[qi]) if d>=0}
  for repeat in range(3):
   row=arrays['candidate'][repeat,qi];pos=[lookup[int(d)] for d in row if d>=0]
   assert pos==sorted(pos)
 fusion=json.loads((dest/'fusion.json').read_text());v=np.load(dest/'holdout_metrics.npz')
 assert int(hold.sum())==cfg['outer_holdout_queries']
 for key,value in fusion['held_out'].items():
  assert np.isclose(np.mean(v[key][:,hold]),value['mean_ndcg_at_10'],atol=1e-13)
 for f in ('semantic.json','lexical.json','bm25.json','fusion.json'):
  assert json.loads((dest/f).read_text())['config_sha256']==frozen
 # Recovered I/O path: check 8 uniformly spaced queries, all repeats, with
 # independently evaluated FP64 native cosine over the saved fresh union.
 # NQ was not recovered; its algorithm already has a 9-query smoke check.
 sample_report={'not_repeated':'NQ used original completed run; 9 FP64 smoke comparisons already passed'}
 if dataset!='nq':
  archive=np.load(dest/'semantic_emitted.npz');emitted=archive['emitted']
  assert np.array_equal(archive['query_ids'],ids)
  e=np.load(ds['semantic_source']/'corpus_embeddings.npy',mmap_mode='r')
  indices=np.linspace(0,cfg['queries']-1,8,dtype=int);overlaps=[];max_gap=0.;exact_order=0
  for repeat in range(3):
   for qi in indices:
    candidates=np.unique(emitted[repeat,qi]);candidates=candidates[candidates>=0]
    score=np.asarray(e[candidates],dtype=np.float64)@np.asarray(ds['queries'][qi],dtype=np.float64)
    reference=candidates[np.lexsort((candidates,-score))[:100]];actual=arrays['semantic'][repeat,qi]
    overlap=len(set(map(int,reference))&set(map(int,actual)))/100;overlaps.append(overlap)
    if np.array_equal(reference,actual):exact_order+=1
    if overlap<1:
     mapping={int(d):float(s) for d,s in zip(candidates,score)}
     kth=float(np.partition(score,-100)[-100]);gap=max(0.,kth-min(mapping[int(d)] for d in actual if d>=0));max_gap=max(max_gap,gap)
     assert gap<1e-6,(dataset,repeat,int(qi),gap)
  sample_report={'queries':indices.tolist(),'query_repeats_checked':24,'mean_top100_set_overlap':float(np.mean(overlaps)),
                'exact_top100_order_equal':exact_order,'maximum_float64_threshold_shortfall':max_gap,'all_members_or_near_boundary_ties_valid':True}
  del emitted,e,archive
 result={'query_count':cfg['queries'],'outer_holdout_queries':int(hold.sum()),'calibration_queries':int((~hold).sum()),'repeats':3,
         'query_order_sha256':hashlib.sha256('\n'.join(ds['query_ids']).encode()).hexdigest(),'artifacts':artifact,
         'all_lexical_candidate_members_preserved_by_exact_scoring':True,'all_lexical_candidates_keep_original_planner_order':True,
         'shared_payload_vocabulary_and_query_ids_match':True,'sampled_fp64_exact_rerank':sample_report,
         'model_sha256':u.sha(dest/'duetrank.joblib'),'all_result_config_hashes_match':True}
 report['datasets'][dataset]=result
 print(dataset,'FINAL CHECKS PASSED',json.dumps(sample_report),flush=True)
 del arrays;gc.collect()
report['status']='passed';report['cache_hash_limit']='Full SHA256 checked when recorded at prerequisite time; large cache files checked by exact size and 4096-byte header fingerprint, not a full content hash.'
report['historical_write_scope']='Execution code writes only under experiment_revision_20261107; historical indexes loaded read-only. Listed source and input fingerprints unchanged; not a hash audit of every original workspace file.'
u.write_json(out,report)
print('FINAL QUALITY VERIFICATION PASSED',flush=True)
