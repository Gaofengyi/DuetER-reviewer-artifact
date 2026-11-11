"""Exact rerank of this run's fresh candidates with sequential embedding load.

Load one corpus at a time (largest dense matrix 13.6 GB) to avoid page-at-a-time
random disk reads. This is an offline quality execution, not online timing.
"""
import argparse,gc,json,time,hashlib
import unified_pipeline as u
import numpy as np
from threadpoolctl import threadpool_limits

def run(dataset):
 ds=u.prepare_dataset(dataset);cfg=ds['cfg'];dest=u.HERE/'results'/cfg['label'];path=dest/'semantic_emitted.npz';final=dest/'semantic.rankings.npz'
 if final.exists():raise FileExistsError(final)
 while not path.exists():time.sleep(2)
 if dataset!='nq':
  while not (dest/'semantic_cloud.json').exists():time.sleep(2)
 archive=np.load(path);emitted=archive['emitted'];assert np.array_equal(archive['query_ids'],np.asarray(ds['query_ids']))
 assert emitted.shape==(u.C['repeats'],len(ds['queries']),u.C['semantic']['probes'],cfg['semantic_depth'])
 emhash=u.sha(path);source=ds['semantic_source'];t=time.perf_counter()
 mapped=np.load(source/'corpus_embeddings.npy',mmap_mode='r')
 shape=mapped.shape;emb=np.empty(shape,dtype=np.float32)
 for first in range(0,len(emb),65536):
  last=min(first+65536,len(emb));emb[first:last]=mapped[first:last]
  if first==0 or last==len(emb) or (first//65536)%16==0:print(dataset,'sequential embeddings',last,'/',len(emb),round(time.perf_counter()-t,1),'s',flush=True)
 del mapped;load_seconds=time.perf_counter()-t
 out=np.full((u.C['repeats'],len(ds['queries']),u.C['semantic']['output_depth']),-1,dtype=np.int32);counts=[];times=[]
 started=time.perf_counter()
 for r in range(u.C['repeats']):
  for qi,q in enumerate(ds['queries']):
   ts=time.perf_counter();ids=np.unique(emitted[r,qi]);ids=ids[ids>=0]
   scores=emb[ids]@q;order=np.lexsort((ids,-scores))[:u.C['semantic']['output_depth']]
   out[r,qi,:len(order)]=ids[order];counts.append(len(ids));times.append((time.perf_counter()-ts)*1000)
   if (qi+1)%1000==0:print(dataset,'rerank',r,qi+1,'/',len(ds['queries']),round(time.perf_counter()-started,1),'s',flush=True)
 del emb;gc.collect()
 np.savez_compressed(final,semantic=out,query_ids=np.asarray(ds['query_ids']),outer_holdout=ds['outer_holdout'])
 cloud=json.loads((dest/'semantic_cloud.json').read_text()) if (dest/'semantic_cloud.json').exists() else {'cloud_batch_seconds':16.3,'source':'semantic_run.log rounded stdout; separated original NQ stage did not persist exact float before interruption'}
 result={'dataset':dataset,'query_count':len(ds['queries']),'repeats':u.C['repeats'],'local_depth':cfg['semantic_depth'],
  'query_radius':u.C['noise']['query_radius'],'coordinate_dtype':'float16','score_dtype':'float32',
  'cloud':cloud,'rerank_ms':u.stats(times),'candidates':u.stats(counts),'sequential_embedding_load_seconds':load_seconds,
  'rerank_wall_seconds':time.perf_counter()-started,'embedding_bytes_loaded':int(np.prod(shape)*4),
  'candidate_artifact_sha256':emhash,'config_sha256':u.sha(u.CONFIG_PATH),'pipeline_sha256':u.sha(u.__file__),'rerank_script_sha256':u.sha(__file__),
  'not_end_to_end':True,'not_interactive_latency':True,'fresh_candidate_provenance':'this 20261107 run semantic_emitted.npz; no prior semantic rankings',
  'io_scope':'sequentially load entire source embedding matrix to RAM, then exact native cosine on only selected candidates; one corpus at a time; preload counted separately',
  'tie_policy':'descending FP32 native cosine then ascending corpus row'}
 u.write_json(dest/'semantic.json',result);print(dataset,'SEMANTIC QUALITY READY',flush=True)

p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=list(u.C['datasets']));a=p.parse_args()
with threadpool_limits(limits=u.C['runtime']['blas_threads']):
 for dataset in a.datasets:
  assert (u.HERE/'config.sha256').read_text().split()[0]==u.sha(u.CONFIG_PATH)
  run(dataset);gc.collect()
