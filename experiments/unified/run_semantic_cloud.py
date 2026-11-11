"""Separated fresh semantic cloud stage, with exactly the tested kernel.

The inspected run_semantic body is identical through emitted-candidate saving.
Its random memmap rerank tail is intentionally replaced by a stage manifest.
Original pipeline/source files are not modified.
"""
import argparse, inspect, hashlib, gc, time
import unified_pipeline as u
from threadpoolctl import threadpool_limits

source=inspect.getsource(u.run_semantic)
boundary='    embeddings=np.load(source/\'corpus_embeddings.npy\',mmap_mode=\'r\')'
assert source.count(boundary)==1
cloud_body=source[:source.index(boundary)].replace('def run_semantic(', 'def run_semantic_cloud(',1)
cloud_body += '''    write_json(dest/'semantic_cloud.json',{'dataset':dataset,'query_count':nq,'repeats':repeats,'local_depth':depth,'query_radius':C['noise']['query_radius'],
      'cloud_batch_seconds':cloud_seconds,'posting_reads':stats(read_counts),'elapsed_seconds':time.perf_counter()-start,
      'config_sha256':sha(CONFIG_PATH),'pipeline_sha256':sha(__file__),'gpu_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
      'coordinate_dtype':'float16','arithmetic':'float32 CUDA mm; TF32 disabled','not_end_to_end':True,
      'fresh_cloud_ranking':True,'query_seed_policy':C['query_seed_policy']})
    print(dataset,'CLOUD COMPLETE',cloud_seconds,flush=True)
    return dest/'semantic_emitted.npz'
'''
ns=dict(u.__dict__);exec(compile(cloud_body,__file__,'exec'),ns)
p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['hotpotqa','msmarco']);a=p.parse_args()
frozen=(u.HERE/'config.sha256').read_text().split()[0]
u.write_json(u.HERE/'semantic_stage_separation.json',{
 'reason':'Random memmap exact-rerank was storage-bound. By the time the verified process was stopped, NQ had completed fully (268.584s) and HotpotQA cloud had completed (38.1s rounded stdout). The interruption is only HotpotQA unfinished rerank. Completed NQ results and fresh HotpotQA emitted candidates are preserved; only MS MARCO cloud remains.',
 'stopped_process_id':28320,'config_sha256':frozen,'original_pipeline_sha256':u.sha(u.__file__),
 'cloud_executed_body_sha256':hashlib.sha256(cloud_body.encode()).hexdigest(),
 'algorithm_change':'none: same fresh query transform, FP32 CUDA scoring, local top-k; only stage separation and offline exact-rerank I/O preparation change',
 'old_final_rankings_used':False})
with threadpool_limits(limits=u.C['runtime']['blas_threads']):
 for dataset in a.datasets:
  assert frozen==u.sha(u.CONFIG_PATH)
  target=u.HERE/'results'/u.C['datasets'][dataset]['label']/'semantic_emitted.npz'
  if target.exists():raise FileExistsError(target)
  print('START CLOUD',dataset,flush=True);ns['run_semantic_cloud'](dataset);gc.collect();u.torch.cuda.empty_cache()
