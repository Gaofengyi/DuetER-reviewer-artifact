import argparse,gc,time
import unified_pipeline as u
from threadpoolctl import threadpool_limits
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['semantic','lexical','bm25','fusion'],required=True);p.add_argument('--datasets',nargs='+',default=list(u.C['datasets']));a=p.parse_args()
frozen=(u.HERE/'config.sha256').read_text().split()[0]
assert frozen==u.sha(u.CONFIG_PATH),'Config changed after freeze'
with threadpool_limits(limits=u.C['runtime']['blas_threads']):
 for dataset in a.datasets:
  assert frozen==u.sha(u.CONFIG_PATH)
  print('START',a.stage,dataset,'UTC',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
  getattr(u,'run_'+a.stage)(dataset)
  gc.collect();u.torch.cuda.empty_cache()
  print('FINISHED',a.stage,dataset,flush=True)
