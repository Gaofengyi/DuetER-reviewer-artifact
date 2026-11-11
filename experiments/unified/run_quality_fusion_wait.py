import time,gc
import unified_pipeline as u
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=u.C['runtime']['blas_threads']):
 for dataset,cfg in u.C['datasets'].items():
  dest=u.HERE/'results'/cfg['label']
  while not (dest/'semantic.json').exists() or not (dest/'bm25.json').exists():time.sleep(2)
  assert (u.HERE/'config.sha256').read_text().split()[0]==u.sha(u.CONFIG_PATH)
  print('START FINAL FUSION',dataset,flush=True);u.run_fusion(dataset);gc.collect()
