"""Unchanged frozen proxy with startup-only diagnostics, cancelled at ready."""
import faulthandler,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'experiment_pprag_precision_network_20261108'/'network_proxy.py'
spec=importlib.util.spec_from_file_location('_components_frozen_proxy',source)
proxy=importlib.util.module_from_spec(spec);spec.loader.exec_module(proxy)
ready=Path(sys.argv[sys.argv.index('--ready')+1]).resolve()
original=proxy.write
def traced_write(path,value):
    if Path(path).resolve()==ready:
        print(json.dumps({'startup':'writing_ready','monotonic':time.monotonic()}),flush=True)
    original(path,value)
    if Path(path).resolve()==ready:
        faulthandler.cancel_dump_traceback_later()
        print(json.dumps({'startup':'ready','startup_traceback_timer_cancelled':True}),flush=True)
proxy.write=traced_write
faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
print(json.dumps({'startup':'begin','engine':str(source)}),flush=True)
proxy.main()
