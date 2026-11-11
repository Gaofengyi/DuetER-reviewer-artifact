"""New actual TCP Search+Fetch component experiment; no historical timing reuse."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, socket, subprocess, sys, time, traceback
from contextlib import ExitStack
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'experiment_pprag_precision_network_20261108'
MODES=('full','late_route','plain_semantic','plain_lexical','planner_quota_lexical','no_native','fixed_rrf')
PROFILE={'id':'symmetric100_rtt20','upload_mbps':100,'download_mbps':100,'mbps':100,'rtt_ms':20}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    paths=[ROOT/n for n in ('EXPERIMENT_PLAN.txt','FUSION_POOL_AMENDMENT.json','components_client.py','server_adapter.py','run_components.py','validate_components.py','build_plain_index.py','proxy_entry.py')]
    paths += [OLD/n for n in ('client_adapter.py','client_runtime_binding.py','runtime_binding.json','network_proxy.py','benchmark_suite.py','common.py','noise_schedule.py')]
    legacy=ROOT.parent/'experiment_projectlift_protocol_20261107'/'protocol'
    paths += [legacy/n for n in ('client.py','records.py','wire.py','bindings.py','config.json','selection.py')]
    paths += [ROOT.parent/'experiment_revision_20261107'/'unified_pipeline.py',
              ROOT.parent/'experiment_project_then_lift_20261107'/'project_then_lift_geometry.py']
    paths += [ROOT.parent/'duetdpe_experiments'/n for n in
              ('benchmark_million_lexical_dpe.py','benchmark_budgeted_lexical_candidates.py',
               'experiment_candidate_rank_fusion.py','experiment_adaptive_fusion.py')]
    return {str(p.resolve()):sha(p) for p in paths}
def schedule(phase,query_count):
    if phase=='functional':
        for m in MODES:yield 'functional',0,m,[0,query_count-1]
    else:
        for label,passes in (('warmup',2),('measurement',3)):
            for repeat in range(passes):
                # Fixed, outcome-independent rotation. Three repeats do not
                # completely balance all seven positions; no such claim is made.
                shift=(repeat*3)%len(MODES)
                order=MODES[shift:]+MODES[:shift]
                q=list(range(query_count));q=q[repeat:]+q[:repeat]
                for m in order:yield label,repeat,m,q
def await_ready(process,path,timeout=300):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if process.poll() is not None:raise RuntimeError('Child exited before ready: '+path.name)
        if time.monotonic()>deadline:raise TimeoutError('Child readiness: '+path.name)
        time.sleep(.15)
    return read(path)

def launch_before_queries(argv,ready,log,registry,pids,mode,*,timeout=60,max_attempts=3):
    """Bounded ordinary startup retries, before any query socket is opened.

    Never resume a suspended process or retry a measured query. Every attempt
    has its own readiness artifact; stdout is retained in the common mode log.
    The active child is registered immediately so outer cleanup also covers an
    interruption during startup. Successful readiness is copied to the
    established canonical path without changing its metadata.
    """
    if ready.exists():raise FileExistsError('Existing canonical readiness: '+str(ready))
    history_path=ready.with_name(ready.stem+'_startup_history.json')
    history={'schema':'component-prequery-startup-v1','queries_started':False,
             'retry_scope':'Process readiness only; no scientific call or warmup has begun.',
             'max_attempts':max_attempts,'timeout_seconds':timeout,'attempts':[]}
    creation=getattr(subprocess,'CREATE_NO_WINDOW',0)
    for attempt in range(1,max_attempts+1):
        actual_ready=ready.with_name(ready.stem+f'_attempt{attempt}.json')
        if actual_ready.exists():raise FileExistsError('Existing attempt readiness: '+str(actual_ready))
        command=list(argv);command[command.index('--ready')+1]=str(actual_ready)
        item={'attempt':attempt,'argv':command,'ready_path':str(actual_ready),
              'started_unix_seconds':time.time(),'status':'starting'}
        history['attempts'].append(item);write(history_path,history)
        log.write(json.dumps({'startup_attempt':attempt,'canonical_ready':str(ready)})+'\n');log.flush()
        process=subprocess.Popen(command,stdout=log,stderr=log,cwd=ROOT,creationflags=creation)
        registry[mode]=process;pids.add(process.pid);item['pid']=process.pid;write(history_path,history)
        try:
            info=await_ready(process,actual_ready,timeout=timeout)
            if process.poll() is not None:raise RuntimeError('Child exited immediately after readiness')
        except Exception as error:
            item.update(status='failed_before_queries',error_type=type(error).__name__,error=str(error))
            write(history_path,history)
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            item.update(exit_code=process.returncode,ended_unix_seconds=time.time())
            pids.discard(process.pid);registry.pop(mode,None);write(history_path,history)
            if attempt==max_attempts:raise RuntimeError('Child readiness failed after bounded startup attempts: '+ready.name) from error
            continue
        item.update(status='ready',ready_sha256=sha(actual_ready),ended_unix_seconds=time.time())
        history.update(status='ready',selected_attempt=attempt,canonical_ready_path=str(ready))
        write(ready,info);write(history_path,history)
        return process,info
    raise AssertionError('Unreachable startup exit')

def run_dataset(dataset,destination,phase):
    # This is a freshly launched interpreter; pinned preload happens before
    # any numerical import, even in functional runs.
    sys.path.insert(0,str(ROOT))
    import components_client as cc
    from network_proxy import proxy_control
    spec=importlib.util.spec_from_file_location('_components_legacy_runner',OLD/'benchmark_suite.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    config=read(ROOT.parent/'experiment_projectlift_protocol_20261107'/'protocol/config.json')
    monitor_root=ROOT.parent/'experiment_pprag_e2e_20261107'/'.runtime_deps'
    sys.path.append(str(monitor_root))
    import psutil
    if monitor_root.resolve() not in Path(psutil.__file__).resolve().parents:raise ValueError('Unexpected telemetry provider')
    private=read(ROOT/'.owner'/dataset/'client_private.json')
    queries=private['queries']
    if len(queries)!=16 or len({str(q['id']) for q in queries})!=16:raise ValueError('Frozen query membership')
    source_start=sources()
    owners={}
    for mode in MODES:
        owner=cc.QueryOwnerAdapter(config,dataset,private,mode)
        if not owners:owner.load_encoder()
        else:
            first=next(iter(owners.values()));owner.encoder=first.encoder
            owner.encoder_parameter_dtype=first.encoder_parameter_dtype
        owner.load_selected_ranker();owner.enforce_threads();owners[mode]=owner
        if phase=='functional':
            spec_audit=importlib.util.spec_from_file_location('_components_numeric_audit',ROOT/'validate_components.py')
            audit=importlib.util.module_from_spec(spec_audit);spec_audit.loader.exec_module(audit)
            audit.install_numeric_audit(owner)
    runtime=cc.verify_loaded_runtime(next(iter(owners.values())).encoder)
    runtime['telemetry_provider']={'version':psutil.__version__,'file':str(Path(psutil.__file__).resolve()),
        'sha256':sha(psutil.__file__)}
    write(destination/'client_setup.json',{'runtime':runtime,'query_ids':[str(q['id']) for q in queries],
        'query_indices':[q['index'] for q in queries],'modes':list(MODES),'source_sha256':source_start,
        'owner_private_sha256':sha(ROOT/'.owner'/dataset/'client_private.json'),
        'ranker_bindings':{m:o.selected_ranker_binding for m,o in owners.items()},
        'semantic_budget_policy':'owner-only common min; rank-major stable unique prefix before native scoring'})
    write(destination/'metadata.json',{'dataset':dataset,'profile':PROFILE,
        'private':{'path':str(ROOT/'.owner'/dataset/'client_private.json'),'sha256':sha(ROOT/'.owner'/dataset/'client_private.json')},
        'runtime_binding':runtime,'source_sha256':source_start,
        'config':{'path':str(ROOT.parent/'experiment_projectlift_protocol_20261107'/'protocol/config.json'),
                  'sha256':sha(ROOT.parent/'experiment_projectlift_protocol_20261107'/'protocol/config.json')}})
    pids={os.getpid()};telemetry=legacy.Telemetry(pids);telemetry.start()
    servers={};proxies={};server_ready={};proxy_ready={};allobs=[];batches=[];success=False
    try:
        with ExitStack() as stack:
            for mode in MODES:
                log=stack.enter_context((destination/f'server_{mode}.log').open('w',encoding='utf-8'))
                ready=destination/f'server_{mode}_ready.json'
                command=[sys.executable,'-I','-B','-u',str(ROOT/'server_adapter.py'),
                    '--dataset',dataset,'--mode',mode,'--ready',str(ready),
                    '--log',str(destination/f'server_{mode}_requests.json')]
                servers[mode],server_ready[mode]=launch_before_queries(command,ready,log,servers,pids,mode)
                log=stack.enter_context((destination/f'proxy_{mode}.log').open('w',encoding='utf-8'))
                ready=destination/f'proxy_{mode}_ready.json'
                command=[sys.executable,'-I','-B','-u',str(ROOT/'proxy_entry.py'),
                    '--upstream-host',server_ready[mode]['host'],'--upstream-port',str(server_ready[mode]['port']),
                    '--ready',str(ready),'--log',str(destination/f'proxy_{mode}_frames.json')]
                proxies[mode],proxy_ready[mode]=launch_before_queries(command,ready,log,proxies,pids,mode)
                proxy_control(proxy_ready[mode],'configure',profile=PROFILE)
            write(destination/'topology.json',{'servers':server_ready,'proxies':proxy_ready,'profile':PROFILE,
                'scope':'Seven mode-isolated processes, one active mode at a time, one CPU2 client and shared OS cache; not seven concurrent workload streams.'})
            with ExitStack() as connections:
                sockets={}
                for mode,ready in proxy_ready.items():
                    sock=connections.enter_context(socket.create_connection((ready['host'],ready['port']),timeout=600))
                    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);sockets[mode]=sock
                for label,repeat,mode,qindices in schedule(phase,len(queries)):
                    batch_id=f'{label}-r{repeat}-{mode}'
                    before=proxy_control(proxy_ready[mode],'stats',drain=True)
                    observed=[];begin=time.perf_counter_ns()
                    try:
                        for qi in qindices:
                            query=queries[qi];rid=f'{dataset}-{batch_id}-q{qi}'
                            obs=cc.measured_query(owners[mode],sockets[mode],query,rid)
                            obs.update(phase=label,repeat=repeat,mode=mode,dataset=dataset,query_ordinal=qi)
                            observed.append(obs)
                    except Exception:
                        write(destination/'failed_partial_batch.json',{'batch_id':batch_id,'observations':observed,
                            'error':traceback.format_exc()});raise
                    end=time.perf_counter_ns()
                    after=proxy_control(proxy_ready[mode],'stats',drain=True)
                    reconcile=legacy.reconcile_proxy_batch(observed,before,after)
                    wall=(end-begin)/1e9
                    batch={'phase':label,'repeat':repeat,'mode':mode,'dataset':dataset,'batch_id':batch_id,
                        'batch_wall_seconds':wall,'queries_per_second':len(observed)/wall,
                        'proxy_reconciliation':reconcile,'observations':observed}
                    write(destination/'batches'/f'{batch_id}.json',batch)
                    allobs.extend(observed);batches.append({k:v for k,v in batch.items() if k!='observations'})
                    print(json.dumps({'event':'batch_complete','dataset':dataset,'phase':label,'repeat':repeat,
                        'mode':mode,'queries':len(observed),'batch_wall_seconds':round(wall,3)}),flush=True)
            for mode,process in proxies.items():
                proxy_control(proxy_ready[mode],'shutdown');process.wait(timeout=30)
                if process.returncode:raise RuntimeError('Proxy failed')
            for mode,process in servers.items():
                legacy.shutdown_server(server_ready[mode]);process.wait(timeout=30)
                if process.returncode:raise RuntimeError('Server failed')
            success=True
    finally:
        for mode,process in proxies.items():
            if process.poll() is None:
                try:proxy_control(proxy_ready[mode],'shutdown');process.wait(timeout=10)
                except Exception:process.terminate();process.wait(timeout=10)
        for mode,process in servers.items():
            if process.poll() is None:
                try:legacy.shutdown_server(server_ready[mode]);process.wait(timeout=10)
                except Exception:process.terminate();process.wait(timeout=10)
        telemetry.stop();write(destination/'telemetry.json',{'samples':telemetry.samples,'boundaries':telemetry.background_snapshots})
        if not success:write(destination/'failure.json',{'status':'failed','source_sha256':source_start,'error':traceback.format_exc()})
    reconciliation={m:legacy.reconcile([o for o in allobs if o['mode']==m],
        read(destination/f'server_{m}_requests.json')) for m in MODES}
    for mode in MODES:
        report=read(destination/f'proxy_{mode}_frames.json')
        if report['status']!='complete':raise ValueError('Incomplete proxy log')
    if sources()!=source_start:raise ValueError('Source changed during execution')
    write(destination/'summary.json',{'status':'complete','dataset':dataset,'phase':phase,'source_sha256':source_start,
        'query_units':16 if phase=='formal' else 2,'batch_count':len(batches),'batches':batches,
        'client_server_reconciliation':reconciliation,'observations':len(allobs),
        'formal_measurements':sum(o['phase']=='measurement' for o in allobs)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('functional','formal'),required=True)
    ap.add_argument('--dataset',choices=('nq','hotpotqa','msmarco'),required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--acceptance',type=Path)
    args=ap.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe tag')
    if args.phase=='formal':
        if args.acceptance is None:raise ValueError('Functional acceptance required')
        gate=read(args.acceptance)
        if gate.get('accepted_for_timing') is not True or args.dataset not in gate.get('datasets',[]):raise ValueError('Dataset not accepted')
        if gate.get('source_sha256')!=sources():raise ValueError('Stale functional gate')
    destination=ROOT/'runs'/args.tag/args.dataset
    if destination.exists():raise FileExistsError('Preserve previous run; use a new tag')
    destination.mkdir(parents=True)
    write(destination/'run_plan.json',{'phase':args.phase,'dataset':args.dataset,'modes':MODES,'profile':PROFILE,
        'source_sha256':sources(),'schedule':list(schedule(args.phase,16)),
        'acceptance':None if args.acceptance is None else {'path':str(args.acceptance),'sha256':sha(args.acceptance)}})
    run_dataset(args.dataset,destination,args.phase)
    runroot=destination.parent
    completed=[p.name for p in runroot.iterdir() if p.is_dir() and (p/'summary.json').exists()]
    write(runroot/'suite.json',{'status':'complete','functional':args.phase=='functional',
        'datasets':sorted(completed),'source_sha256':sources(),'profile':PROFILE,
        'acceptance':None if args.acceptance is None else {'path':str(args.acceptance),'sha256':sha(args.acceptance)}})
if __name__=='__main__':main()
