"""Observed joint FP16 quality/cost, paired by query; no synthetic timing."""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import validate_components as audit

ROOT=Path(__file__).resolve().parent
METRICS=('latency_ms','upload_bytes','search_upload_bytes','search_download_bytes',
         'fetch_upload_bytes','fetch_download_bytes','total_bytes','fetched_plaintext_bytes',
         'ndcg_at_10','recall_at_100')

def percentile(values,p):
    values=sorted(values)
    if not values:return None
    position=(len(values)-1)*p;lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

def describe(values):
    return {'mean':statistics.fmean(values),'p50':percentile(values,.5),
            'p95':percentile(values,.95),'min':min(values),'max':max(values),'observations':len(values)}

def values_of(o):
    up=o['actual_application_upload_bytes'];down=o['actual_application_download_bytes']
    return {'latency_ms':o['end_to_end_ms'],'upload_bytes':up,
            'search_upload_bytes':o['search_rpc']['sent_application_bytes'],
            'search_download_bytes':o['search_rpc']['received_application_bytes'],
            'fetch_upload_bytes':o['fetch_rpc']['sent_application_bytes'],
            'fetch_download_bytes':o['fetch_rpc']['received_application_bytes'],
            'total_bytes':up+down,'fetched_plaintext_bytes':o['fetched_plaintext_bytes'],
            **o['joint_quality']}

def paired(baseline,variant,indices):
    audit.require(len(baseline)==len(variant)>0,'Paired vector cardinality')
    b=statistics.fmean(baseline);v=statistics.fmean(variant)
    differences=[];ratios=[];undefined=0
    for ix in indices:
        bm=sum(baseline[i] for i in ix)/len(ix);vm=sum(variant[i] for i in ix)/len(ix)
        differences.append(vm-bm)
        if bm==0:undefined+=1
        else:ratios.append(vm/bm)
    return {'full_mean':b,'variant_mean':v,'difference_from_full':v-b,
            'paired_difference_ci95':[percentile(differences,.025),percentile(differences,.975)],
            'ratio_to_full':v/b if b else None,
            'paired_ratio_ci95':None if undefined else [percentile(ratios,.025),percentile(ratios,.975)],
            'undefined_ratio_resamples':undefined,
            'ratio_note':'Ratio interval withheld if any resample has zero Full mean; differences retain every resample.'}

def summarize(run,bootstrap=10000,seed=20261109,cached_quality=None):
    checked,all_observations=audit.audit_run(run)
    audit.require(not checked['accepted_for_timing'],'Functional timings must not become scientific costs')
    results={}
    for dataset,allobs in all_observations.items():
        rows=[o for o in allobs if o['phase']=='measurement']
        qids=sorted({str(o['query_id']) for o in rows});audit.require(len(qids)==16,'Expected sixteen independent query units')
        rng=random.Random(seed);indices=[tuple(rng.randrange(len(qids)) for _ in qids) for _ in range(bootstrap)]
        modes={};query_values={}
        batches=[b for b in checked['dataset_checks'][dataset]['batches'] if b['phase']=='measurement']
        for mode in audit.MODES:
            items=[o for o in rows if o['mode']==mode]
            audit.require(len(items)==48,'Every formal mode needs all 48 observations')
            byquery={q:[o for o in items if str(o['query_id'])==q] for q in qids}
            audit.require(all(len(x)==3 for x in byquery.values()),'Unequal repeats per query')
            query_values[mode]={key:[statistics.fmean(values_of(o)[key] for o in byquery[q]) for q in qids] for key in METRICS}
            walls=[b['batch_wall_seconds'] for b in batches if b['mode']==mode]
            stage_keys=sorted({k for o in items for k,v in o.items() if k.endswith('_ms') and isinstance(v,(int,float))})
            outer_qids=[q for q in qids if byquery[q][0]['outer_holdout']]
            modes[mode]={'observations':len(items),'query_units':len(qids),
                'metrics':{key:describe([values_of(o)[key] for o in items]) for key in METRICS},
                'observed_queries_per_second':len(items)/sum(walls),'measured_batch_wall_seconds':sum(walls),
                'batch_wall_seconds':walls,'stage_diagnostics_ms':{k:describe([o[k] for o in items]) for k in stage_keys},
                'candidate_accounting':{k:describe([o[k] for o in items]) for k in (
                    'semantic_wire_slots','semantic_unique_candidates','semantic_admitted_candidates',
                    'semantic_discarded_unique_before_scoring','semantic_scored_candidates',
                    'lexical_wire_slots','lexical_survivors','lexical_exact_scored','returned_record_open_count')},
                'outer_subset_descriptive':{'queries':len(outer_qids),
                    'ndcg_at_10':statistics.fmean(values_of(o)['ndcg_at_10'] for q in outer_qids for o in byquery[q]) if outer_qids else None,
                    'recall_at_100':statistics.fmean(values_of(o)['recall_at_100'] for q in outer_qids for o in byquery[q]) if outer_qids else None}}
        comparisons={mode:{key:paired(query_values['full'][key],query_values[mode][key],indices) for key in METRICS}
                     for mode in audit.MODES if mode!='full'}
        results[dataset]={'queries':16,'measurement_repeats':3,'profile':audit.read(Path(run)/dataset/'metadata.json')['profile'],
            'outer_queries':checked['dataset_checks'][dataset]['outer_queries'],
            'modes':modes,'paired_comparisons_against_full':comparisons,
            'query_units':qids,'query_averaged_values':query_values,
            'raw_batch_files':checked['dataset_checks'][dataset]['files']}
    result={'schema':'matched-budget-component-joint-summary-v1','status':'complete',
        'suite':audit.spec(Path(run)/'suite.json'),'source_sha256':checked['source_sha256'],
        'summarizer':audit.spec(__file__),'validator':audit.spec(audit.__file__),
        'bootstrap_samples':bootstrap,'bootstrap_seed':seed,'bootstrap_generator':'Python random.Random MT19937; percentile linear interpolation',
        'statistical_unit':'Query; average three executions before paired resampling. Ratios divide means.',
        'datasets':results,'all_measurements_retained':True,
        'scope':{'new_common_budget_full':True,'cost_precision':'Real returned 384D FP16 vectors, FP32 native scores',
            'joint_quality':'The same live 16-query per-corpus calls and fused Top100; mixed calibration/reused-outer cohort',
            'throughput':'Completed queries divided by actual sequential batch wall time; not reciprocal mean latency or server capacity',
            'p95':'Descriptive pooled 48-execution percentile; not an independent-query tail confidence interval',
            'stage_clocks':'Nested diagnostics; never summed into endpoint latency',
            'network':'Actual TCP streaming proxy; 100Mbps each direction, 20ms configured RTT; application framing only',
            'inference':'Exploratory frozen-model/cohort comparisons; no equivalence or privacy claim'}}
    if cached_quality:
        q=audit.read(cached_quality)
        result['separate_cached_quality']={'source':audit.spec(cached_quality),'reported_status':q.get('status'),
            'datasets':q.get('datasets'),'scope':'Separate 9005 reused-outer cached FP32 quality study; not the same calls/precision as live joint quality and cost.'}
    return result,checked

def markdown(report):
    lines=['# Matched-budget component observations','',
        'New common-budget Full; all seven rows measured anew. Quality below uses the same live FP16 calls as the costs. Paired intervals use 16 query units per corpus.','',
        '| Corpus | Mode | nDCG@10 | Recall@100 | Mean ms | Latency ratio / Full [95% CI] | KiB/query | Observed q/s |',
        '|---|---|---:|---:|---:|---|---:|---:|']
    for dataset,ds in report['datasets'].items():
        for mode,r in ds['modes'].items():
            m=r['metrics'];ratio='1.000 (reference)'
            if mode!='full':
                c=ds['paired_comparisons_against_full'][mode]['latency_ms'];ci=c['paired_ratio_ci95']
                ratio=f"{c['ratio_to_full']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(f"| {dataset} | {mode} | {m['ndcg_at_10']['mean']:.6f} | {m['recall_at_100']['mean']:.6f} | {m['latency_ms']['mean']:.2f} | {ratio} | {m['total_bytes']['mean']/1024:.2f} | {r['observed_queries_per_second']:.3f} |")
    lines+=['','Full includes owner-only B* admission before native scores. late_route scores all cells then applies original routing eligibility; its output is required to equal Full. no_native retains the candidate/crypto budget but intentionally performs zero native scores. Search and Fetch bytes are separately retained in JSON.','',
            'The JSON contains every paired difference/ratio interval for quality, latency and communication, query-averaged values, raw-file bindings and nested stage diagnostics. No slow samples are removed.']
    return '\n'.join(lines)+'\n'

def self_test():
    audit.require(percentile([1,2,3,4],.5)==2.5,'Percentile')
    p=paired([1.,2.],[2.,4.],[(0,1),(0,0),(1,1)])
    audit.require(p['ratio_to_full']==2 and p['paired_ratio_ci95']==[2.,2.],'Paired ratio')
    p=paired([0.,1.],[1.,2.],[(0,0),(0,1)])
    audit.require(p['paired_ratio_ci95'] is None and p['undefined_ratio_resamples']==1,'Zero denominator')
    return {'status':'passed','tests':3,'scope':'Synthetic standard-library checks only'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261109);p.add_argument('--cached-quality',type=Path);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        if args.run is None:p.error('--run required')
        result,checked=summarize(args.run,args.bootstrap,args.seed,args.cached_quality)
        audit.write(args.run/'independent_audit.json',checked);audit.write(args.run/'component_summary.json',result)
        output=args.run/'component_summary.md';output.write_text(markdown(result),encoding='utf-8')
        print(json.dumps({'status':'complete','summary':str(args.run/'component_summary.json'),'table':str(output)}))
