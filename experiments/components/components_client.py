"""Actual authenticated client for one-component-at-a-time budgeted ablations."""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'experiment_pprag_precision_network_20261108'
sys.dont_write_bytecode=True
sys.path.insert(0,str(PREVIOUS))
from client_runtime_binding import preload_client_runtime, verify_loaded_runtime
RUNTIME_PRELOAD=preload_client_runtime()
spec=importlib.util.spec_from_file_location('_components_frozen_fp16_client',PREVIOUS/'client_adapter.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
parent,np=base.parent,base.np
MODES=('full','late_route','plain_semantic','plain_lexical','planner_quota_lexical','no_native','fixed_rrf')

def rank_major_unique(groups):
    seen=set();out=[]
    for rank in range(max(map(len,groups),default=0)):
        for group in groups:
            if rank<len(group):
                row=int(group[rank])
                if row>=0 and row not in seen:seen.add(row);out.append(row)
    return out

class QueryOwnerAdapter(base.QueryOwnerAdapter):
    def __init__(self,config,dataset,private,mode):
        if mode not in MODES:raise ValueError('Unknown component mode')
        super().__init__(config,dataset,private,'dueter_projectlift')
        self.component_mode=mode
        self.fixed_rrf_weight={'nq':1.0,'hotpotqa':.9,'msmarco':.9}[dataset]

    def encode_text(self,text):
        embedding=super().encode_text(text)
        self._latest_embedding=embedding
        return embedding

    def request(self,text,query_id,embedding,request_id,lexical_mode=None,semantic_enabled=True):
        if not semantic_enabled:raise ValueError('Both paths required')
        started=time.perf_counter_ns();query_id=str(query_id)
        terms=list(dict.fromkeys(parent.tokenize(text)))
        tokens={term:parent.keyed_term(term) for term in terms}
        frequencies={token:int(self.private['document_frequencies'][token]) for token in tokens.values()
            if self.private['document_frequencies'].get(token,0)>0}
        selected=sorted(parent.choose_terms(frequencies,self.cfg['posting_budget'],self.cfg['documents']))
        term_slots=self.private['term_slots']
        if len(selected)>term_slots:raise ValueError('Too many lexical labels')
        padding=self.private['dummy_term_labels'][:term_slots-len(selected)]
        labels=b''.join(bytes.fromhex(self.wire_label(token)) for token in selected+padding)
        probes=self.config['semantic']['probes']
        residual=embedding-self.mean;residual/=max(float(np.linalg.norm(residual)),1e-12)
        route_scores=self.centroids@residual
        cells=np.argpartition(-route_scores,probes-1)[:probes]
        if self.component_mode=='plain_semantic':
            sembytes=np.asarray(embedding,dtype='<f4').tobytes()
        else:
            sem_base=self.pipeline.semantic_global_queries(embedding[None,:],[query_id],
                self.semantic_source/'semantic_dpe_key.npz',repeat=0,config=self.config)[0]
            view_cells=np.arange(len(self.labels)) if self.component_mode=='late_route' else cells
            semantic=self.pipeline.semantic_query_views(sem_base,view_cells,config=self.config)
            sembytes=np.asarray(semantic,dtype='<f2').tobytes()
        if self.component_mode=='planner_quota_lexical':lexbytes=b''
        else:
            vector=self.pipeline.query_mips(text,frequencies,self.cfg['documents'],self.config['lexical']['hash_dimension'])
            if self.component_mode=='plain_lexical':lexbytes=np.asarray(vector[:1024],dtype='<f4').tobytes()
            else:
                lexical=self.geometry.query_views(vector[None],[query_id],repeat=0,
                    metadata=self.geometry_metadata,noise=True,quantize=True)[0]
                lexbytes=np.asarray(lexical,dtype='<f2').tobytes()
        blocks=[b''.join(bytes(self.labels[int(c)]) for c in cells),sembytes,labels,lexbytes]
        meta={'kind':'search','dataset':self.dataset,'request_id':request_id,'term_slots':term_slots,
              'semantic_enabled':True,'semantic_method':'dueter','lexical_mode':'projectlift',
              'component_mode':self.component_mode}
        trace={'query_id':query_id,'component_mode':self.component_mode,'selected_terms':len(selected),
            'estimated_postings':sum(frequencies[t] for t in selected),'semantic_wire_coordinate_bytes':len(sembytes),
            'lexical_wire_coordinate_bytes':len(lexbytes),'semantic_query_token_sha256':hashlib.sha256(sembytes).hexdigest(),
            'lexical_query_token_sha256':hashlib.sha256(lexbytes).hexdigest(),
            'request_blocks_sha256':[hashlib.sha256(b).hexdigest() for b in blocks],
            'request_build_ms':(time.perf_counter_ns()-started)/1e6}
        return meta,blocks,trace

    def rerank(self,text,embedding,response_meta,records,trace,backoff=None,lexical_mode=None,
               semantic_enabled=True,query_id=None):
        started=time.perf_counter_ns();slots=response_meta['slots']
        if len(slots)!=len(records):raise ValueError('Record cardinality')
        semantic,lexical,locators={},{},{};groups=[[] for _ in range(self.config['semantic']['probes'])]
        dummy=sem_received=0
        for (path,group,ordinal),record in zip(slots,records):
            row,locator,version,payload=self.owner.open_record(record)
            if version==0:dummy+=1;continue
            if version!=1:raise ValueError('Unknown authenticated version')
            if row in locators and locators[row]!=locator:raise ValueError('Join locator mismatch')
            locators[row]=locator
            if path==1:
                if len(payload)!=768:raise ValueError('Semantic return precision mismatch')
                vector=np.frombuffer(payload,dtype='<f2').astype(np.float32)
                if not np.isfinite(vector).all():raise ValueError('Nonfinite native vector')
                semantic.setdefault(row,vector);groups[int(group)].append(int(row));sem_received+=1
            elif path==2:
                if row in lexical:raise ValueError('Duplicate lexical row')
                lexical[row]=(int(ordinal),payload)
            else:raise ValueError('Unknown path')
        recovered=time.perf_counter_ns()
        admission=rank_major_unique(groups)
        qid=str(query_id if query_id is not None else trace['query_id'])
        budget=int(self.private['component_budgets'][qid])
        if not 100<=budget<=len(admission):raise ValueError('Invalid common budget')
        admitted=admission[:budget]
        admitted_done=time.perf_counter_ns()
        if self.component_mode=='no_native':
            sem_rank=np.asarray(admitted[:100],dtype=np.int64);sem_scored=0
        else:
            sem_rows=np.asarray(sorted(admitted),dtype=np.int64)
            sem_scores=np.stack([semantic[int(row)] for row in sem_rows])@embedding
            sem_rank=sem_rows[np.lexsort((sem_rows,-sem_scores))[:100]];sem_scored=len(sem_rows)
        semantic_done=time.perf_counter_ns()
        lex_rows=sorted(lexical,key=lambda row:lexical[row][0]);cap=self.config['lexical']['exact_scoring_cap']
        if len(lex_rows)>cap:raise ValueError('Lexical native budget exceeded')
        if self.component_mode=='no_native':
            lex_rank=np.asarray(lex_rows,dtype=np.int64);lex_scored=0
        else:
            terms=list(dict.fromkeys(parent.tokenize(text)))
            pairs=[(self.private['token_ids'].get(t),self.private['document_frequencies'].get(parent.keyed_term(t),0)) for t in terms]
            pairs=[(t,f) for t,f in pairs if t is not None and f>0]
            tf=np.zeros((len(lex_rows),len(pairs)),dtype=np.float64)
            lengths=np.zeros((len(lex_rows),1),dtype=np.float64)
            for ordinal,row in enumerate(lex_rows):
                tokens=parent.open_lexical_payload(lexical[row][1]);counts=parent.Counter(tokens)
                lengths[ordinal,0]=len(tokens)
                for j,(token,_) in enumerate(pairs):tf[ordinal,j]=counts[token]
            n,average=self.cfg['documents'],self.private['average_document_length']
            dfs=np.asarray([f for _,f in pairs],dtype=np.float64)
            idf=np.log1p((n-dfs+.5)/(dfs+.5))[None,:]
            norm=1.2*(1-.75+.75*lengths/average)
            score=np.sum(idf*tf*2.2/np.maximum(tf+norm,1e-12),axis=1)
            lex_array=np.asarray(lex_rows,dtype=np.int64)
            lex_rank=lex_array[np.lexsort((lex_array,-score))];lex_scored=len(lex_rows)
        lexical_done=time.perf_counter_ns()
        sem_padded=np.full((1,100),-1,dtype=np.int32);sem_padded[0,:len(sem_rank)]=sem_rank
        lex_padded=np.full((1,cap),-1,dtype=np.int32);lex_padded[0,:len(lex_rank)]=lex_rank
        if self.component_mode=='fixed_rrf':
            final=self.mix_rankings(sem_padded,lex_padded,self.fixed_rrf_weight)
        else:
            coefficient=self.fusion_backoff if backoff is None else float(backoff)
            if self.model is None or coefficient is None:raise ValueError('Frozen model missing')
            features=self.query_features([text],sem_padded[None],lex_padded[:,:100],
                {'terms':[trace['selected_terms']],'estimated_postings':[trace['estimated_postings']]})
            learned=self.learned_rankings(self.model,sem_padded,lex_padded,features)
            final=learned if coefficient==0 else self.mix_rankings(sem_padded,learned,coefficient)
        top100=[int(r) for r in final[0,:100] if r>=0];top20=top100[:20]
        if len(top100)!=len(set(top100)) or len(top20)!=20:raise ValueError('Invalid final ranking')
        finished=time.perf_counter_ns()
        metrics={'payload_join_decrypt_dedup_ms':(recovered-started)/1e6,
            'semantic_budget_admission_ms':(admitted_done-recovered)/1e6,
            'semantic_exact_rerank_ms':(semantic_done-admitted_done)/1e6,
            'lexical_payload_exact_bm25_ms':(lexical_done-semantic_done)/1e6,
            'fusion_ms':(finished-lexical_done)/1e6,'client_rerank_total_ms':(finished-started)/1e6,
            'returned_record_open_count':len(records),'semantic_wire_slots':sum(int(s[0])==1 for s in slots),
            'lexical_wire_slots':sum(int(s[0])==2 for s in slots),
            'semantic_received_records':sem_received,'semantic_unique_candidates':len(semantic),
            'semantic_admitted_candidates':budget,'matched_semantic_budget':budget,
            'semantic_discarded_unique_before_scoring':len(semantic)-budget,'semantic_scored_candidates':sem_scored,
            'lexical_survivors':len(lex_rows),'lexical_exact_scored':lex_scored,'dummy_slots':dummy,
            'semantic_admitted_rows':admitted,'semantic_rank100':sem_padded[0].tolist(),
            'lexical_rank':lex_padded[0].tolist(),'lexical_selected_rows':lex_rows,
            'fused_rank100':top100,'semantic_return_element_bits':16,'semantic_return_payload_bytes':768}
        return top20,[locators[r] for r in top20],metrics

def measured_query(owner,sock,query,request_id):
    result=parent.measured_query(owner,sock,query,request_id,owner.fusion_backoff,'projectlift',True)
    fresh=np.asarray(query['fresh_embedding'],dtype=np.float32)
    if not np.array_equal(owner._latest_embedding,fresh):raise ValueError('Fresh runtime query changed since preparation')
    result['component_mode']=owner.component_mode
    return result
