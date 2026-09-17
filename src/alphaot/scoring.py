"""Continuous two-view affinity over observed exact coordinates."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


def effect(q, activity):
    q, activity = np.asarray(q, float), np.asarray(activity, float)
    if q.shape != activity.shape or not np.isfinite(q).all() or not np.isfinite(activity).all():
        raise ValueError('Invalid paired quantiles')
    if (abs(q)>1).any() or ((activity<0)|(activity>1)).any():
        raise ValueError('Quantiles out of bounds')
    return abs(q)*activity


def support(si, sj):
    return float(2*si*sj/(si+sj)) if si+sj else 0.


def rms(values):
    values=list(values)
    return math.sqrt(math.fsum(float(v)**2 for v in values)/len(values)) if values else 0.


def cosine(x, y):
    denominator = math.sqrt(math.fsum(float(v)**2 for v in x))*math.sqrt(math.fsum(float(v)**2 for v in y))
    return float(np.clip(math.fsum(float(a)*float(b) for a,b in zip(x,y))/denominator,-1,1)) if denominator else 0.


def audit_catalog(catalog,contexts,cfg):
    c=catalog.copy().fillna('')
    c['context']=c.ontology_curie
    for key in contexts:
        if key.startswith('GTEx:'):
            mask=c.data_source.eq('gtex') & c.gtex_tissue.eq(key.split(':',1)[1])
            c.loc[mask,'context']=key
    def reason(row):
        if row.context not in contexts:return 'context_not_selected'
        if str(row.genetically_modified).lower() not in ['false','0']:return 'genetically_modified_or_unknown'
        if not row.context.startswith('GTEx:') and row.biosample_name!=contexts[row.context]['name']:
            return 'biosample_name_mismatch'
        expected={'CHIP_TF':f'{row.ontology_curie} TF ChIP-seq {row.transcription_factor}',
                  'ATAC':f'{row.ontology_curie} ATAC-seq','DNASE':f'{row.ontology_curie} DNase-seq'}
        if row.modality in expected and row['name']!=expected[row.modality]:return 'unreviewed_track_name'
        titles={'CHIP_TF':['TF ChIP-seq'],'ATAC':['ATAC-seq'],'DNASE':['DNase-seq'],
                'RNA_SEQ':['polyA plus RNA-seq','total RNA-seq']}
        if row['Assay title'] not in titles.get(row.modality,[]):return 'unreviewed_assay'
        return 'included'
    c['review_reason']=c.apply(reason,axis=1)
    c['included']=c.review_reason.eq('included')
    c['tier']=c.context.map(lambda x:contexts.get(x,{}).get('tier',''))
    if c.track_id.duplicated().any():raise ValueError('Duplicate track IDs')
    return c

def prepare_scores(scores, audit, cfg):
    d=scores.copy().fillna({'gene_id':'','gene_name':''})
    if d.duplicated(['variant_id','modality','track_id','gene_id']).any():
        raise ValueError('Duplicate exact score coordinates')
    if not d.track_id.isin(audit.track_id).all():raise ValueError('Unknown track')
    if not d.modality.eq(d.track_id.map(audit.set_index('track_id').modality)).all():
        raise ValueError('Track modality mismatch')
    if not np.isfinite(d[['raw','q','active_raw','active_q']].to_numpy(float)).all():
        raise ValueError('Nonfinite paired score')
    d['d']=d.q
    d['a']=d.active_q
    d['r']=effect(d.d,d.a)
    d=d.merge(audit.drop(columns='modality'),on='track_id',validate='many_to_one')
    d=d[d.included].copy()
    if (d.modality.eq('RNA_SEQ') & d.gene_id.eq('')).any():raise ValueError('RNA needs gene ID')
    d['coordinate']=np.where(d.modality.eq('RNA_SEQ'),d.gene_id+'|'+d.track_id,d.track_id)
    d['diff_scorer']='DIFF_LOG2_SUM'
    d['active_scorer']='ACTIVE_SUM'
    return d


@dataclass
class Profile:
    tf: pd.DataFrame
    rna: pd.DataFrame
    access: pd.DataFrame
    pathway: pd.DataFrame
    rna_strength: float = 0.


def build_profiles(rows, mapping, ids):
    if mapping.duplicated(['gene_id','pathway_id']).any():raise ValueError('Duplicate pathway membership')
    sizes=mapping.groupby('pathway_id').gene_id.nunique()
    degree=mapping.groupby('gene_id').pathway_id.nunique()
    mapping=mapping.copy()
    expected=1/np.sqrt(mapping.pathway_id.map(sizes)*mapping.gene_id.map(degree))
    if not np.allclose(mapping.projection_weight,expected):raise ValueError('Pathway weights mismatch')
    profiles={}; members=[]
    for vid in ids:
        d=rows[rows.variant_id.eq(vid)]
        def table(modalities):
            return d[d.modality.isin(modalities)].set_index('coordinate').sort_index()
        rna=table(['RNA_SEQ'])
        joined=rna.reset_index().merge(mapping[['gene_id','pathway_id','projection_weight']],on='gene_id')
        # A pathway/track is observed only when at least one mapped gene was measured.
        # Both direction channels exist for that observed coordinate; no missing gene is filled.
        projected=[]
        for direction, sign in [('up',1),('down',-1)]:
            j=joined.copy()
            j['direction']=direction
            j['diff_contribution']=np.maximum(sign*j.d,0)*j.projection_weight
            j['strength_contribution']=np.where(sign*j.d>0,j.r,0)*j.projection_weight
            members.append(j)
            grouped=j.groupby(['pathway_id','track_id'],sort=True)[['diff_contribution','strength_contribution']].sum()
            for (path,track),values in grouped.iterrows():
                projected.append({'coordinate':f'{path}|{track}|{direction}','d':values.diff_contribution,
                                  'r':values.strength_contribution,'track_id':track})
        path=pd.DataFrame(projected,columns=['coordinate','d','r','track_id']).set_index('coordinate')
        # Equal-weight RNA tracks: first aggregate genes within each track, then
        # aggregate the track-level RMS values. This strength is independent of
        # Reactome projection weights and coordinate multiplicity.
        track_strength=rna.groupby('track_id',sort=True).r.apply(rms) if len(rna) else pd.Series(dtype=float)
        profiles[vid]=Profile(table(['CHIP_TF']),rna,table(['ATAC','DNASE']),path,rms(track_strength))
    return profiles,pd.concat(members,ignore_index=True) if members else pd.DataFrame()


def exact_affinity(x,y,details=False):
    keys=x.index.intersection(y.index).sort_values()
    n=len(keys)
    result={'available':bool(n),'shared_coordinates':n,'coordinates_i':len(x),'coordinates_j':len(y),
            'coverage_i':n/len(x) if len(x) else 0.,'coverage_j':n/len(y) if len(y) else 0.,
            'K_signed':None,'K':None,'H':None,'strength_i':None,'strength_j':None,'affinity':None}
    if not n:return result,{},[]
    a,b=x.loc[keys],y.loc[keys]
    if 'context' in a and not a.context.equals(b.context):raise ValueError('Coordinate metadata mismatch')
    di,dj=a.d.to_numpy(float),b.d.to_numpy(float)
    si=rms(a.r);sj=rms(b.r)
    signed=cosine(di,dj); k=max(signed,0.);h=support(si,sj)
    result.update(K_signed=signed,K=k,H=h,strength_i=si,strength_j=sj,affinity=k*h)
    evidence=[]
    if details:
        for key in keys:
            evidence.append({'coordinate':key,'d_i':float(a.loc[key,'d']),'d_j':float(b.loc[key,'d']),
                             'r_i':float(a.loc[key,'r']),'r_j':float(b.loc[key,'r'])})
    return result,{},evidence


def accessibility_gate(x,y):
    return exact_affinity(x,y)[0]


def compare(x,y):
    tf=exact_affinity(x.tf,y.tf)[0]
    gate=accessibility_gate(x.access,y.access)
    path=exact_affinity(x.pathway,y.pathway)[0]
    diagnostics={'tf':tf,'accessibility':gate,'pathway':path}
    for mod in ['ATAC','DNASE']:
        diagnostics[mod.lower()]=exact_affinity(x.access[x.access.modality.eq(mod)],y.access[y.access.modality.eq(mod)])[0]
    ans={f'{name}_{key}':v for name,diag in diagnostics.items() for key,v in diag.items()}
    regulation_strength=math.sqrt(tf['H']*gate['H']) if tf['available'] and gate['available'] else None
    regulation=tf['K']*gate['K']*regulation_strength if regulation_strength is not None else None
    rna_h=support(x.rna_strength,y.rna_strength)
    pathway_coverage_gate=support(path['coverage_i'],path['coverage_j']) if path['available'] else None
    pathway=path['K']*rna_h*pathway_coverage_gate if path['available'] else None
    ans['pathway_projected_affinity']=path['affinity']
    ans['pathway_affinity']=pathway
    ans.update(regulation_strength=regulation_strength,rna_strength_i=x.rna_strength,
               rna_strength_j=y.rna_strength,rna_H=rna_h,
               pathway_coverage_gate=pathway_coverage_gate)
    views={'regulation':regulation,'pathway':pathway}
    available={k:v for k,v in views.items() if v is not None}
    total=max(available.values()) if available else None
    if total is not None and total>1+1e-10:
        raise ValueError('Projected affinity exceeds one; cannot silently clip the specified formula')
    ans.update({f'affinity_{k}':v for k,v in views.items()})
    ans.update(available=bool(available),regulation_available=regulation is not None,affinity=total,
               cost=1-total if total is not None else None,
               dominant_views='|'.join(k for k,v in available.items() if v==total),
               unavailable_reason='' if available else 'no_shared_regulation_or_pathway_coordinates')
    return ans
