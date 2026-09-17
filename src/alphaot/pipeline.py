"""Run one global two-view variant transport problem per cancer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
from .io import config_at, digest, read_candidates, excluded_mass, validate_config
from .scoring import audit_catalog, prepare_scores, build_profiles, compare, exact_affinity
from .transport import unbalanced_transport


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n')


def run_cancer(name,spec,cfg,root,out):
    validate_config(cfg)
    if out.exists() and any(out.iterdir()):raise ValueError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    candidates=read_candidates(root/spec['input'],name)
    cohorts={cfg['source_cohort'],cfg['target_cohort']}
    if not set(candidates.cohort).issubset(cohorts):raise ValueError('Unconfigured cohort in input')
    catalog_path=root/'resources/atlas/track_catalog.parquet'
    score_path=root/spec['scores']
    scores=pd.read_parquet(score_path)
    catalog=pd.read_parquet(catalog_path)
    mapping_path=root/'resources/reactome_mapping.parquet'
    mapping=pd.read_parquet(mapping_path)
    sizes=mapping.groupby('pathway_id').gene_id.nunique()
    degree=mapping.groupby('gene_id').pathway_id.nunique()
    weights=1/np.sqrt(mapping.pathway_id.map(sizes)*mapping.gene_id.map(degree))
    if not np.allclose(mapping.projection_weight,weights):raise ValueError('Pathway size/degree weights mismatch')
    if not sizes.between(*cfg['pathway_size_range']).all():raise ValueError('Mapping outside configured pathway size range')
    fetch_prov_path=root/'resources/atlas'/f'{name}_provenance.json'
    fetch_prov=json.loads(fetch_prov_path.read_text())
    if fetch_prov['data_sha256']!=digest(score_path) or fetch_prov['catalog_sha256']!=digest(catalog_path):
        raise ValueError('Atlas snapshot hash does not match fetch provenance')
    if not fetch_prov['source'].startswith('AlphaGenome Atlas'):raise ValueError('Expected explicitly sourced Atlas scalar scores')
    audit=audit_catalog(catalog,spec['contexts'],cfg)
    audit.to_csv(out/'track_audit.csv',index=False)
    ids=sorted(candidates.variant_id.unique())
    rows=prepare_scores(scores[scores.variant_id.isin(ids)],audit,cfg)
    rows.to_parquet(out/'exact_track_effects.parquet',index=False)
    profiles,members=build_profiles(rows,mapping,ids)
    members.to_csv(out/'pathway_member_contributions.csv',index=False)
    coverage=[]
    for vid in ids:
        d=rows[rows.variant_id.eq(vid)]
        coverage.append({'variant_id':vid,'score_snapshot_present':bool(scores.variant_id.eq(vid).any()),
            **{f'{m}_coordinates':int(d.modality.eq(m).sum()) for m in ['RNA_SEQ','CHIP_TF','ATAC','DNASE']},
            'nonzero_effect_coordinates':int(d.r.ne(0).sum()),'rna_strength':profiles[vid].rna_strength})
    pd.DataFrame(coverage).to_csv(out/'score_coverage.csv',index=False)
    candidates=candidates.merge(pd.DataFrame(coverage),on='variant_id',validate='many_to_one')
    candidates.to_csv(out/'candidate_masses.csv',index=False)
    left=candidates[candidates.cohort.eq(cfg['source_cohort'])]
    right=candidates[candidates.cohort.eq(cfg['target_cohort'])]
    if left.empty or right.empty:raise ValueError('Both configured cohorts need at least one SNV')
    left_ids=sorted(left.variant_id.unique());right_ids=sorted(right.variant_id.unique())
    table=[]
    for i,a in enumerate(left_ids):
        for b in right_ids:
            result=compare(profiles[a],profiles[b])
            ca,pa=a.split(':')[:2];cb,pb=b.split(':')[:2]
            result.update({'cancer':name,'source_variant':a,'target_variant':b,
                'same_variant':a==b,'reused_prediction':a==b,
                'pair_class':'same_variant' if a==b else ('within_1Mb' if ca==cb and abs(int(pa)-int(pb))<=1048576 else 'distant_or_cross_chromosome')})
            table.append(result)
        if (i+1)%25==0 or i+1==len(left_ids):print(f'{name}: affinity rows {i+1}/{len(left_ids)}',flush=True)
    affinity=pd.DataFrame(table)
    affinity.to_csv(out/'variant_affinities.csv',index=False)
    for field in ['affinity','affinity_regulation','affinity_pathway']:
        affinity.pivot(index='source_variant',columns='target_variant',values=field).to_csv(out/f'{field}_matrix.csv')
    metrics=affinity.set_index(['source_variant','target_variant'])
    a=left.sort_values('variant_id');b=right.sort_values('variant_id')
    av,bv=list(a.variant_id),list(b.variant_id)
    aa,bb=a.candidate_mass.to_numpy(),b.candidate_mass.to_numpy()
    penalty=cfg['unmatched_penalty']
    cost=affinity.cost.to_numpy(float).reshape(len(av),len(bv))
    available=affinity.available.to_numpy(bool).reshape(cost.shape)
    result=unbalanced_transport(aa,bb,cost,penalty,available)
    exclusions_path=root/'resources/excluded_signals.csv'
    gaps=excluded_mass(candidates,pd.read_csv(exclusions_path),name,cohorts)
    edge_rows=[];unmatched_rows=[]
    for i,x in enumerate(av):
        for j,y in enumerate(bv):
            row=metrics.loc[(x,y)].to_dict()
            row.update(source_variant=x,target_variant=y,source_signal=a.iloc[i].signal_id,target_signal=b.iloc[j].signal_id,
                       source_rsid=a.iloc[i].get('rsid',''),target_rsid=b.iloc[j].get('rsid',''),
                       source_mass=float(aa[i]),target_mass=float(bb[j]),transport_mass=float(result['plan'][i,j]),
                       transport_eligible=bool(result['eligible'][i,j]),
                       transport_gain=float(result['plan'][i,j]*(2*penalty-cost[i,j])) if available[i,j] else 0.)
            edge_rows.append(row)
    summary_row={'cancer':name,'lambda':penalty,'lp_count':result['lp_count'],
                 'transported_mass':result['transported_mass'],'gain_over_unmatched':result['gain_over_unmatched'],
                 'objective':result['objective']+penalty*sum(gaps.values())}
    for side,cohort,table,mass,unmatched in [
        ('source',cfg['source_cohort'],a,aa,result['unmatched_a']),
        ('target',cfg['target_cohort'],b,bb,result['unmatched_b'])]:
        for row,capacity,residual in zip(table.itertuples(),mass,unmatched):
            unmatched_rows.append(dict(side=side,cohort=cohort,variant_id=row.variant_id,signal_id=row.signal_id,
                                       input_mass=capacity,unmatched_mass=residual,reason='retained_SNV'))
        unmatched_rows.append(dict(side=side,cohort=cohort,variant_id='',signal_id='',input_mass=gaps[cohort],
                                   unmatched_mass=gaps[cohort],reason='excluded_or_unresolved'))
        summary_row.update({side+'_excluded_mass':gaps[cohort],side+'_unmatched':float(unmatched.sum()+gaps[cohort]),
                            side+'_input_mass':float(mass.sum()+gaps[cohort])})
    edges=pd.DataFrame(edge_rows)
    edges.to_csv(out/'transport_edges.csv',index=False)
    edges[edges.transport_mass>1e-10].to_csv(out/'transported_edges.csv',index=False)
    pd.DataFrame([summary_row]).to_csv(out/'transport_summary.csv',index=False)
    pd.DataFrame(unmatched_rows).to_csv(out/'unmatched_mass.csv',index=False)
    # Audit and trace the strongest edges plus every transported edge.
    explain=affinity.sort_values('affinity',ascending=False).head(20)[['source_variant','target_variant']]
    explain=pd.concat([explain,edges[edges.transport_mass>1e-10][['source_variant','target_variant']]]).drop_duplicates()
    evidence=[]
    for a,b in explain.itertuples(index=False,name=None):
        for view,tx,ty in [('tf',profiles[a].tf,profiles[b].tf),('accessibility',profiles[a].access,profiles[b].access),('pathway',profiles[a].pathway,profiles[b].pathway)]:
            _,_,ev=exact_affinity(tx,ty,details=True)
            for record in ev:record.update({'source_variant':a,'target_variant':b,'view':view})
            evidence.extend(ev)
    pd.DataFrame(evidence,columns=['source_variant','target_variant','view','coordinate','d_i','d_j','r_i','r_j']).to_csv(out/'edge_coordinate_evidence.csv',index=False)
    summary={'cancer':name,'source_variants':len(av),'target_variants':len(bv),'variant_pairs':len(affinity),
             'lp_count':1,'transported_edges':int((edges.transport_mass>1e-10).sum()),
             'transported_mass':result['transported_mass'],'unavailable_pairs':int((~affinity.available).sum())}
    write_json(out/'summary.json',summary)
    inputs=[root/spec['input'],score_path,catalog_path,mapping_path,fetch_prov_path,exclusions_path]
    code=sorted((root/'src/alphaot').glob('*.py'))
    write_json(out/'provenance.json',{'algorithm_version':'3.0.0','source':'AlphaGenome Atlas precomputed scores',
        'configuration':cfg,'input_sha256':{str(p.relative_to(root)):digest(p) for p in inputs},
        'code_sha256':{str(p.relative_to(root)):digest(p) for p in code},
        'python':platform.python_version(),'libraries':{k:version(k) for k in ['numpy','pandas','scipy','pyarrow']},
        'transport_scope':'one global EAS x EUR LP per cancer; raw capacities; no positional restrictions',
        'scorer_pairing':'Frozen wide rows produced by flatten var/obs equality checks; original separate response metadata unavailable',
        'scorer_semantics':{'q':'DIFF_LOG2_SUM signed quantile','active_q':'ACTIVE_SUM quantile'},
        'strength_semantics':'Regulation uses geometric mean of TF/accessibility H; pathway uses pre-projection equal-track RNA RMS harmonic support',
        'pathway_observation':'Both direction channels of pathway/track with at least one observed mapped gene; shared intersection with harmonic coverage gate'})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/analysis.json')
    p.add_argument('--output',default='outputs/results')
    p.add_argument('--cancer',nargs='+')
    p.add_argument('--lambda',dest='penalty',type=float)
    args=p.parse_args()
    cfg,root=config_at(args.config)
    if args.penalty is not None:cfg['unmatched_penalty']=args.penalty
    validate_config(cfg)
    out=Path(args.output);out=out if out.is_absolute() else root/out
    summaries=[]
    for name in args.cancer or cfg['cancers']:
        summaries.append(run_cancer(name,cfg['cancers'][name],cfg,root,out/name))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':main()
