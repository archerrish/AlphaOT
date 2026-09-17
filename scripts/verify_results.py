"""Independently verify two-view exports, raw global capacities and excluded mass."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(root,repo):
    a=pd.read_csv(root/'variant_affinities.csv');e=pd.read_csv(root/'transport_edges.csv')
    s=pd.read_csv(root/'transport_summary.csv');u=pd.read_csv(root/'unmatched_mass.csv')
    c=pd.read_csv(root/'candidate_masses.csv');p=json.loads((root/'provenance.json').read_text())
    cfg=p['configuration'];name=root.name
    assert len(s)==1 and s.lp_count.iloc[0]==1
    sr=s.iloc[0];penalty=sr['lambda']
    assert not c.duplicated(['cohort','variant_id']).any()
    assert np.allclose(c.candidate_mass,c.cs_specific_prob,rtol=0,atol=0)
    raw=pd.read_csv(repo/cfg['cancers'][name]['input'])
    assert np.array_equal(raw.cs_specific_prob,c.cs_specific_prob)
    assert len(e)==len(a)==c.cohort.eq(cfg['source_cohort']).sum()*c.cohort.eq(cfg['target_cohort']).sum()
    assert not a.duplicated(['source_variant','target_variant']).any()
    for prefix in ['tf','accessibility','atac','dnase']:
        ok=a[prefix+'_available'];v=a.loc[ok]
        assert (v[prefix+'_shared_coordinates']>0).all()
        assert np.allclose(v[prefix+'_K'],v[prefix+'_K_signed'].clip(lower=0))
        x,y=v[prefix+'_strength_i'],v[prefix+'_strength_j']
        h=np.divide(2*x*y,x+y,out=np.zeros(len(v)),where=(x+y).to_numpy()>0)
        assert np.allclose(v[prefix+'_H'],h)
        assert np.allclose(v[prefix+'_affinity'],v[prefix+'_K']*h)
        assert a.loc[~ok,prefix+'_affinity'].isna().all()
    ok=a.pathway_available;v=a.loc[ok]
    assert (v.pathway_shared_coordinates>0).all()
    assert np.allclose(v.pathway_K,v.pathway_K_signed.clip(lower=0))
    projected_h=np.divide(2*v.pathway_strength_i*v.pathway_strength_j,
                          v.pathway_strength_i+v.pathway_strength_j,
                          out=np.zeros(len(v)),where=(v.pathway_strength_i+v.pathway_strength_j).to_numpy()>0)
    assert np.allclose(v.pathway_H,projected_h)
    assert np.allclose(v.pathway_projected_affinity,v.pathway_K*v.pathway_H)
    regulation_strength=np.sqrt(a.tf_H*a.accessibility_H)
    assert np.allclose(a.regulation_strength,regulation_strength,equal_nan=True)
    assert np.allclose(a.affinity_regulation,a.tf_K*a.accessibility_K*regulation_strength,equal_nan=True)
    rna_h=np.divide(2*a.rna_strength_i*a.rna_strength_j,a.rna_strength_i+a.rna_strength_j,
                    out=np.zeros(len(a)),where=(a.rna_strength_i+a.rna_strength_j).to_numpy()>0)
    assert np.allclose(a.rna_H,rna_h)
    coverage_gate=np.divide(2*v.pathway_coverage_i*v.pathway_coverage_j,
                            v.pathway_coverage_i+v.pathway_coverage_j,
                            out=np.zeros(len(v)),where=(v.pathway_coverage_i+v.pathway_coverage_j).to_numpy()>0)
    assert np.allclose(v.pathway_coverage_gate,coverage_gate)
    assert np.allclose(v.affinity_pathway,v.pathway_K*v.rna_H*v.pathway_coverage_gate)
    assert a.loc[~ok,'affinity_pathway'].isna().all()
    assert np.allclose(a.affinity,a[['affinity_regulation','affinity_pathway']].max(axis=1),equal_nan=True)
    assert np.array_equal(a.available,a.affinity.notna())
    assert np.allclose(a.cost,1-a.affinity,equal_nan=True)
    assert a.loc[a.available,'affinity'].between(0,1+1e-10).all()
    assert 'affinity_rna' not in a
    assert np.allclose(e.affinity,a.affinity,equal_nan=True)
    assert (e.transport_mass>=0).all()
    expected_eligible=e.available & (e.cost<2*penalty) & (e.source_mass>0) & (e.target_mass>0)
    assert np.array_equal(e.transport_eligible,expected_eligible)
    assert (e.loc[~e.transport_eligible,'transport_mass']==0).all()
    total=e.transport_mass.sum();assert np.isclose(total,sr.transported_mass)
    # Independent original inventory: includes fully excluded signals exactly once.
    inventory=pd.read_csv(repo/'resources/source_signals.csv')
    inventory=inventory[inventory.Cancer.eq({'breast':'BsC','prostate':'PrC','thyroid':'ThC'}[name])]
    assert len(inventory)>0
    for side,cohort in [('source',cfg['source_cohort']),('target',cfg['target_cohort'])]:
        caps=c[c.cohort.eq(cohort)].set_index('variant_id').candidate_mass
        sums=e.groupby(side+'_variant').transport_mass.sum().reindex(caps.index)
        assert (sums<=caps+1e-8).all()
        assert np.allclose(e[side+'_mass'],e[side+'_variant'].map(caps))
        original=inventory.loc[inventory['Meta-analysis'].eq(cohort),'original_probability_total'].sum()
        gap=original-caps.sum()
        assert np.isclose(sr[side+'_excluded_mass'],gap)
        assert np.isclose(sr[side+'_input_mass'],original)
        side_u=u[u.side.eq(side)];excluded=side_u[side_u.reason.eq('excluded_or_unresolved')]
        assert len(excluded)==1 and np.isclose(excluded.unmatched_mass.iloc[0],gap)
        retained=side_u[side_u.reason.eq('retained_SNV')].set_index('variant_id')
        assert not retained.index.duplicated().any()
        assert np.allclose(retained.unmatched_mass.reindex(caps.index),caps-sums)
        assert np.isclose(side_u.unmatched_mass.sum(),sr[side+'_unmatched'])
        assert np.isclose(total+sr[side+'_unmatched'],original)
    objective=(e.transport_mass*e.cost).sum()+penalty*(sr.source_unmatched+sr.target_unmatched)
    assert np.isclose(objective,sr.objective)
    assert np.isclose(sr.gain_over_unmatched,penalty*(sr.source_input_mass+sr.target_input_mass)-objective)
    import hashlib
    for rel,sha in {**p['input_sha256'],**p['code_sha256']}.items():
        assert hashlib.sha256((repo/rel).read_bytes()).hexdigest()==sha,rel
    print(f'{name}: one global LP; formulas, raw capacities, excluded mass, objective and provenance verified')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=Path('outputs/results'))
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    paths=sorted(args.results.glob('*/transport_summary.csv'))
    if not paths:raise ValueError('No results')
    for path in paths:verify(path.parent,repo)
    print('Verification passed')

if __name__=='__main__':main()
