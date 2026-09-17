"""Strict SNV inputs and portable configuration paths."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def config_at(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    return config, path.parent.parent


def read_candidates(path, cancer):
    d = pd.read_csv(path, dtype={'chrom':str, 'ref':str, 'alt':str, 'signal_id':str})
    required = ['cancer','cohort','signal_id','variant_id','chrom','position','ref','alt','assembly','cs_specific_prob','signal_probability_total']
    if set(required)-set(d):
        raise ValueError(f'Missing input fields: {sorted(set(required)-set(d))}')
    if d.empty or d[required].isna().any().any():
        raise ValueError('Inputs must contain complete SNV membership rows')
    if not d.cancer.eq(cancer).all() or not d.assembly.isin(['GRCh38','hg38']).all():
        raise ValueError('Cancer or assembly mismatch')
    if not (d.ref.str.fullmatch('[ACGT]') & d.alt.str.fullmatch('[ACGT]') & d.ref.ne(d.alt)).all():
        raise ValueError('Only biallelic SNVs with explicit uppercase REF/ALT are accepted')
    if not d.chrom.str.fullmatch(r'chr([1-9]|1[0-9]|2[0-2]|X|Y|M)').all():
        raise ValueError('Chromosome must be chr1..chr22, chrX, chrY or chrM')
    pos = pd.to_numeric(d.position, errors='raise')
    if not (np.isfinite(pos) & (pos>=1) & (pos==np.floor(pos))).all():
        raise ValueError('Positions must be positive 1-based integers')
    d['position'] = pos.astype(int)
    expected = d.chrom+':'+d.position.astype(str)+':'+d.ref+'>'+d.alt
    if not expected.eq(d.variant_id).all():
        raise ValueError('variant_id must equal chrom:position:REF>ALT')
    if d.duplicated(['cohort','variant_id']).any():
        raise ValueError('Duplicate cohort variant')
    for col in ['cs_specific_prob','signal_probability_total']:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not np.isfinite(d[col]).all():
            raise ValueError(f'Nonfinite {col}')
    if (d.cs_specific_prob<0).any() or (d.cs_specific_prob>1).any() or (d.signal_probability_total<=0).any():
        raise ValueError('Invalid credible-set probabilities')
    if (d.groupby(['cohort','signal_id']).signal_probability_total.nunique()!=1).any():
        raise ValueError('Inconsistent original signal probability total')
    mass = d.cs_specific_prob
    if 'candidate_mass' in d and not np.allclose(d.candidate_mass,mass,atol=1e-10,rtol=1e-8):
        raise ValueError('candidate_mass must equal raw cs_specific_prob')
    d['candidate_mass'] = mass
    grouped = d.groupby(['cohort','signal_id'])
    if (grouped.candidate_mass.sum() > grouped.signal_probability_total.first()+1e-8).any():
        raise ValueError('Retained mass exceeds original signal total')
    return d


def excluded_mass(candidates, excluded, cancer, cohorts):
    """Original minus retained mass per signal, then aggregate once per cohort."""
    totals=candidates.groupby(['cohort','signal_id']).signal_probability_total.first()
    retained=candidates.groupby(['cohort','signal_id']).candidate_mass.sum()
    extra=excluded[excluded.cancer.eq(cancer)].rename(columns={'Meta-analysis':'cohort'})
    if extra.duplicated(['cohort','signal_id']).any():raise ValueError('Duplicate excluded signal')
    if not set(extra.cohort).issubset(cohorts):raise ValueError('Unknown excluded cohort')
    if not np.isfinite(extra.original_probability_total).all() or (extra.original_probability_total<0).any():
        raise ValueError('Invalid excluded mass')
    for row in extra.itertuples():
        key=(row.cohort,row.signal_id)
        if key in totals.index:raise ValueError('Excluded signal overlaps retained signal')
        totals.loc[key]=row.original_probability_total
    gaps=totals-retained.reindex(totals.index,fill_value=0)
    if (gaps < -1e-8).any():raise ValueError('Negative excluded mass')
    return {c:float(gaps.clip(lower=0).groupby(level=0).sum().get(c,0)) for c in cohorts}


def validate_config(cfg):
    allowed={'schema_version','unmatched_penalty','integration','source_cohort','target_cohort',
             'pathway_size_range','cancers','lambda_sensitivity'}
    if set(cfg)!=allowed:raise ValueError('Missing or unsupported configuration fields')
    if cfg.get('schema_version')!=4 or cfg.get('integration')!='max':raise ValueError('Expected schema 4 max integration')
    if cfg['source_cohort']==cfg['target_cohort']:raise ValueError('Cohorts must differ')
    for p in [cfg['unmatched_penalty'],*cfg['lambda_sensitivity']]:
        if not np.isfinite(p) or p<=0:raise ValueError('Invalid lambda')
    if not cfg['lambda_sensitivity'] or len(set(cfg['lambda_sensitivity']))!=len(cfg['lambda_sensitivity']):
        raise ValueError('Sensitivity lambdas must be nonempty and unique')
    if len(cfg['pathway_size_range'])!=2 or not 0<cfg['pathway_size_range'][0]<=cfg['pathway_size_range'][1]:
        raise ValueError('Invalid pathway size range')
    if not cfg['cancers']:raise ValueError('No cancers configured')
    for spec in cfg['cancers'].values():
        if set(spec)!={'input','scores','contexts'}:raise ValueError('Unsupported cancer configuration')
        for context in spec['contexts'].values():
            if 'name' not in context or set(context)-{'name','tier','ontology_curie'}:
                raise ValueError('Unsupported context configuration')
