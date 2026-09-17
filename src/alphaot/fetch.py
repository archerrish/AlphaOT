"""Fetch resumable Atlas scores; no model inference and no credential in outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from .io import config_at, read_candidates, digest

MODALITIES = ['RNA_SEQ','CHIP_TF','ATAC','DNASE']
IDENTITY = ['name','strand','Assay title','ontology_curie','biosample_name','biosample_type',
            'data_source','endedness','genetically_modified','transcription_factor','gtex_tissue']


def identity(row):
    return tuple(str(row.get(k, '') if pd.notna(row.get(k, '')) else '') for k in IDENTITY)


def catalog_from(client):
    metadata = client.scorer_metadata()
    frames, lookup = [], {}
    for mod in MODALITIES:
        frame = metadata[mod].track_metadata.fillna('').copy()
        active = metadata[mod+'_ACTIVE'].track_metadata.fillna('')
        if not frame.equals(active):
            raise ValueError(f'{mod}: differential and activity track catalogs differ')
        mapping = {}
        for index, row in frame.iterrows():
            key = identity(row)
            if key in mapping:
                raise ValueError(f'{mod}: ambiguous full track identity')
            mapping[key] = f'{mod}:{index}'
        frame['track_id'] = [mapping[identity(r)] for _,r in frame.iterrows()]
        frame['modality'] = mod
        frame['atlas_index'] = frame.index
        frames.append(frame.reset_index(drop=True)); lookup[mod] = mapping
    return pd.concat(frames, ignore_index=True).fillna(''), lookup


def flatten(result, variant_id, lookup):
    rows = []
    for mod in MODALITIES:
        if mod not in result and mod+'_ACTIVE' not in result:
            continue  # No assay in the requested ontology filter.
        if mod not in result or mod+'_ACTIVE' not in result:
            raise ValueError(f'{variant_id}: unpaired scorer {mod}')
        d, a = result[mod], result[mod+'_ACTIVE']
        if not d.var.equals(a.var) or not d.obs.equals(a.obs):
            raise ValueError(f'{variant_id}: active/differential coordinates differ')
        raw, q, ar, aq = map(np.asarray, [d.X,d.layers['quantiles'],a.X,a.layers['quantiles']])
        if not all(x.shape==(len(d.obs),len(d.var)) for x in [raw,q,ar,aq]):
            raise ValueError('Unexpected Atlas score shape')
        for j, (_,t) in enumerate(d.var.fillna('').iterrows()):
            track_id = lookup[mod][identity(t)]
            for i, (_,g) in enumerate(d.obs.iterrows()):
                rows.append({'variant_id':variant_id,'modality':mod,'track_id':track_id,
                    'gene_id':str(g.get('gene_id','')).split('.')[0],
                    'gene_name':g.get('gene_name',''), 'gene_strand':g.get('strand',''),
                    'raw':float(raw[i,j]),'q':float(q[i,j]),
                    'active_raw':float(ar[i,j]),'active_q':float(aq[i,j])})
    return pd.DataFrame(rows,columns=['variant_id','modality','track_id','gene_id','gene_name','gene_strand','raw','q','active_raw','active_q'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/analysis.json')
    p.add_argument('--env-file',type=Path,default=Path('.env'))
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--cancer',nargs='+')
    args=p.parse_args()
    if not 1<=args.workers<=10: p.error('workers must be 1..10')
    from dotenv import load_dotenv
    from alphagenome.atlas import atlas
    from alphagenome.data import genome
    load_dotenv(args.env_file)
    key=os.environ.get('ALPHAGENOME_API_KEY') or os.environ.get('ALPHA_GENOME_API_KEY')
    if not key: raise SystemExit('Set ALPHAGENOME_API_KEY in a local .env; never put the key in a CSV or command line.')
    cfg,root=config_at(args.config)
    dest=root/'resources/atlas'; dest.mkdir(parents=True,exist_ok=True)
    client=atlas.create(key,timeout=90)
    catalog,lookup=catalog_from(client)
    catpath=dest/'track_catalog.parquet'
    if catpath.exists() and not pd.read_parquet(catpath).equals(catalog):
        raise ValueError('Atlas catalog changed; start a separate snapshot instead of mixing versions')
    catalog.to_parquet(catpath,index=False)
    catalog_hash=digest(catpath)
    names=args.cancer or list(cfg['cancers'])
    requested=[s for m in MODALITIES for s in [m,m+'_ACTIVE']]
    for name in names:
        spec=cfg['cancers'][name]
        inp=root/spec['input']; candidates=read_candidates(inp,name)
        terms=sorted({v.get('ontology_curie',k) for k,v in spec['contexts'].items()})
        cache=dest/'variants'/name;cache.mkdir(parents=True,exist_ok=True)
        request={'catalog_sha256':catalog_hash,'ontology_terms':terms,'scorers':requested,'source':'AlphaGenome Atlas precomputed scores'}
        def fetch_one(vid):
            stem=vid.replace(':','_').replace('>','_'); target=cache/(stem+'.parquet'); manifest=cache/(stem+'.json')
            if target.exists() and manifest.exists():
                m=json.loads(manifest.read_text())
                if any(m.get(k)!=v for k,v in request.items()) or m.get('data_sha256')!=digest(target):
                    raise ValueError(f'Cache provenance mismatch: {vid}')
                return pd.read_parquet(target),None
            for attempt in range(3):
                try:
                    result=client.query_variant(genome.Variant.from_str(vid),requested_scorers=requested,ontology_terms=terms)
                    frame=flatten(result,vid,lookup)
                    frame.to_parquet(target,index=False)
                    manifest.write_text(json.dumps({**request,'variant_id':vid,'queried_at':datetime.now(timezone.utc).isoformat(),'data_sha256':digest(target),'model_inference':False},indent=2)+'\n')
                    return frame,None
                except Exception as exc:
                    if attempt==2:
                        # Avoid putting opaque server messages or credentials in logs.
                        return None,{'variant_id':vid,'error_type':type(exc).__name__}
                    time.sleep(1+attempt)
        frames=[];failures=[]
        ids=sorted(candidates.variant_id.unique())
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(fetch_one,vid):vid for vid in ids}
            for count, future in enumerate(as_completed(futures),1):
                frame,failure=future.result()
                if failure:failures.append(failure)
                else:frames.append(frame)
                if count%10==0 or count==len(ids):print(f'{name}: {count}/{len(ids)} fetched; failures={len(failures)}',flush=True)
        pd.DataFrame(failures,columns=['variant_id','error_type']).to_csv(dest/f'{name}_fetch_failures.csv',index=False)
        if failures:
            raise RuntimeError(f'{name}: {len(failures)} Atlas queries failed; retry to resume the cache')
        combined=pd.concat(frames,ignore_index=True).sort_values(['variant_id','modality','track_id','gene_id'])
        combined.to_parquet(root/spec['scores'],index=False)
        (dest/f'{name}_provenance.json').write_text(json.dumps({**request,'input_sha256':digest(inp),'n_variants':len(ids),'rows':len(combined),'model_inference':False,'data_sha256':digest(root/spec['scores'])},indent=2)+'\n')


if __name__=='__main__':main()
