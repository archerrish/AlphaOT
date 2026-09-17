"""Render the retained AlphaOT panels from frozen results and native tracks."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/alphaot-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANCERS = ['breast', 'prostate', 'thyroid']
REF, ALT = '#3284bd', '#df4545'
REG, PATH, YELLOW = '#0072B2', '#CC79A7', '#E6B800'
MODALITIES = ['RNA_SEQ', 'CHIP_TF', 'ATAC', 'DNASE']
MOD_COLORS = ['#d9c3d0', '#80c1da', '#e7bf92', '#86aaa7']
MOD_LABELS = ['RNA-seq', 'TF ChIP-seq', 'ATAC-seq', 'DNase-seq']
REG_PAIR = ('chr10:79129101:G>A', 'chr1:121522281:G>A')
PATH_PAIR = ('chr10:121577603:C>T', 'chr10:121579461:A>G')
RNA_TRACKS = ['RNA_SEQ:' + str(i) for i in [114, 115, 116, 259, 358, 46, 47, 48, 49, 55]]
RNA_LABELS = ['MCF 10A / total', 'MCF-7 / polyA+', 'MCF-7 / total',
              'Breast epith. / total', 'Breast epith. / polyA+', 'Myoepithelial / polyA+',
              'Luminal epith. / polyA+', 'Mammary epith. / polyA+',
              'Mammary epith. / total', 'Mammary stem / polyA+']


def subrect(rect, x, y, w, h):
    a, b, c, d = rect
    return [a + x*c, b + y*d, w*c, h*d]


def axis(fig, rect, *bounds):
    return fig.add_axes(subrect(rect, *bounds))


def chromosome_key(v):
    chrom, pos, allele = v.split(':')
    order = {str(i): i for i in range(1, 23)} | {'X': 23, 'Y': 24, 'M': 25}
    return order[chrom[3:]], int(pos), allele


def panel_b(fig, rect, data):
    positions = [(0,.47),(.50,.47),(0,.02)]
    for cancer,(x,y) in zip(CANCERS,positions):
        counts = data['counts'][cancer]
        ax = axis(fig,rect,x,y,.48,.46)
        ax.pie(counts,colors=MOD_COLORS,startangle=90,counterclock=False,
               wedgeprops={'width':.36,'edgecolor':'white','linewidth':.6},
               autopct=lambda p: f'{p:.1f}%' if p>=10 else '',pctdistance=.80,
               textprops={'fontsize':5.5})
        ax.text(0,0,cancer.title()+'\ncancer',ha='center',va='center',fontsize=6)
    ax=axis(fig,rect,.50,.01,.50,.43);ax.axis('off')
    ax.legend([Rectangle((0,0),1,1,color=c) for c in MOD_COLORS],MOD_LABELS,
              frameon=False,fontsize=6,loc='center left',labelspacing=.9,handlelength=1)
    fig.text(rect[0]+rect[2]/2,rect[1]+rect[3], 'Included functional tracks',ha='center',fontsize=7)


def panel_c(fig, rect, data):
    ax=axis(fig,rect,0,.06,1,.90);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    edges=data['edges']['breast']; left=sorted(edges.source_variant.unique(),key=chromosome_key)
    right=sorted(edges.target_variant.unique(),key=chromosome_key)
    yy=[dict(zip(ids,np.linspace(.89,.09,len(ids)))) for ids in [left,right]]
    for row in edges.itertuples():
        y0,y1=yy[0][row.source_variant],yy[1][row.target_variant]
        path=MplPath([(.27,y0),(.44,y0),(.56,y1),(.73,y1)], [1,4,4,4])
        ax.add_patch(PathPatch(path,facecolor='none',edgecolor=YELLOW if row.same_variant else '#858585',
                               lw=1.3 if row.same_variant else .8,zorder=1))
    for side,ids,x,field in [(0,left,.27,'source'),(1,right,.73,'target')]:
        labels=edges.drop_duplicates(field+'_variant').set_index(field+'_variant')[field+'_rsid']
        for v in ids:
            ax.scatter(x,yy[side][v],s=10,c='#222222',zorder=3)
            ax.text(x+(-.035 if side==0 else .035),yy[side][v],labels[v],fontsize=5.4,
                    ha='right' if side==0 else 'left',va='center')
    ax.text(.27,.98,'EAS',ha='center',color=REF,fontsize=7)
    ax.text(.73,.98,'EUR',ha='center',color='#cf753e',fontsize=7)
    fig.text(rect[0]+rect[2]/2,rect[1]+rect[3], 'Breast cancer: 12 transported pairs',ha='center',fontsize=7)
    ax.legend([Line2D([0],[0],c='#858585'),Line2D([0],[0],c=YELLOW)],
              ['Different variants','Identical variant'],frameon=False,fontsize=5.5,ncol=2,
              loc='lower center',bbox_to_anchor=(.5,-.105),handlelength=1)


def panel_d(fig, rect, data):
    thresholds=np.linspace(0,.85,340)
    for k,cancer in enumerate(CANCERS):
        ax=axis(fig,rect,.18,.72-k*.32,.78,.23);a=data['affinities'][cancer]
        for view,color in [('regulation',REG),('pathway',PATH)]:
            v=a['affinity_'+view].to_numpy(float)
            frac=np.array([np.count_nonzero(np.isfinite(v)&(v>=x))/len(v) for x in thresholds])
            frac[frac==0]=np.nan;ax.plot(thresholds,frac,c=color,lw=.85)
        ax.axvline(.4,ls='--',lw=.6,c='#777777')
        ax.set(yscale='log',xlim=(0,.85),ylim=(.6/len(a),1.3))
        ax.set_xticks([0,.2,.4,.6,.8]);ax.tick_params(labelsize=5.3,length=2,pad=1)
        ax.text(.97,.93,cancer.title()+' cancer',ha='right',va='top',fontsize=6,transform=ax.transAxes)
        if k==2:ax.set_xlabel('Affinity',fontsize=6,labelpad=2)
    fig.text(rect[0],rect[1]+rect[3]/2,'Fraction of all variant pairs',rotation=90,ha='center',va='center',fontsize=6)
    fig.legend([Line2D([0],[0],c=REG),Line2D([0],[0],c=PATH)],['Regulation','Pathway'],
               loc='lower center',bbox_to_anchor=(rect[0]+rect[2]*.56,rect[1]+rect[3]*.96),
               frameon=False,ncol=2,fontsize=5.5,handlelength=1.3,columnspacing=1)


def ticks(ids):
    ch=[v.split(':')[0][3:] for v in ids]; pos=[];labels=[];bounds=[];start=0
    for i in range(1,len(ids)+1):
        if i==len(ids) or ch[i]!=ch[start]:
            pos.append((start+i-1)/2);labels.append(ch[start]);bounds.append(i-.5);start=i
    return pos,labels,bounds[:-1]


def panel_e(fig, rect, data):
    a=data['affinities']['breast']; left=sorted(a.source_variant.unique(),key=chromosome_key)
    right=sorted(a.target_variant.unique(),key=chromosome_key)
    li={v:i for i,v in enumerate(left)};ri={v:i for i,v in enumerate(right)}
    positive=set(zip(data['edges']['breast'].source_variant,data['edges']['breast'].target_variant))
    for k,(column,title,color,vmax) in enumerate([('affinity_regulation','Regulation affinity',REG,.81),
                                                ('affinity_pathway','Pathway affinity',PATH,.42)]):
        ax=axis(fig,rect,.14,.56-k*.47,.73,.38)
        matrix=a.pivot(index='source_variant',columns='target_variant',values=column).loc[left,right]
        cmap=LinearSegmentedColormap.from_list(title,['white',color]);cmap.set_bad('#d9d9d9')
        im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),cmap=cmap,vmin=0,vmax=vmax,aspect='auto',interpolation='nearest')
        x,l,b=ticks(right);ax.set_xticks(x,l,rotation=90)
        for z in b:ax.axvline(z,c='white',lw=.25)
        x,l,b=ticks(left);ax.set_yticks(x,l)
        for z in b:ax.axhline(z,c='white',lw=.25)
        ax.set_title(title,fontsize=6.5,pad=3);ax.tick_params(length=1,pad=1,labelsize=5)
        ax.set_ylabel('EAS chromosome',fontsize=5.5,labelpad=1)
        if k==1:ax.set_xlabel('EUR chromosome',fontsize=5.5,labelpad=1)
        if k==0:
            for r in a[a.same_variant | a.apply(lambda r: (r.source_variant,r.target_variant) in positive,axis=1)].itertuples():
                pair=(r.source_variant,r.target_variant);marker='s' if r.same_variant and pair in positive else ('D' if r.same_variant else 'o')
                edge='#7137aa' if marker=='s' else ('#ef2525' if marker=='D' else '#eda16e')
                ax.scatter(ri[pair[1]],li[pair[0]],s=12,marker=marker,facecolors='none',edgecolors=edge,lw=.65)
        pair=REG_PAIR if k==0 else PATH_PAIR
        ax.add_patch(Rectangle((ri[pair[1]]-.6,li[pair[0]]-.6),1.2,1.2,fill=False,edgecolor=YELLOW if k==0 else '#eb5266',lw=.9))
        cax=axis(fig,rect,.90,.56-k*.47,.028,.38);fig.colorbar(im,cax=cax).ax.tick_params(labelsize=5,length=1,pad=1)
    legend=[Line2D([0],[0],marker=m,c='none',markerfacecolor='none',markeredgecolor=c,markersize=3.8)
            for m,c in [('s','#7137aa'),('D','#ef2525'),('o','#eda16e')]]
    fig.legend(legend,['Identical, transported','Identical, untransported','Different, transported'],
               loc='upper left',bbox_to_anchor=(rect[0]+.01,rect[1]+.009),fontsize=5.2,frameon=False,handlelength=1,labelspacing=.3)


def spatial(rsid,key):
    prefix=ROOT/'resources/spatial'/rsid;meta=json.loads(prefix.with_suffix('.json').read_text());f=meta['features'][key]
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as a:
        ref=a[key+'_ref'].copy();alt=a[key+'_alt'].copy()
    assert len(ref)==len(alt) and len(ref)*f['resolution']==f['end']-f['start']
    assert np.isfinite(ref).all() and np.isfinite(alt).all() and min(ref.min(),alt.min())>=0
    return meta,f,ref,alt


def draw_signal(ax,rsid,key,marker='#ffcc00'):
    meta,f,ref,alt=spatial(rsid,key);width=f['display_window_bp'];center=meta['position']-1
    edges=f['start']+np.arange(len(ref)+1)*f['resolution']-center
    limit=max(ref.max(),alt.max(),1e-6)*1.15
    mirror=key in ['ATAC','DNASE']
    if mirror:
        assert width==512 and len(ref)==512 and f['resolution']==1
        ax.stairs(ref,edges,color=REF,fill=True,lw=.2);ax.stairs(-alt,edges,color=ALT,fill=True,lw=.2)
        ax.set_ylim(-limit,limit);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{abs(v):g}'))
        ax.axhline(0,c='#999999',lw=.25)
    else:
        xx=np.r_[edges[0],(edges[:-1]+edges[1:])/2,edges[-1]]
        rr=np.r_[ref[0],ref,ref[-1]];aa=np.r_[alt[0],alt,alt[-1]]
        ax.fill_between(xx,rr,color=REF,alpha=.15,lw=0)
        ax.plot(xx,rr,c=REF,lw=.65);ax.plot(xx,aa,c=ALT,lw=.65);ax.set_ylim(0,limit)
    ax.set_xlim(-width/2,width/2);ax.set_xticks([]);ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=5,length=2,pad=1);ax.spines[['top','right','bottom']].set_visible(False)
    ax.axvline(0,c=marker,lw=.6)
    if marker=='#ffcc00':ax.plot(0,limit,marker='v',c=marker,ms=3,clip_on=False)
    return meta


def panel_f(fig, rect, data):
    annotation=pd.read_feather(ROOT/'resources/spatial/gencode_v46_case_genes.feather')
    for col,(rsid,symbol,gid) in enumerate([('rs9728991','EMBP1','ENSG00000291141'),('rs7097066','ZMIZ1','ENSG00000108175')]):
        x=.13+col*.47;w=.39
        ax=axis(fig,rect,x,.875,w,.09);g=annotation[annotation.gene_id.str.split('.').str[0].eq(gid)]
        gene=g[g.Feature.eq('gene')].iloc[0];exons=g[g.Feature.eq('exon')][['Start','End']].drop_duplicates()
        meta,_,_,_=spatial(rsid,'FOXA1');pos=meta['position']-1;start,end=int(gene.Start),int(gene.End)
        assert start<=pos<end;pad=(end-start)*.03
        ax.set(xlim=(start-pad,end+pad),ylim=(0,1));ax.axis('off');ax.plot([start,end],[.4,.4],c='#444444',lw=.5)
        for p in np.linspace(start,end,18)[1:-1]:ax.plot(p,.4,marker='>',c='#444444',ms=2)
        for a,b in exons.itertuples(index=False,name=None):ax.add_patch(Rectangle((a,.23),b-a,.34,color='#444444',lw=0))
        ax.axvline(pos,c='#ffcc00',lw=.6);ax.text(.5,.9,symbol+' (+)',ha='center',fontsize=6.5,fontstyle='italic',transform=ax.transAxes)
        ax.text(.5,-.15,f"{meta['chromosome']}:{meta['position']:,} G>A",ha='center',fontsize=5.7,color='#bb9200',transform=ax.transAxes)
        for row,key in enumerate(['FOXA1','GATA3','ESR1','ATAC','DNASE']):
            y=.70-row*.158;ax=axis(fig,rect,x,y,w,.12);draw_signal(ax,rsid,key)
            if col==0:ax.set_ylabel('DNase' if key=='DNASE' else key,fontsize=6,labelpad=3)
            if row==0:ax.set_title('MCF-7 (10 kb)',fontsize=6,pad=5)
            if row==3:ax.set_title('MCF-7 (512 bp)',fontsize=6,pad=5)
    fig.text(rect[0]+rect[2]*.55,rect[1]-.009,'ESR1: modified-cell illustration; excluded from affinity',ha='center',fontsize=5.2,color='#666666')


def panel_g(fig, rect, data):
    for row,rsid in enumerate(['rs9728991','rs7097066']):
        ax=axis(fig,rect,.18,.57-row*.48,.80,.33);meta=draw_signal(ax,rsid,'H3K4me1',marker='#5278c9')
        ax.set_title(f"{meta['chromosome']}:{meta['position']:,} G>A",fontsize=6,pad=4)
        ax.set_ylabel('H3K4me1',fontsize=6,labelpad=2)
    fig.text(rect[0]+rect[2]*.59,rect[1]-.006,'MCF-7 (10 kb)',ha='center',fontsize=6)


def panel_i(fig, rect, data):
    r=data['rna'];mapping=data['mapping']
    genes=r[r.variant_id.eq(PATH_PAIR[0])][['gene_id','gene_name']].drop_duplicates().sort_values('gene_id')
    mapped=set(genes.gene_id)&set(mapping.gene_id)
    order=[g for g in genes.gene_id if g in mapped]+[g for g in genes.gene_id if g not in mapped]
    names=genes.set_index('gene_id').gene_name.to_dict();members=mapping[mapping.gene_id.isin(order)]
    paths=sorted(members.pathway_id.unique());assert len(order)==17 and len(paths)==17 and len(mapped)==2
    ax=axis(fig,rect,0,0,1,.95);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ys=np.linspace(.86,.18,17);py=np.linspace(.86,.18,17);cmap=plt.get_cmap('RdBu_r')
    for col,v in enumerate(PATH_PAIR):
        x0=.30+col*.17;q=r[r.variant_id.eq(v)].pivot(index='gene_id',columns='track_id',values='q').loc[order,RNA_TRACKS]
        assert q.notna().all().all()
        for n,g in enumerate(order):
            for t in range(10):ax.add_patch(Rectangle((x0+t*.0105,ys[n]-.015),.01,.03,facecolor=cmap((q.iloc[n,t]+1)/2),edgecolor='none'))
        ax.text(x0+.05,.92,['rs2912780','rs1078806'][col],ha='center',fontsize=6,color=REF if col==0 else '#cf753e')
        ax.text(x0+.05,.115,'10 RNA tracks',ha='center',fontsize=5.2)
    for n,g in enumerate(order):
        ax.text(.287,ys[n],names[g],ha='right',va='center',fontsize=5.4,fontstyle='italic',color='#222222' if g in mapped else '#aaaaaa')
        if g in mapped:ax.scatter(.60,ys[n],s=5,c=PATH)
    for row in members.itertuples():
        ax.plot([.604,.73],[ys[order.index(row.gene_id)],py[paths.index(row.pathway_id)]],c=PATH,lw=row.projection_weight*7,alpha=.55)
    for n,p in enumerate(paths):ax.scatter(.735,py[n],s=5,c=PATH)
    # Display selected short labels while exporting exact names in the source table.
    short={'R-HSA-5654695':'PI-3K cascade: FGFR2','R-HSA-109704':'PI3K cascade','R-HSA-1257604':'PIP3 activates AKT'}
    used=[]
    for p,label in short.items():
        if p in paths:
            y=py[paths.index(p)]
            if not used or all(abs(y-v)>.055 for v in used):ax.text(.76,y,label,fontsize=4.8,va='center');used.append(y)
    ax.text(.765,.38,'FGFR2 signalling\nand DNA repair\npathways',fontsize=5,va='center',linespacing=1.5)
    ax.text(.77,.11,'17 pathways',fontsize=5.5)
    ax.text(.65,.035,r'$w_{gp}=1/\sqrt{|p|\delta_g}$',ha='center',fontsize=6.5)


def panel_j(fig, rect, data):
    r=data['rna'];genes=['FGFR2','NSMCE4A']
    for row,v in enumerate(PATH_PAIR):
        ax=axis(fig,rect,.09,.69-row*.25,.77,.14)
        d=r[r.variant_id.eq(v) & r.gene_name.isin(genes)];q=d.pivot(index='gene_name',columns='track_id',values='q').loc[genes,RNA_TRACKS]
        a=d.pivot(index='gene_name',columns='track_id',values='active_q').loc[genes,RNA_TRACKS]
        assert q.notna().all().all() and a.notna().all().all()
        im=ax.imshow(q,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');yy,xx=np.indices(q.shape)
        ax.scatter(xx.ravel(),yy.ravel(),s=12*a.to_numpy().ravel(),facecolors='none',edgecolors='#333333',linewidths=.35)
        ax.set_yticks([0,1],genes,fontsize=6,fontstyle='italic');ax.yaxis.tick_right()
        ax.set_xticks(range(10),RNA_LABELS if row else ['']*10,rotation=48,ha='right',fontsize=5.3)
        ax.tick_params(length=0,pad=2);ax.set_xticks(np.arange(-.5,10,1),minor=True);ax.set_yticks([-.5,.5,1.5],minor=True)
        ax.grid(which='minor',c='white',lw=.4);ax.tick_params(which='minor',length=0)
        label=['rs2912780 (EAS)','rs1078806 (EUR)'][row]
        ax.set_title(label+' | '+v,fontsize=5.8,loc='left',pad=4,color=REF if row==0 else '#cf753e')
    cax=axis(fig,rect,.15,.14,.38,.018);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-1,0,1])
    cb.ax.tick_params(labelsize=5,length=1,pad=1);cb.set_label('RNA signed-effect quantile',fontsize=5.5,labelpad=2)
    ax=axis(fig,rect,.66,.065,.24,.12);ax.axis('off');ax.set(xlim=(-.5,2.5),ylim=(0,1))
    for n,a in enumerate([.25,.5,1.]):
        ax.scatter(n,.45,s=12*a,facecolors='none',edgecolors='#333333',lw=.35);ax.text(n,.02,str(a),ha='center',fontsize=5)
    ax.text(1,.88,'RNA activity (area)',ha='center',fontsize=5.5)
    row=data['affinities']['breast'].set_index(['source_variant','target_variant']).loc[PATH_PAIR]
    fig.text(rect[0]+rect[2]/2,rect[1]-.006,f'Pathway affinity = {row.affinity_pathway:.3f}; not transported at λ = 0.30',ha='center',fontsize=6,color='#9b4e80')


PANELS={'b':panel_b,'c':panel_c,'d':panel_d,'e':panel_e,'f':panel_f,'g':panel_g,'i':panel_i,'j':panel_j}
LAYOUT={'b':(.05,.64,.27,.19),'c':(.345,.64,.30,.19),'d':(.70,.64,.26,.19),
        'e':(.05,.30,.235,.285),'f':(.32,.28,.375,.305),'g':(.735,.445,.225,.14),
        'i':(.05,.048,.46,.20),'j':(.54,.048,.40,.20)}


def load_data(results):
    data={'affinities':{},'edges':{},'counts':{}}
    for c in CANCERS:
        p=results/c;data['affinities'][c]=pd.read_csv(p/'variant_affinities.csv');data['edges'][c]=pd.read_csv(p/'transported_edges.csv')
        audit=pd.read_csv(p/'track_audit.csv');counts=audit[audit.included].groupby('modality').track_id.nunique()
        data['counts'][c]=counts.reindex(MODALITIES,fill_value=0).to_numpy(int)
    assert [len(data['edges'][c]) for c in CANCERS]==[12,11,3]
    assert data['counts']['breast'].tolist()==[10,68,2,5]
    effects=pd.read_parquet(results/'breast/exact_track_effects.parquet')
    data['rna']=effects[effects.modality.eq('RNA_SEQ') & effects.variant_id.isin(PATH_PAIR)]
    data['mapping']=pd.read_parquet(ROOT/'resources/reactome_mapping.parquet')
    assert not ((data['edges']['breast'].source_variant==PATH_PAIR[0])&(data['edges']['breast'].target_variant==PATH_PAIR[1])).any()
    return data


def save(fig,stem):
    for extension in ['pdf','png']:
        kwargs={'metadata': {'CreationDate': None, 'ModDate': None}} if extension=='pdf' else {}
        fig.savefig(stem.with_suffix('.'+extension),dpi=300,facecolor='white',**kwargs)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=ROOT/'build/results')
    p.add_argument('--output',type=Path,default=ROOT/'build/figures');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6,'axes.linewidth':.5,'pdf.fonttype':42})
    data=load_data(args.results)
    for letter,draw in PANELS.items():
        _,_,w,h=LAYOUT[letter]
        fig=plt.figure(figsize=(8.3*w+1.05,11.7*h+.8))
        draw(fig,(.12,.16,.80,.70),data)
        fig.text(.025,.96,letter,fontsize=12,fontweight='bold',va='top')
        save(fig,args.output/f'panel_{letter}')
    print(args.output)


if __name__=='__main__':main()
