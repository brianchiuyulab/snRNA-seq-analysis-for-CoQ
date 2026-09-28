"""Summarize the bounded negative control without counting resamples as donors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=R/'data/depth_control';O=R/'tables/depth_control';O.mkdir(exist_ok=True)
RX={'COQ3m_pos':'CoQ synthesis','ACITL_pos':'ACLY','PDHm_pos':'PDH','SUCD1m_pos':'SDH'}
meta=pd.read_json(P/'profiles.json').set_index('profile')
raw=pd.concat([pd.read_csv(P/b/'reactions.tsv',sep='\t',index_col=0) for b in ['pilot','remaining']],axis=1)
assert raw.columns.is_unique and set(raw.columns)==set(meta.index)
assert set(RX)<=set(raw.index) and np.isfinite(raw.values).all()
assert raw.min().min()>=-1e-9, 'Negative reaction penalties'
(R/'figures').mkdir(exist_ok=True)
s=-np.log1p(raw.clip(lower=0));ref=pd.read_json(R/'tables/bi13_scores.json').set_index('compass_sample_id')
maxdiff=max(abs(s.loc[rx,d+'__full__0']-ref.loc['MuSC__'+d,rx]) for d in meta.donor.unique() for rx in RX)
assert maxdiff<1e-8,maxdiff
rows=[]
for profile,m in meta.iterrows():
    for rx,label in RX.items():
        base=s.loc[rx,m.donor+'__full__0'];score=s.loc[rx,profile]
        rows.append(dict(profile=profile,donor=m.donor,age=m.age,condition=m.condition,replicate=m.replicate,
            reaction=rx,label=label,baseline_score=base,score=score,change_vs_full=score-base,
            total_umi=m.total_umi,n_genes_detected=m.n_genes_detected,n_sampled_nuclei=m.n_sampled_nuclei))
details=pd.DataFrame(rows);details.to_json(O/'all_results.json',orient='records',indent=2)
agg=details[details.condition!='full'].groupby(['donor','age','condition','reaction','label']).agg(
    baseline_score=('baseline_score','first'),mean_score=('score','mean'),min_score=('score','min'),max_score=('score','max'),
    mean_change=('change_vs_full','mean'),min_change=('change_vs_full','min'),max_change=('change_vs_full','max')).reset_index()
agg.to_json(O/'donor_changes.json',orient='records',indent=2)
rr=[];dm=pd.read_json(R/'tables/donors.json').set_index('compass_sample_id').loc[ref.index]
for (condition,rx),b in agg.groupby(['condition','reaction']):
    gap=ref.loc[dm.old.eq(1),rx].mean()-ref.loc[dm.old.eq(0),rx].mean()
    rr.append({'condition':condition,'reaction':rx,'n_donors':len(b),'n_mean_decreased':int((b.mean_change<0).sum()),
      'median_change':float(b.mean_change.median()),'range_donor_mean_change':[float(b.mean_change.min()),float(b.mean_change.max())],
      'original_age_mean_difference':float(gap),'median_technical_drop_over_original_age_drop':float(b.mean_change.median()/gap),
      'young_mean_change':float(b[b.age=='Young<=46'].mean_change.mean()),'older_mean_change':float(b[b.age=='Old>=74'].mean_change.mean())})
summary={'n_profiles':len(meta),'replicates_per_condition':3,'full_reference_max_abs_difference':maxdiff,
 'wall_seconds':{},'results':rr,'interpretation':'Within-donor input reduction can produce lower scores. Ratios compare magnitudes only; they do not estimate the proportion of biological age effect caused by depth.'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,4,figsize=(12,7.2));fig.subplots_adjust(left=.075,right=.98,bottom=.10,top=.85,hspace=.55,wspace=.40)
for row,condition in enumerate(['nuclei5','umi10pct']):
    for col,(rx,label) in enumerate(RX.items()):
        ax=axes[row,col];b=agg[(agg.condition==condition)&(agg.reaction==rx)]
        for j,r in enumerate(b.sort_values('donor').itertuples()):
            co='#287B9F' if r.age=='Young<=46' else '#C25B32';offset=(j-(len(b)-1)/2)*.013
            ax.plot([0+offset,1+offset],[r.baseline_score,r.mean_score],color=co,alpha=.65,lw=.9,marker='o',ms=3)
            ax.plot([1+offset]*2,[r.min_score,r.max_score],color=co,lw=1.3)
        ax.set_xticks([0,1],['Full input','5 nuclei' if row==0 else '10% UMI']);ax.set_xlim(-.15,1.15);ax.set_title(f'{label} | n={len(b)} donors')
        if col==0:ax.set_ylabel('COMPASS score')
        ax.grid(axis='y',color='#E7EAED',lw=.5);ax.set_axisbelow(True)
fig.suptitle('Within-donor data reduction lowers predicted reaction scores',fontsize=15,y=.97)
fig.text(.5,.91,'Young: blue   |   Older: orange   |   Reduced input: mean and range of 3 resamples',ha='center',fontsize=10)
fig.savefig(R/'figures/depth_control.png',dpi=220);fig.savefig(R/'figures/depth_control.svg');plt.close(fig)
print('Depth control: 79 profiles; full-reference agreement verified.')
