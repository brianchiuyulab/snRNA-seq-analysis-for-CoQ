"""Reproduce donor-level statistics from the deposited COMPASS penalties."""
from pathlib import Path
from itertools import combinations, permutations
import json, hashlib, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, t, pearsonr
import statsmodels.api as sm

R=Path(__file__).resolve().parents[1];D=R/'data';T=R/'tables';T.mkdir(exist_ok=True)
RX={'COQ3m_pos':'CoQ (COQ3)','PDHm_pos':'PDH','ACITL_pos':'ACLY','ACS_pos':'ACS (AACS/ACSS2)',
    'CSm_pos':'Citrate synthase','ICDHy_pos':'IDH1','ICDHyrm_pos':'IDH2','SUCD1m_pos':'SDH'}
AGE_RX=['COQ3m_pos','ACITL_pos','ACS_pos','PDHm_pos','CSm_pos','ACONTm_pos','ICDHxm_pos',
        'ICDHyrm_pos','AKGDm_pos','SUCOASm_pos','SUCOAS1m_pos','SUCD1m_pos','FUMm_pos','MDHm_pos','PCm_pos']
def bh(p):
    p=np.asarray(p,float);ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1);return q
def save(df,name):df.to_json(T/(name+'.json'),orient='records',indent=2,double_precision=15)
def load_scores(name):
    a=pd.read_csv(D/name,sep='\t',index_col=0)
    assert a.index.is_unique and a.columns.is_unique
    assert np.isfinite(a).all().all() and a.min().min()>=-1e-9
    return -np.log1p(a.clip(lower=0))
def exact_rank(a,b):
    r=rankdata(np.round(np.r_[a,b],9));ix=np.array(list(combinations(range(len(r)),len(a))))
    null=r[ix].sum(1);obs=r[:len(a)].sum()
    return min(1.,2*min(np.mean(null<=obs+1e-10),np.mean(null>=obs-1e-10)))
def rank_effect(a,b):
    diff=np.round(a,9)[:,None]-np.round(b,9)[None,:]
    return float(np.sign(diff).mean())
def main():
    meta=pd.read_csv(D/'all21_metadata.tsv',sep='\t');inv=pd.read_csv(D/'donor_inventory.tsv',sep='\t').set_index('donor_id')
    cover=pd.read_csv(D/'model_coverage.tsv',sep='\t').set_index('donor_id')
    for c in ['Age','Sex','Cohort','Barthel_Index_BI','sites','original_BI_group']:meta[c]=meta.donor_id.map(inv[c])
    meta['coverage']=meta.donor_id.map(cover.n_recon2_genes_detected)
    meta['old']=meta.age_group.eq('Old>=74').astype(int);meta['asian']=meta.Cohort.eq('Asian Chinese').astype(int)
    meta['age_label']=np.where(meta.old,'Older','Young')
    meta['COQ8A_group']='';meta['cutoff_CPM']=np.nan
    for age,idx in meta.groupby('old').groups.items():
        cut=meta.loc[idx,'COQ8A_CPM'].median()
        meta.loc[idx,'cutoff_CPM']=cut;meta.loc[idx,'COQ8A_group']=np.where(meta.loc[idx,'COQ8A_CPM']>cut,'High','Low')
    bmeta=pd.read_csv(D/'bi13_metadata.tsv',sep='\t')
    meta['in_BI13']=meta.donor_id.isin(bmeta.donor_id)
    assert len(meta)==21 and meta.donor_id.is_unique and meta.n_nuclei.sum()==2878
    assert meta.old.value_counts().to_dict()=={1:14,0:7}
    save(meta,'donors')
    scores=load_scores('all21_penalties.tsv.gz');bi=load_scores('bi13_penalties.tsv.gz')
    old=bmeta.loc[bmeta.age_group.eq('Old>=74'),'compass_sample_id'].tolist()
    young=bmeta.loc[bmeta.age_group.eq('Young<=46'),'compass_sample_id'].tolist()
    assert set(x.split('__')[1] for x in old)=={'P3','P23','P29','P17','P21','P27'}
    assert len(young)==7 and len(old)==6
    # Selected-reaction optimization uses the complete network. Check it agrees with full-model output.
    common=scores.index.intersection(bi.index)
    agreement=float(np.max(np.abs(scores.loc[common,bi.columns].values-bi.loc[common].values)))
    assert agreement<1e-5,agreement
    v=bi[old+young].to_numpy();rank=rankdata(np.round(v,9),axis=1);ix=np.array(list(combinations(range(13),6)))
    ps=[]
    for j in range(0,len(rank),256):
        r=rank[j:j+256];null=r[:,ix].sum(2);obs=r[:,:6].sum(1)[:,None]
        ps.extend(np.minimum(1.,2*np.minimum(np.mean(null<=obs+1e-10,axis=1),np.mean(null>=obs-1e-10,axis=1))))
    testable=np.ptp(np.round(v,9),axis=1)>0
    age=pd.DataFrame({'reaction':bi.index,'p_raw':ps,'testable':testable,'q_bh_6533':1.,
         'median_older':np.median(v[:,:6],axis=1),'median_young':np.median(v[:,6:],axis=1)})
    age.loc[testable,'q_bh_6533']=bh(age.loc[testable,'p_raw'])
    age['median_difference']=age.median_older-age.median_young
    age['rank_biserial_older_minus_young']=[rank_effect(a[:6],a[6:]) for a in v]
    sd=np.sqrt((5*v[:,:6].var(1,ddof=1)+6*v[:,6:].var(1,ddof=1))/11)
    age['cohens_d_older_minus_young']=np.divide(v[:,:6].mean(1)-v[:,6:].mean(1),sd,out=np.zeros(len(v)),where=sd>0)
    md=pd.read_csv(D/'reaction_metadata.csv.gz').set_index('rxn_code_nodirection')
    base=age.reaction.str.replace(r'_(pos|neg)$','',regex=True)
    for c in ['rxn_name_long','subsystem','genes_associated_with_rxn','rxn_formula']:age[c]=base.map(md[c])
    save(age,'age_all_reactions');save(age[age.reaction.isin(AGE_RX)],'age_selected_reactions')
    save(age[age.subsystem.eq('Ubiquinone synthesis')],'age_all_ubiquinone_reactions')
    bi.T.rename_axis('compass_sample_id').reset_index().to_json(T/'bi13_scores.json',orient='records',double_precision=15)
    scores.T.rename_axis('compass_sample_id').reset_index().to_json(T/'all21_scores.json',orient='records',double_precision=15)
    corr=[];partial=[];split=[];inter=[]
    for scope,b in [('all21',meta),('young7',meta[meta.old==0]),('old14',meta[meta.old==1])]:
        rng=np.random.default_rng(27092026)
        perm=np.array(list(permutations(range(len(b))))) if len(b)<=7 else np.argsort(rng.random((49999,len(b))),axis=1)
        xr=rankdata(b.COQ8A_CPM);xp=xr-xr.mean()
        for rx,label in RX.items():
            y=scores.loc[rx,b.compass_sample_id].to_numpy();yr=rankdata(y);yp=yr-yr.mean()
            r=float(np.corrcoef(xr,yr)[0,1]);null=xp[perm]@yp/np.sqrt((xp@xp)*(yp@yp))
            cnt=int((np.abs(null)>=abs(r)-1e-12).sum());p=cnt/len(perm) if len(b)<=7 else (cnt+1)/(len(perm)+1)
            corr.append(dict(scope=scope,reaction=rx,label=label,method='Spearman',n=len(b),coefficient=r,p_raw=p))
            for method,x in [('Pearson_CPM',b.COQ8A_CPM.values),('Pearson_log1p_CPM',np.log1p(b.COQ8A_CPM.values))]:
                pr,pp=pearsonr(x,y);corr.append(dict(scope=scope,reaction=rx,label=label,method=method,n=len(b),coefficient=pr,p_raw=pp))
            for adjustment,cols in [('unadjusted',[]),('age_cohort',['old','asian']),('age_cohort_coverage',['old','asian','coverage'])]:
                X=np.column_stack([np.ones(len(b))]+[rankdata(b[c]) for c in cols]);x=xr-X@np.linalg.lstsq(X,xr,rcond=None)[0];yy=yr-X@np.linalg.lstsq(X,yr,rcond=None)[0]
                rr=float(np.corrcoef(x,yy)[0,1]);df=len(b)-np.linalg.matrix_rank(X)-1
                pp=float(2*t.sf(abs(rr)*np.sqrt(df/max(1-rr*rr,1e-15)),df))
                partial.append(dict(scope=scope,reaction=rx,label=label,adjustment=adjustment,coefficient=rr,p_raw_t_approx=pp,df=df))
            if scope!='all21':
                high=b[b.COQ8A_group=='High'];low=b[b.COQ8A_group=='Low'];a=scores.loc[rx,high.compass_sample_id].values;z=scores.loc[rx,low.compass_sample_id].values
                split.append(dict(scope=scope,reaction=rx,label=label,n_high=len(a),n_low=len(z),cutoff_CPM=float(b.cutoff_CPM.iloc[0]),
                    median_high_minus_low=float(np.median(a)-np.median(z)),p_raw=exact_rank(a,z),rank_biserial=rank_effect(a,z)))
    c=pd.DataFrame(corr);c['q_bh_72']=bh(c.p_raw);save(c,'correlations')
    p=pd.DataFrame(partial);p['q_bh_72']=bh(p.p_raw_t_approx);save(p,'partial_correlations')
    g=pd.DataFrame(split);g['q_bh_16']=bh(g.p_raw);save(g,'high_low')
    for scale,xx in [('CPM',meta.COQ8A_CPM.values),('log1p_CPM',np.log1p(meta.COQ8A_CPM.values))]:
        x=(xx-xx.mean())/xx.std(ddof=1)
        for adj,extra in [('cohort',['asian']),('cohort_coverage',['asian','coverage'])]:
            X=pd.DataFrame({'intercept':np.ones(len(meta)),'coq':x,'old':meta.old,'coq_x_old':x*meta.old})
            for col in extra:vv=meta[col].values;X[col]=(vv-vv.mean())/vv.std(ddof=1)
            for rx,label in RX.items():
                fit=sm.OLS(scores.loc[rx,meta.compass_sample_id].values,X).fit(cov_type='HC3',use_t=True);ci=fit.conf_int().loc['coq_x_old']
                inter.append(dict(reaction=rx,label=label,scale=scale,adjustment=adj,beta=fit.params['coq_x_old'],p_raw=fit.pvalues['coq_x_old'],ci_low=ci.iloc[0],ci_high=ci.iloc[1]))
    it=pd.DataFrame(inter);it['q_bh_32']=bh(it.p_raw);save(it,'age_interactions')
    # Independent comparison against the previously generated results; tolerance allows JSON rounding only.
    prior=pd.read_json(R/'provenance/pearson_spearman_age_20260927__correlations.json')
    joined=c.merge(prior,on=['scope','reaction','method'],suffixes=('','_prior'))
    assert len(joined)==72
    assert np.allclose(joined.coefficient,joined.correlation_coefficient,atol=1e-8)
    assert np.allclose(joined.p_raw,joined.p_raw_prior,atol=1e-8)
    audit={'n_donors':21,'n_nuclei':int(meta.n_nuclei.sum()),'bi13_n_young':7,'bi13_n_older':6,
           'n_age_testable':int(testable.sum()),'full_selected_score_max_abs_difference':agreement,
           'historical_72_correlations_reproduced':True,'seed':27092026,
           'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    assert int(testable.sum())==6533
    (R/'provenance/reproduction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
