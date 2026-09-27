"""Audit BI13 normalization and age/depth confounding without changing the roster."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr,rankdata
R=Path(__file__).resolve().parents[1];O=R/'tables/qc_age_audit';O.mkdir(exist_ok=True)
m=pd.read_json(R/'tables/donors.json').set_index('compass_sample_id')
s=pd.read_json(R/'tables/bi13_scores.json').set_index('compass_sample_id');m=m.loc[s.index].copy()
e=pd.read_csv(R/'data/all21_expression_cpm.tsv.gz',sep='\t',index_col=0)
a=pd.read_json(R/'tables/age_all_reactions.json');v=a[a.testable]
qc=pd.read_csv(R/'data/frozen_musc_membership.tsv.gz',sep='\t')
for col in ['total_counts','n_genes_by_counts','pct_counts_mt','pct_counts_ribo']:
    m['median_nucleus_'+col]=m.donor_id.map(qc.groupby('subject_id')[col].median())
m['log_umi']=np.log1p(m.pseudobulk_umi);m['log_nuclei']=np.log1p(m.n_nuclei)
var=v.reaction.tolist();m['global_rank_index']=((rankdata(np.round(s[var].values.T,9),axis=1)-1)/12).mean(0)
rows=[]
for target in ['global_rank_index','COQ3m_pos','ACITL_pos','PDHm_pos','SUCD1m_pos']:
    y=m[target].values if target in m else s[target].values
    for cols in [[],['asian'],['coverage'],['log_umi'],['log_nuclei'],['asian','coverage']]:
        X=pd.DataFrame({'const':np.ones(len(m)),'old':m.old.values})
        for c in cols:X[c]=(m[c].values-m[c].mean())/m[c].std(ddof=1)
        fit=sm.OLS(y,X).fit(cov_type='HC3',use_t=True)
        rows.append({'target':target,'covariates':'+'.join(cols) or 'none','older_minus_young_beta':fit.params['old'],'p_HC3':fit.pvalues['old']})
corr=[]
for target in ['global_rank_index','COQ3m_pos']:
    y=m[target] if target in m else s[target]
    for c in ['coverage','log_umi','log_nuclei','median_nucleus_total_counts','median_nucleus_n_genes_by_counts','median_nucleus_pct_counts_mt']:
        r,p=spearmanr(y,m[c]);corr.append({'target':target,'covariate':c,'coefficient':r,'p_approx':p})
pd.DataFrame(rows).to_json(O/'age_adjustment.json',orient='records',indent=2)
pd.DataFrame(corr).to_json(O/'technical_correlations.json',orient='records',indent=2)
m.reset_index().to_json(O/'donor_qc.json',orient='records',indent=2)
up=v[v.median_difference>1e-9].sort_values('p_raw');up.to_json(O/'older_higher_reactions.json',orient='records',indent=2)
summary={'cpm_column_sum_min':float(e.sum().min()),'cpm_column_sum_max':float(e.sum().max()),
 'median_direction_counts':{'older_lower':int((v.median_difference < -1e-9).sum()),'older_higher':int((v.median_difference>1e-9).sum()),'tied':int((v.median_difference.abs()<=1e-9).sum())},
 'higher_q05':int((up.q_bh_6533<.05).sum()),'higher_p05':int((up.p_raw<.05).sum()),
 'group_medians':m.groupby('age_label')[['n_nuclei','pseudobulk_umi','coverage','median_nucleus_total_counts','median_nucleus_n_genes_by_counts','median_nucleus_pct_counts_mt']].median().to_dict(orient='index')}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));print(pd.DataFrame(corr).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
print(up[['reaction','rxn_name_long','p_raw','q_bh_6533']].head(8).to_string(index=False))
