"""Independent statistical and input checks for the generated result tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import permutation_test, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

R=Path(__file__).resolve().parents[1]
T=R/'tables'
m=pd.read_json(T/'donors.json').set_index('compass_sample_id')
s=pd.read_json(T/'bi13_scores.json').set_index('compass_sample_id')
a=pd.read_json(T/'age_all_reactions.json')
old=m.loc[s.index,'old'].eq(1)
x=s.loc[old,'COQ3m_pos'].to_numpy()
y=s.loc[~old,'COQ3m_pos'].to_numpy()
def rank_sum(x,y):
    return rankdata(np.round(np.concatenate([x,y]),9))[:len(x)].sum()
independent=permutation_test((x,y),rank_sum,permutation_type='independent',
                             n_resamples=np.inf,alternative='two-sided',vectorized=False)
coq=a.set_index('reaction').loc['COQ3m_pos']
assert abs(independent.pvalue-coq.p_raw)<1e-12
families=[('age_all_reactions','p_raw','q_bh_6533','testable'),
          ('correlations','p_raw','q_bh_72',None),
          ('partial_correlations','p_raw_t_approx','q_bh_72',None),
          ('high_low','p_raw','q_bh_16',None),
          ('age_interactions','p_raw','q_bh_32',None),
          ('global_pathway_ranking','pathway_p_raw','pathway_q_bh90','n_variable_reactions')]
family_sizes={}
for name,p,q,mask in families:
    z=pd.read_json(T/(name+'.json'))
    if mask: z=z[z[mask].astype(bool)]
    assert np.allclose(multipletests(z[p],method='fdr_bh')[1],z[q],rtol=0,atol=1e-12),name
    family_sizes[name]=len(z)
scores=pd.read_json(T/'all21_scores.json').set_index('compass_sample_id')
c=pd.read_json(T/'correlations.json')
for r in c[c.method.eq('Spearman')].itertuples():
    b=m if r.scope=='all21' else m[m.old.eq(0 if r.scope=='young7' else 1)]
    coefficient=spearmanr(b.COQ8A_CPM,scores.loc[b.index,r.reaction]).statistic
    assert abs(coefficient-r.coefficient)<1e-12
e=pd.read_csv(R/'data/all21_expression_cpm.tsv.gz',sep='\t',index_col=0)
assert e.index.is_unique and e.columns.is_unique
assert set(e.columns)==set(m.index)==set(scores.index)
assert np.isfinite(e.to_numpy()).all() and e.to_numpy().min()>=0
assert np.allclose(e.sum(),1e6,rtol=0,atol=.02)
assert np.allclose(e.loc['COQ8A',m.index],m.COQ8A_CPM,rtol=1e-7,atol=1e-8)
membership=pd.read_csv(R/'data/frozen_musc_membership.tsv.gz',sep='\t')
assert membership.cell_id.is_unique and len(membership)==2878
counts=membership.groupby('subject_id').size()
assert np.array_equal(m.donor_id.map(counts),m.n_nuclei)
assert not membership.sample_id.isin(['om5_gm_snrna_seq_1','om9_gm_snrna_seq_1']).any()
for p in (R/'provenance').glob('*layout*.json'):
    obj=json.loads(p.read_text(encoding='utf8'))
    if isinstance(obj,list):
        for row in obj: assert not row.get('text_outside_canvas',[]),row
report={'independent_CoQ_exact_p':float(independent.pvalue),'BH_family_sizes':family_sizes,
        'all_24_Spearman_coefficients':'verified','CPM_and_COQ8A_alignment':'verified',
        'frozen_2878_nuclei_and_donor_counts':'verified','status':'pass'}
(R/'provenance/statistical_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,indent=2))
