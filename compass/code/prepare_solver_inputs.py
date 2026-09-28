"""Materialize the deposited CPM matrix and the fixed BI13 subset for COMPASS."""
from pathlib import Path
import gzip, shutil
import pandas as pd
R=Path(__file__).resolve().parents[1]
out=R/'solver_inputs';out.mkdir(exist_ok=True)
with gzip.open(R/'data/all21_expression_cpm.tsv.gz','rb') as source, (out/'all21.tsv').open('wb') as target:
    shutil.copyfileobj(source,target)
matrix=pd.read_csv(out/'all21.tsv',sep='\t',index_col=0)
meta=pd.read_csv(R/'data/bi13_metadata.tsv',sep='\t')
matrix.loc[:,meta.compass_sample_id].to_csv(out/'bi13.tsv',sep='\t',float_format='%.8g')
print(out)
