"""All-reaction rankings and donor-level subsystem summaries for BI13."""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

R=Path(__file__).resolve().parents[1];T=R/'tables';F=R/'figures'
RX={'COQ3m_pos':'CoQ synthesis (COQ3)','ACITL_pos':'ACLY','PDHm_pos':'PDH','ACS_pos':'ACS (AACS/ACSS2)',
 'CSm_pos':'Citrate synthase','ICDHyrm_pos':'IDH2','SUCD1m_pos':'SDH','ICDHxm_pos':'IDH3','ACONTm_pos':'Aconitase'}
FOCAL=['Ubiquinone synthesis','Citric acid cycle','Pyruvate metabolism','Glycolysis/gluconeogenesis',
 'Oxidative phosphorylation','Pentose phosphate pathway','Fatty acid oxidation','Fatty acid synthesis']
def dump(df,name):df.to_json(T/(name+'.json'),orient='records',indent=2,double_precision=15)
def fmt(p):return f'{p:.4f}' if p>=.001 else f'{p:.2g}'
def rank_interval(lo,hi):return str(int(lo)) if lo==hi else f'{int(lo):,}-{int(hi):,}'
def main():
    d=pd.read_json(T/'age_all_reactions.json');v=d[d.testable].copy()
    # Do not impose a hidden ordering within a p-value tie block.
    for pfx,col,asc in [('p','p_raw',True),('q','q_bh_6533',True),('effect_abs','cohens_d_older_minus_young',False)]:
        s=v[col].abs() if pfx=='effect_abs' else v[col]
        v[pfx+'_rank_first']=s.rank(method='min',ascending=asc).astype(int)
        v[pfx+'_rank_last']=s.rank(method='max',ascending=asc).astype(int)
    v['p_rank_percentile_end']=100*v.p_rank_last/len(v)
    v['direction']=np.where(v.median_difference<0,'Older lower',np.where(v.median_difference>0,'Older higher','Equal medians'))
    allrank=d.merge(v[['reaction']+[c for c in v.columns if 'rank' in c and c!='rank_biserial_older_minus_young']+['direction']],on='reaction',how='left')
    dump(allrank.sort_values(['testable','p_raw','reaction'],ascending=[False,True,True]),'global_reaction_ranking')
    focal=v.set_index('reaction').loc[list(RX)].reset_index();focal['label']=focal.reaction.map(RX);dump(focal,'global_focal_reaction_ranking')
    meta=pd.read_json(T/'donors.json').set_index('compass_sample_id')
    scores=pd.read_json(T/'bi13_scores.json').set_index('compass_sample_id')
    meta=meta.loc[scores.index].sort_values(['old','donor_id']);scores=scores.loc[meta.index]
    old=np.flatnonzero(meta.old.values==1);young=np.flatnonzero(meta.old.values==0)
    assert len(old)==6 and len(young)==7
    ix=np.array(list(combinations(range(13),6)))
    rows=[];profiles={}
    for pathway,b in d.groupby('subsystem',sort=True):
        variable=b[b.testable];n=len(variable)
        if n:
            # Each reaction contributes one within-reaction donor percentile-rank profile.
            per=(rankdata(np.round(scores[variable.reaction].to_numpy().T,9),axis=1)-1)/12
            profile=per.mean(axis=0);profiles[pathway]=profile
            obs=profile[old].mean()-profile[young].mean()
            sums=profile[ix].sum(1);null=sums/6-(profile.sum()-sums)/7
            p=min(1.,2*min(np.mean(null<=obs+1e-12),np.mean(null>=obs-1e-12)))
        else:profile=np.repeat(.5,13);obs=0.;p=1.
        rows.append({'pathway':pathway,'n_all_reactions':len(b),'n_variable_reactions':n,
          'n_reactions_q05_older_lower':int(((variable.q_bh_6533<.05)&(variable.median_difference<0)).sum()),
          'n_reactions_q05_older_higher':int(((variable.q_bh_6533<.05)&(variable.median_difference>0)).sum()),
          'fraction_reactions_q05':float((variable.q_bh_6533<.05).mean()) if n else 0.,
          'median_reaction_cohens_d':float(variable.cohens_d_older_minus_young.median()) if n else 0.,
          'mean_rank_index_older':float(profile[old].mean()),'mean_rank_index_young':float(profile[young].mean()),
          'older_minus_young_rank_index':float(obs),'pathway_p_raw':float(p),'pathway_q_bh90':1.})
    pw=pd.DataFrame(rows);mask=pw.n_variable_reactions>0;assert mask.sum()==90
    pw.loc[mask,'pathway_q_bh90']=multipletests(pw.loc[mask,'pathway_p_raw'],method='fdr_bh')[1]
    # The exact probability grid (multiples of 1/1716) provides stable tie handling.
    for pfx,col in [('p','pathway_p_raw'),('q','pathway_q_bh90')]:
        pw.loc[mask,pfx+'_rank_first']=pw.loc[mask,col].round(12).rank(method='min').astype(int)
        pw.loc[mask,pfx+'_rank_last']=pw.loc[mask,col].round(12).rank(method='max').astype(int)
    pw=pw.sort_values(['pathway_p_raw','pathway']);dump(pw,'global_pathway_ranking')
    pd.DataFrame(profiles,index=meta.index).rename_axis('compass_sample_id').reset_index().to_json(T/'global_pathway_donor_indices.json',orient='records',indent=2,double_precision=15)
    valid=pw[pw.n_variable_reactions>0]
    summary={'n_total_reactions':len(d),'n_variable_reactions':len(v),'n_constant_reactions':len(d)-len(v),
      'reaction_q05':int((v.q_bh_6533<.05).sum()),'reaction_q05_older_lower':int(((v.q_bh_6533<.05)&(v.median_difference<0)).sum()),
      'n_total_subsystems':len(pw),'n_testable_subsystems':len(valid),'pathway_q05':int((valid.pathway_q_bh90<.05).sum()),
      'pathway_definition':'Per donor, average of within-reaction percentile ranks over all variable directed reactions assigned to the subsystem; exact donor-label permutation of the older-minus-young mean index; BH across 90 subsystems',
      'age_comparison':'BI13: Young7 vs Older6; donor is the biological unit'}
    (T/'global_ranking_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    draw(v,pw,focal,meta,profiles)
    write_report(summary,focal,pw)
    print(json.dumps(summary,indent=2))
    print(focal[['label','p_raw','q_bh_6533','p_rank_first','p_rank_last']].to_string(index=False))
    print(pw[pw.pathway.isin(FOCAL)][['pathway','pathway_p_raw','pathway_q_bh90','p_rank_first','p_rank_last']].to_string(index=False))

def draw(v,pw,focal,meta,profiles):
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':10,'axes.labelsize':8,
      'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
      'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    layout=[]
    with PdfPages(F/'COMPASS_global_ranking.pdf') as pdf:
        def save(fig,name):
            fig.canvas.draw();rr=fig.canvas.get_renderer();bad=[]
            for tx in fig.findobj(matplotlib.text.Text):
                if not tx.get_visible() or not tx.get_text():continue
                bb=tx.get_window_extent(rr)
                if bb.width and bb.height and (bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5):bad.append(tx.get_text())
            layout.append({'figure':name,'text_outside_canvas':bad})
            pdf.savefig(fig);fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=220);plt.close(fig)
        fig=plt.figure(figsize=(9,7.7));fig.text(.06,.962,'Genome-wide reaction ranking | Young 7 vs Older 6',fontsize=13,weight='bold')
        ax=fig.add_axes([.11,.57,.83,.29]);vv=v.sort_values('p_raw').reset_index(drop=True)
        ax.plot(np.arange(1,len(vv)+1),vv.q_bh_6533,color='#6C899B',lw=1.6)
        ax.axhline(.05,color='#AA4B32',ls='--',lw=.8);ax.set_yscale('log');ax.set_ylim(.02,1.2);ax.set_xlim(1,6533)
        ax.set_xticks([1,1000,2000,3000,4000,5000,6000]);ax.yaxis.set_major_locator(FixedLocator([.03,.05,.1,.5,1.]));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Reaction rank by raw p value (ties occupy intervals)');ax.set_ylabel('BH-adjusted p (q)')
        ax.text(.98,.13,'3,722 / 6,533 reactions: q < 0.05\nAll 3,722 have lower older-group medians',transform=ax.transAxes,ha='right',fontsize=8)
        rr=focal.set_index('reaction').loc['COQ3m_pos'];ax.axvspan(rr.p_rank_first,rr.p_rank_last,color='#CC7041',alpha=.14)
        ax.annotate('CoQ: tied ranks 1,970-3,082\nq = 0.02965',xy=((rr.p_rank_first+rr.p_rank_last)/2,rr.q_bh_6533),xytext=(.22,.80),textcoords='axes fraction',fontsize=9,
          arrowprops={'arrowstyle':'-','color':'#CC7041'})
        ax=fig.add_axes([.24,.105,.35,.32]);f=focal.iloc[::-1];yy=np.arange(len(f));color=np.where(f.q_bh_6533<.05,'#287B9F','#9AA5AC')
        ax.scatter(f.cohens_d_older_minus_young,yy,c=color,s=43);ax.axvline(0,color='#777777',lw=.6)
        ax.set_yticks(yy,f.label);ax.set_xlabel("Cohen's d (Older - Young)");ax.set_ylim(-.7,len(f)-.3)
        ax.set_title('Focal reaction effects',loc='left',pad=12);ax.grid(axis='x',color='#E4E9ED',lw=.4)
        aa=fig.add_axes([.62,.105,.35,.32]);aa.set_xlim(0,1);aa.set_ylim(-.7,len(f)-.3);aa.axis('off')
        for xx,lab in [(.04,'p'),(.30,'BH q'),(.72,'p-rank interval')]:aa.text(xx,len(f)+.05,lab,ha='center',weight='bold',fontsize=8)
        for j,r in enumerate(f.itertuples()):
            for xx,tx in [(.04,fmt(r.p_raw)),(.30,fmt(r.q_bh_6533)),(.72,rank_interval(r.p_rank_first,r.p_rank_last))]:aa.text(xx,j,tx,ha='center',va='center',fontsize=8)
        save(fig,'Figure_global_reaction_ranks')
        # Every testable subsystem is shown; transport and unassigned sets are explicitly retained.
        valid=pw[pw.n_variable_reactions>0].copy()
        cmap=plt.get_cmap('RdBu_r');norm=TwoSlopeNorm(vmin=-.65,vcenter=0,vmax=.65)
        for half in range(2):
            b=valid.iloc[half*45:(half+1)*45].iloc[::-1]
            fig=plt.figure(figsize=(9,11.5));fig.text(.045,.969,f'All subsystem summaries | {half*45+1}-{(half+1)*45} of 90',fontsize=13,weight='bold')
            ax=fig.add_axes([.40,.105,.31,.79]);yy=np.arange(len(b));sizes=16+100*b.fraction_reactions_q05
            im=ax.scatter(b.older_minus_young_rank_index,yy,c=b.older_minus_young_rank_index,cmap=cmap,norm=norm,s=sizes,edgecolor='#526471',linewidth=.35)
            ax.axvline(0,color='#999999',lw=.6);ax.set_ylim(-1,len(b));ax.set_xlim(-.65,.12)
            labels=[('CoQ / Ubiquinone synthesis' if s=='Ubiquinone synthesis' else s.strip()) for s in b.pathway]
            ax.set_yticks(yy,labels);ax.tick_params(axis='y',length=0,labelsize=6.8)
            for tx,name in zip(ax.get_yticklabels(),b.pathway):
                if name in FOCAL:tx.set_fontweight('bold');tx.set_color('#975233')
            ax.set_xlabel('Pathway rank index: Older - Young');ax.grid(axis='x',lw=.4,color='#E5E9EC');ax.set_axisbelow(True)
            aa=fig.add_axes([.74,.105,.235,.79]);aa.set_ylim(-1,len(b));aa.set_xlim(0,1);aa.axis('off')
            for x,lab in [(.1,'n rxn'),(.43,'BH q'),(.80,'p rank')]:aa.text(x,len(b)+.5,lab,ha='center',fontsize=7,weight='bold')
            for j,r in enumerate(b.itertuples()):
                for x,tx in [(.1,str(r.n_variable_reactions)),(.43,fmt(r.pathway_q_bh90)),(.80,rank_interval(r.p_rank_first,r.p_rank_last))]:aa.text(x,j,tx,ha='center',va='center',fontsize=6.6)
            ax.set_title('Color: pathway difference   |   Size: fraction of reactions with q < 0.05',loc='right',pad=24,fontsize=8)
            for sz,lab in [(16,'0%'),(66,'50%'),(116,'100%')]:ax.scatter([],[],s=sz,facecolor='white',edgecolor='#526471',label=lab)
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.065),ncol=3,frameon=False,fontsize=7,title='Reaction-level significant fraction',title_fontsize=7)
            save(fig,f'Figure_global_pathways_{half+1}')
        # Donor heatmap for all 90 ranked summaries, with matched row order and donor annotations.
        names=valid.pathway.tolist();mat=np.array([profiles[x] for x in names]);fig=plt.figure(figsize=(10,14.5))
        fig.text(.05,.975,'All subsystem donor profiles | within-reaction rank index',fontsize=13,weight='bold')
        ax=fig.add_axes([.40,.075,.47,.86]);im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_yticks(range(90),[x.strip() for x in names],fontsize=6);ax.set_xticks(range(13),meta.donor_id,rotation=90);ax.tick_params(length=0)
        ax.axvline(6.5,color='white',lw=1.5);ax.text(3,-1.5,'Young (n=7)',ha='center',color='#287B9F');ax.text(9.5,-1.5,'Older (n=6)',ha='center',color='#BB5730')
        for tx,name in zip(ax.get_yticklabels(),names):
            if name in FOCAL:tx.set_fontweight('bold');tx.set_color('#975233')
        cb=fig.add_axes([.91,.39,.016,.22]);fig.colorbar(im,cax=cb,label='Mean within-reaction donor percentile rank',ticks=[0,.5,1])
        save(fig,'Figure_global_pathway_heatmap')
    (R/'provenance/global_ranking_layout.json').write_text(json.dumps(layout,indent=2)+'\n')

def write_report(s,f,pw):
    lines=['# 全代謝模型排名：原 BI13 年輕7 vs 老年6','',
      '## 全反應概況','',f"模型共{s['n_total_reactions']:,}個有方向反應，{s['n_variable_reactions']:,}個有變異、可檢驗；另外{s['n_constant_reactions']:,}個為固定值，保留在完整表但不列推論排名。",'',
      f"全反應BH校正後，{s['reaction_q05']:,}個反應q<0.05，這些反應的老年組中位數皆較低。CoQ屬於廣泛下降的一部分，並非排名最前或唯一改變的路徑。",'',
      '## 焦點反應排名','',
      '主要排名依雙側原始p由小到大，同p者列同名次涵蓋區間。完整表另保存BH q排名和絕對Cohen\'s d效應量排名；三種定義不可混用。', '',
      '|反應|原始p|BH q（6,533反應）|p排名區間|Older−Young Cohen\'s d|','|---|---:|---:|---|---:|']
    for r in f.itertuples():lines.append(f'|{r.label}|{r.p_raw:.6f}|{r.q_bh_6533:.6f}|{rank_interval(r.p_rank_first,r.p_rank_last)}|{r.cohens_d_older_minus_young:.3f}|')
    lines+=['','同名次很多，是因為13位donor的精確置換p值離散，以及模型反應相互耦合；不能把區間內隨意排序說成唯一的精確名次。CoQ末端COQ3m_pos及其他13個CoQ合成反應具有相同donor排序。',
      '', '## Pathway層級：新增的donor摘要檢驗','',
      '每個反應先將13位donor的分數轉成0–1百分位排名，再在同一RECON2 subsystem內，平均其有變異反應的排名，得到每位donor一個路徑摘要分數。比較老年6與年輕7的平均差，精確枚舉1,716種donor標籤；雙側p為兩尾較小者的兩倍。此方法維持donor為樣本，保留反應間依賴，不把每個reaction算成一個人。',
      '',f"共有{s['n_total_subsystems']}個原始subsystems，其中{s['n_testable_subsystems']}個含可檢驗反應，BH在這90個新路徑檢驗內校正；其中{s['pathway_q05']}個q<0.05。Transport、Exchange、Unassigned等模型分類也全部保留；不能把90個皆稱為狹義生化途徑。",'',
      '|Pathway|可變反應數|原始p|路徑BH q（90項）|p排名區間|','|---|---:|---:|---:|---|']
    for r in pw[pw.pathway.isin(FOCAL)].itertuples():lines.append(f'|{r.pathway}|{r.n_variable_reactions}|{r.pathway_p_raw:.6f}|{r.pathway_q_bh90:.6f}|{rank_interval(r.p_rank_first,r.p_rank_last)}|')
    lines+=['','路徑摘要是本輪新定義的整體排名指標，不是原COMPASS套件輸出的net pathway flux，也不是GSEA。它包含模型註記下的有方向反應，不代表每步都同向；原始逐反應統計完整保留。',
      '', '## 圖與檔案','',
      '- `figures/COMPASS_global_ranking.pdf`：反應排名、全90個路徑dot plot（兩頁）、全路徑donor heatmap。',
      '- `tables/global_reaction_ranking.json`：全部10,211個反應，包括常數反應與公式／gene mapping。',
      '- `tables/global_pathway_ranking.json`：全部subsystem統計與排名。',
      '- `tables/global_pathway_donor_indices.json`：13位donor路徑摘要值。',
      '- `code/global_ranking.py`：重現程式，不重跑COMPASS。',
      '', '此處仍沿用固定註記與原BI13，未調整cohort或基因覆蓋度；廣泛下降不證明CoQ特異性或因果性。既有覆蓋度敏感度結論不變。']
    (R/'manuscript/GLOBAL_RANKING_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':main()
