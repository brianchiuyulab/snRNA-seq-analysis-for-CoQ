"""Publication figures. All markers represent donors or explicitly labelled reactions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

R=Path(__file__).resolve().parents[1];T=R/'tables';F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':9,'axes.labelsize':7.5,
 'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,
 'ytick.major.width':.6,'savefig.facecolor':'white'})
Y='#247A9E';O='#C25B32';AS='#7763A6';EU='#4B9A82';GRAY='#D4D9DD'
CM=plt.get_cmap('RdBu_r')
RX={'COQ3m_pos':'CoQ synthesis','PDHm_pos':'PDH','ACITL_pos':'ACLY','ACS_pos':'ACS',
    'CSm_pos':'Citrate synthase','ICDHy_pos':'IDH1','ICDHyrm_pos':'IDH2','SUCD1m_pos':'SDH'}
MORE={**RX,'ACONTm_pos':'Aconitase','ICDHxm_pos':'IDH3','AKGDm_pos':'OGDH','SUCOASm_pos':'SCS (ATP)',
      'SUCOAS1m_pos':'SCS (GTP)','FUMm_pos':'Fumarase','MDHm_pos':'MDH2','PCm_pos':'Pyruvate carboxylase'}
AGE_RX=['COQ3m_pos','PDHm_pos','ACITL_pos','ACS_pos','CSm_pos','ACONTm_pos','ICDHxm_pos','ICDHyrm_pos',
        'AKGDm_pos','SUCOASm_pos','SUCOAS1m_pos','SUCD1m_pos','FUMm_pos','MDHm_pos','PCm_pos']
meta=pd.read_json(T/'donors.json').set_index('compass_sample_id')
bi=pd.read_json(T/'bi13_scores.json').set_index('compass_sample_id')
allsc=pd.read_json(T/'all21_scores.json').set_index('compass_sample_id')
age=pd.read_json(T/'age_all_reactions.json').set_index('reaction')
corr=pd.read_json(T/'correlations.json');split=pd.read_json(T/'high_low.json')
partial=pd.read_json(T/'partial_correlations.json');interaction=pd.read_json(T/'age_interactions.json')
layout=[]
def pv(p):return f'{p:.3f}' if p>=.001 else f'{p:.1e}'
def title(fig,num,text):
    fig.text(.045,.975,num,fontsize=12,fontweight='bold',va='top')
    fig.text(.12,.975,text,fontsize=10,va='top')
def panel(ax,letter):ax.text(-min(.17,.065/ax.get_position().width),1.08,letter,transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')
def clean(ax):ax.tick_params(length=3);ax.grid(axis='y',color='#E9ECEF',linewidth=.4);ax.set_axisbelow(True)
def cor(scope,rx,method='Spearman'):return corr[(corr.scope==scope)&(corr.reaction==rx)&(corr.method==method)].iloc[0]
def points(ax,b,rx,score,x,col,width=.14):
    # Deterministic vertical-neighborhood swarm, with cohort-specific shapes.
    vals=score.loc[b.index,rx].values;span=max(np.ptp(vals),.05)
    placed=[]
    for k in np.argsort(vals):
        yy=vals[k];choices=[0,.07,-.07,.14,-.14,.21,-.21,.28,-.28]
        xx=next((v for v in choices if all(abs(yy-py)>span*.07 or abs(v-px)>=.069 for px,py in placed)),choices[-1])
        placed.append((xx,yy));row=b.iloc[k]
        ax.scatter(x+xx,yy,s=19,marker='o' if row.Cohort=='European' else '^',color=col,edgecolor='white',linewidth=.45,zorder=3)
    ax.plot([x-.24,x+.24],[np.median(vals)]*2,color='#20252B',lw=1.2,zorder=4)
def save(fig,name,pdf):
    fig.canvas.draw()
    # Check all visible text against the figure canvas; review overlaps visually after rendering.
    renderer=fig.canvas.get_renderer();bounds=fig.bbox;outside=[]
    for tx in fig.findobj(matplotlib.text.Text):
        if tx.get_visible() and tx.get_text():
            box=tx.get_window_extent(renderer)
            if box.width and box.height and (box.x0<-.5 or box.y0<-.5 or box.x1>bounds.width+.5 or box.y1>bounds.height+.5):outside.append(tx.get_text())
    layout.append({'figure':name,'text_outside_canvas':outside,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight()})
    pdf.savefig(fig)
    fig.savefig(F/(name+'.svg'))
    fig.savefig(F/(name+'.png'),dpi=240)
    plt.close(fig)
def main1(pdf):
    fig=plt.figure(figsize=(7.2,9.5));title(fig,'1','Age-associated metabolic reaction scores in MuSCs')
    b=meta.loc[bi.index].sort_values(['old','donor_id']);ids=b.index
    # Annotated donor heatmap with individual reaction standardization, never cross-reaction raw-score comparison.
    ax=fig.add_axes([.24,.525,.62,.32]);v=bi.loc[ids,AGE_RX].T
    z=(v-v.mean(axis=1).values[:,None])/v.std(axis=1,ddof=0).replace(0,np.nan).values[:,None]
    im=ax.imshow(z,aspect='auto',cmap=CM,vmin=-2,vmax=2,interpolation='nearest')
    ax.set_yticks(range(len(v)),[MORE[r] for r in AGE_RX]);ax.set_xticks(range(len(b)),b.donor_id,rotation=90)
    ax.tick_params(length=0,pad=3);ax.axvline(6.5,color='white',lw=1.5)
    panel(ax,'A')
    cb=fig.add_axes([.90,.625,.014,.16]);fig.colorbar(im,cax=cb,label='Within-reaction z score',ticks=[-2,0,2])
    tracks=[('Age',b.old.values,ListedColormap([Y,O]),0,1),
            ('Cohort',b.asian.values,ListedColormap([EU,AS]),0,1),
            ('BI',b.Barthel_Index_BI.values,plt.get_cmap('YlGnBu').with_extremes(bad='#E6E8EB'),0,100),
            ('Nuclei',np.log10(b.n_nuclei.values),plt.get_cmap('Greys'),0,3.1)]
    for k,(lab,arr,cm,vmin,vmax) in enumerate(tracks):
        aa=fig.add_axes([.24,.861+(3-k)*.016,.62,.012]);aa.imshow(np.asarray(arr)[None,:],aspect='auto',cmap=cm,vmin=vmin,vmax=vmax)
        aa.set_axis_off();aa.text(-.015,.5,lab,ha='right',va='center',transform=aa.transAxes,fontsize=7)
        if lab in ['BI','Nuclei']:
            texts=['NA' if pd.isna(x) else str(int(x)) for x in (b.Barthel_Index_BI if lab=='BI' else b.n_nuclei)]
            for j,tx in enumerate(texts):
                value=arr[j];color='white' if pd.notna(value) and value>(70 if lab=='BI' else 1.8) else '#30363B'
                aa.text(j,0,tx,ha='center',va='center',fontsize=6.1,color=color)
    fig.legend(handles=[Line2D([],[],color=Y,lw=5,label='Young (n=7)'),Line2D([],[],color=O,lw=5,label='Older (n=6)'),
      Line2D([],[],color=EU,lw=5,label='European'),Line2D([],[],color=AS,lw=5,label='Asian Chinese')],loc='upper center',bbox_to_anchor=(.53,.946),ncol=4,frameon=False,fontsize=6.7,handlelength=1)
    for i,rx in enumerate(['COQ3m_pos','ACITL_pos','ACS_pos','CSm_pos','ICDHyrm_pos','SUCD1m_pos']):
        row,col=divmod(i,3);aa=fig.add_axes([.105+col*.305,.285-row*.225,.225,.15]);clean(aa)
        for k,co in [(0,Y),(1,O)]:points(aa,b[b.old==k],rx,bi,k,co)
        aa.set_xticks([0,1],['Young','Older']);aa.set_xlim(-.5,1.5);aa.set_title(RX[rx],pad=18)
        rr=age.loc[rx];aa.text(.5,1.045,f'p = {pv(rr.p_raw)}   q = {pv(rr.q_bh_6533)}',transform=aa.transAxes,ha='center',fontsize=6.6)
        if col==0:aa.set_ylabel('COMPASS score')
        panel(aa,chr(66+i))
    save(fig,'Figure_1_age',pdf)
def main2(pdf):
    fig=plt.figure(figsize=(7.2,9.5));title(fig,'2','COQ8A-metabolism associations within age groups')
    handles=[Line2D([],[],marker='o',color=Y,lw=0,label='Young (n=7)'),Line2D([],[],marker='o',color=O,lw=0,label='Older (n=14)'),
             Line2D([],[],marker='o',color='#444444',lw=0,label='European'),Line2D([],[],marker='^',color='#444444',lw=0,label='Asian Chinese')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.945),ncol=4,frameon=False,fontsize=7)
    for i,rx in enumerate(['COQ3m_pos','PDHm_pos','ACITL_pos','SUCD1m_pos']):
        row,col=divmod(i,2);ax=fig.add_axes([.11+col*.475,.66-row*.31,.365,.17]);clean(ax);panel(ax,chr(65+i))
        for old,scope,color in [(0,'young7',Y),(1,'old14',O)]:
            b=meta[meta.old==old]
            for cohort,mk in [('European','o'),('Asian Chinese','^')]:
                bb=b[b.Cohort==cohort];ax.scatter(np.log1p(bb.COQ8A_CPM),allsc.loc[bb.index,rx],s=24,marker=mk,color=color,edgecolor='white',lw=.5,zorder=3)
            cc=cor(scope,rx);ax.text(0,1.25-.115*old,f'{"Older" if old else "Young"}: coefficient = {cc.coefficient:.2f}; p = {pv(cc.p_raw)}',
                transform=ax.transAxes,color=color,fontsize=7.1)
        ax.set_title(RX[rx],loc='left',pad=43);ax.set_xlabel('COQ8A (CPM; log1p axis)');ax.set_ylabel('COMPASS score')
        ax.set_xticks(np.log1p([0,10,30,100,300]),['0','10','30','100','300'])
    ax=fig.add_axes([.14,.115,.80,.105]);panel(ax,'E');rxs=list(RX)
    vals=np.array([[cor(sc,r).coefficient for r in rxs] for sc in ['young7','old14']])
    im=ax.imshow(vals,aspect='auto',cmap=CM,vmin=-1,vmax=1)
    ax.set_yticks([0,1],['Young','Older']);ax.set_xticks(range(8),[RX[r] for r in rxs],rotation=30,ha='right');ax.tick_params(length=0)
    for row,sc in enumerate(['young7','old14']):
        for col,rx in enumerate(rxs):
            cc=cor(sc,rx);color='white' if abs(cc.coefficient)>.58 else '#1B242C'
            ax.text(col,row,f'{cc.coefficient:.2f}\np={pv(cc.p_raw)}',ha='center',va='center',fontsize=6.1,color=color)
    ax.set_title('Spearman coefficient and permutation p value',loc='left',pad=10)
    cb=fig.add_axes([.35,.050,.35,.010]);fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[-1,0,1],label='Spearman coefficient')
    save(fig,'Figure_2_associations',pdf)
def supplement1(pdf):
    fig=plt.figure(figsize=(7.2,9.5));title(fig,'S1','Within-age COQ8A high-low donor comparisons')
    fig.legend(handles=[Line2D([],[],marker='o',color=Y,lw=0,label='Young: low 4 / high 3'),Line2D([],[],marker='o',color=O,lw=0,label='Older: low 7 / high 7')],
       loc='upper center',bbox_to_anchor=(.53,.944),ncol=2,frameon=False,fontsize=7)
    for i,rx in enumerate(RX):
        row,col=divmod(i,2);ax=fig.add_axes([.11+col*.48,.715-row*.205,.36,.13]);clean(ax);panel(ax,chr(65+i))
        for old,scope,color in [(0,'young7',Y),(1,'old14',O)]:
            for hi in [0,1]:
                b=meta[(meta.old==old)&(meta.COQ8A_group==('High' if hi else 'Low'))];points(ax,b,rx,allsc,old*3+hi,color)
            s=split[(split.scope==scope)&(split.reaction==rx)].iloc[0]
            ax.text(.02+.52*old,1.05,f'p={pv(s.p_raw)}; q={pv(s.q_bh_16)}',transform=ax.transAxes,color=color,fontsize=6.3)
        ax.set_xticks([0,1,3,4],['Y low','Y high','O low','O high']);ax.set_xlim(-.5,4.5);ax.set_title(RX[rx],pad=22);ax.set_ylabel('COMPASS score')
    save(fig,'Figure_S1_high_low',pdf)
def supplement2(pdf):
    fig=plt.figure(figsize=(7.2,9.5));title(fig,'S2','Network context and ubiquinone synthesis reactions')
    ax=fig.add_axes([.12,.65,.80,.23]);panel(ax,'A')
    a=age[age.testable];sig=a.q_bh_6533<.05
    ax.scatter(a.cohens_d_older_minus_young,-np.log10(a.q_bh_6533),s=3,c=np.where(sig,np.where(a.cohens_d_older_minus_young<0,Y,O),'#C9CED3'),alpha=.40,rasterized=True)
    ax.axvline(0,color='#666666',lw=.6);ax.axhline(-np.log10(.05),color='#555555',lw=.6,ls='--')
    ax.set_xticks(np.arange(np.ceil(a.cohens_d_older_minus_young.min()),np.floor(a.cohens_d_older_minus_young.max())+1))
    ax.set_xlabel("Cohen's d (Older - Young)");ax.set_ylabel('-log10(BH q)');ax.set_title('All 6,533 variable reactions',loc='left')
    for rx,offset in [('COQ3m_pos',(14,7)),('ACITL_pos',(15,-16)),('SUCD1m_pos',(-40,16))]:
        rr=age.loc[rx];xx=rr.cohens_d_older_minus_young;yy=-np.log10(rr.q_bh_6533)
        ax.scatter([xx],[yy],s=20,c='#222222',zorder=5);ax.annotate(RX[rx],(xx,yy),xytext=offset,textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','lw':.5})
    ub=age[age.subsystem.eq('Ubiquinone synthesis')];b=meta.loc[bi.index].sort_values(['old','donor_id'])
    v=bi.loc[b.index,ub.index].T;z=(v-v.mean(1).values[:,None])/v.std(1,ddof=0).replace(0,np.nan).values[:,None]
    ax=fig.add_axes([.235,.155,.54,.34]);panel(ax,'B');im=ax.imshow(z,cmap=CM,vmin=-2,vmax=2,aspect='auto')
    ax.set_yticks(range(len(ub)),ub.index.str.replace('_pos','',regex=False));ax.set_xticks(range(len(b)),b.donor_id,rotation=90);ax.tick_params(length=0);ax.axvline(6.5,color='white',lw=1.5)
    ax.set_title('Ubiquinone synthesis: all 14 modeled reactions',loc='left',pad=25)
    ax.text(3,-1,'Young (n=7)',ha='center',color=Y);ax.text(9.5,-1,'Older (n=6)',ha='center',color=O)
    aa=fig.add_axes([.81,.155,.16,.34]);aa.set_xlim(0,1);aa.set_ylim(len(ub)-.5,-.5);aa.set_axis_off()
    aa.text(.1,-1,'p',ha='center');aa.text(.66,-1,'BH q',ha='center')
    for j,rr in enumerate(ub.itertuples()):aa.text(.1,j,pv(rr.p_raw),ha='center',va='center',fontsize=6.8);aa.text(.66,j,pv(rr.q_bh_6533),ha='center',va='center',fontsize=6.8)
    cb=fig.add_axes([.31,.067,.36,.012]);fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[-2,0,2],label='Within-reaction z score')
    save(fig,'Figure_S2_network_CoQ',pdf)
def supplement3(pdf):
    fig=plt.figure(figsize=(7.2,9.5));title(fig,'S3','Correlation and age-interaction sensitivity analyses')
    # Age-specific correlation across three methods: all values retained regardless of p.
    ax=fig.add_axes([.20,.60,.75,.27]);panel(ax,'A')
    cols=[(sc,method) for sc in ['young7','old14'] for method in ['Spearman','Pearson_CPM','Pearson_log1p_CPM']]
    vals=np.array([[cor(sc,rx,method).coefficient for sc,method in cols] for rx in RX]);im=ax.imshow(vals,aspect='auto',cmap=CM,vmin=-1,vmax=1)
    ax.set_yticks(range(8),list(RX.values()));ax.set_xticks(range(6),['Y Spearman','Y Pearson','Y Pearson log','O Spearman','O Pearson','O Pearson log'],rotation=30,ha='right');ax.tick_params(length=0)
    for i,rx in enumerate(RX):
        for j,(sc,method) in enumerate(cols):
            rr=cor(sc,rx,method);ax.text(j,i,f'{rr.coefficient:.2f} / {pv(rr.p_raw)}',ha='center',va='center',fontsize=6,color='white' if abs(rr.coefficient)>.58 else '#222222')
    ax.set_title('Correlation coefficient / p value',loc='left',pad=14)
    ax=fig.add_axes([.20,.255,.35,.205]);panel(ax,'B');cols=['unadjusted','age_cohort','age_cohort_coverage']
    a=partial[partial.scope=='all21'];vals=np.array([[a[(a.reaction==rx)&(a.adjustment==c)].iloc[0].coefficient for c in cols] for rx in RX])
    ax.imshow(vals,cmap=CM,vmin=-1,vmax=1,aspect='auto');ax.set_yticks(range(8),list(RX.values()));ax.set_xticks(range(3),['None','Age + cohort','+ coverage'],rotation=35,ha='right');ax.tick_params(length=0)
    for i,rx in enumerate(RX):
        for j,c in enumerate(cols):
            rr=a[(a.reaction==rx)&(a.adjustment==c)].iloc[0];ax.text(j,i,f'{rr.coefficient:.2f}\np={pv(rr.p_raw_t_approx)}',ha='center',va='center',fontsize=6,color='white' if abs(rr.coefficient)>.58 else '#222222')
    ax.set_title('All 21 donors: partial Spearman',loc='left',pad=15,fontsize=8)
    aa=fig.add_axes([.69,.255,.26,.205]);panel(aa,'C');sub=interaction[(interaction.scale=='CPM')&(interaction.adjustment=='cohort')].set_index('reaction').loc[list(RX)]
    yy=np.arange(8);aa.errorbar(sub.beta,yy,xerr=np.vstack([sub.beta-sub.ci_low,sub.ci_high-sub.beta]),fmt='o',color='#425B6F',ms=3,lw=.8,capsize=2)
    aa.axvline(0,color='#888888',ls='--',lw=.6);aa.set_yticks(yy,[]);aa.set_ylim(7.5,-.5);aa.set_xlabel('Age interaction (95% CI)');aa.set_title('COQ8A x age',loc='left',pad=15,fontsize=8)
    for j,rr in enumerate(sub.itertuples()):aa.text(.98,j,f'p={pv(rr.p_raw)}',transform=aa.get_yaxis_transform(),ha='right',va='center',fontsize=5.8,bbox={'facecolor':'white','edgecolor':'none','pad':.2})
    cb=fig.add_axes([.30,.12,.40,.012]);fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[-1,0,1],label='Correlation coefficient')
    save(fig,'Figure_S3_sensitivity',pdf)
def main():
    with PdfPages(F/'COMPASS_figures.pdf',metadata={'Title':'COQ8A and MuSC metabolism','Author':'','Subject':'Donor-level COMPASS analysis'}) as pdf:
        for fn in [main1,main2,supplement1,supplement2,supplement3]:fn(pdf)
    (R/'provenance/layout_check.json').write_text(json.dumps(layout,indent=2)+'\n')
    print(json.dumps(layout,indent=2))
if __name__=='__main__':main()
