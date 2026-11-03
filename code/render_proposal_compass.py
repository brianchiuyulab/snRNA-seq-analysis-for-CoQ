"""Render proposal panels at their printed sizes. Input is the analysis release directory."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--release", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root, out = args.release, args.out
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"Arial","font.size":7,"axes.linewidth":.55,
    "xtick.major.size":2,"ytick.major.size":2,"xtick.major.width":.5,"ytick.major.width":.5,
    "xtick.major.pad":1.3,"ytick.major.pad":1.4,"axes.labelpad":1.7,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none",
    "axes.unicode_minus":False,"legend.fontsize":7,"legend.frameon":False})
class Panel:
    def __init__(self, label, title, width, height):
        self.label,self.w,self.h=label,width,height
        self.fig=plt.figure(figsize=(width/72,height/72))
        self.text(0,9,label[-1],9,weight="bold")
        self.text(12,9,title,8,weight="bold")
    def text(self,x,y,s,size=7,**kw):
        return self.fig.text(x/self.w,1-y/self.h,s,fontsize=size,**kw)
    def ax(self,x,y,w,h):
        return self.fig.add_axes([x/self.w,1-(y+h)/self.h,w/self.w,h/self.h])
    def save(self):
        for ext in ("pdf","svg"):
            self.fig.savefig(out/f"{self.label}.{ext}",dpi=420,facecolor="white")
        plt.close(self.fig)
    def key(self,x,y,color,label):
        self.text(x,y,"●",7,color=color)
        self.text(x+7,y,label)
def tab(name):
    return pd.read_csv(root/"tables"/name,sep="\t")

donors=tab("COMPASS_CoQ_donor_audit_20261102.tsv").sort_values(["age_label","donor_id"],key=lambda s:s.map({"Young":0,"Older":1}) if s.name=="age_label" else s)
assert donors.groupby("age_label").size().to_dict()=={"Older":6,"Young":7}
bi=pd.read_json(root/"tables/bi13_scores.json").set_index("compass_sample_id").loc[donors.compass_sample_id]
rx=["COQ3m_pos","PDHm_pos","ACITL_pos","ACS_pos","CSm_pos","ACONTm_pos","ICDHxm_pos","ICDHyrm_pos","AKGDm_pos","SUCOASm_pos","SUCOAS1m_pos","SUCD1m_pos","FUMm_pos","MDHm_pos","PCm_pos"]
labels=["CoQ synthesis","PDH","ACLY","ACS","Citrate synthase","Aconitase","IDH3","IDH2","OGDH","SCS (ATP)","SCS (GTP)","SDH","Fumarase","MDH2","Pyruvate carboxylase"]
colors=["#247a9e","#c25b32"]
p=Panel("8A","MuSC reaction potential by donor",283,199)
x0,width=73,198
p.text(x0+width*3.5/13,20,"Young (n = 7)",ha="center",color=colors[0],weight="bold")
p.text(x0+width*10/13,20,"Older (n = 6)",ha="center",color=colors[1],weight="bold")
for j,(lab,vals) in enumerate([("Cohort",donors.cohort.map({"European":"EU","Asian Chinese":"AC"})),("BI",donors.barthel_index.map(lambda x:"–" if pd.isna(x) else str(int(x)))),("Nuclei",donors.n_nuclei)]):
    y=31+j*9;p.text(x0-3,y,lab,ha="right")
    for i,val in enumerate(vals):p.text(x0+(i+.5)*width/13,y,str(val),ha="center")
v=bi[rx].T;z=v.sub(v.mean(axis=1),axis=0).div(v.std(axis=1,ddof=0).replace(0,np.nan),axis=0)
ax=p.ax(x0,53,width,111);im=ax.imshow(z,aspect="auto",vmin=-2,vmax=2,cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#dddddd"),interpolation="nearest")
ax.axvline(6.5,c="white",lw=1);ax.set(yticks=range(15),xticks=range(13));ax.set_yticklabels(labels);ax.set_xticklabels(donors.donor_id,rotation=90);ax.tick_params(length=0)
p.text(2,188,"Within-reaction z-score");p.fig.colorbar(im,cax=p.ax(85,180,60,4),orientation="horizontal",ticks=[-2,0,2]);p.text(165,188,"EU: European   AC: Asian Chinese")
p.save()
p=Panel("8B","CoQ biosynthesis in MuSCs",190,191)
ax=p.ax(41,28,141,111);assert np.allclose(bi.COQ3m_pos,donors.score)
for j,age in enumerate(["Young","Older"]):
    v=donors[donors.age_label==age].sort_values("score")
    for i,r in enumerate(v.itertuples()):
        ax.scatter(j+[-.08,.03,.11,-.03,.07,-.1,.0][i],r.score,s=21,marker=r.marker,c=colors[j],edgecolors="white",lw=.3)
    ax.plot([j-.25,j+.25],[v.score.median()]*2,c="#252525",lw=1.2,zorder=0)
ax.set(xlim=(-.47,1.47),xticks=[0,1],ylabel="COMPASS score",ylim=(-11.47,-10.76))
ax.set_xticklabels(["Young (7)","Older (6)"]);p.text(42,22,"Permutation P = 0.014; BH q = 0.030")
p.text(16,162,"● European     ▲ Asian Chinese");p.text(16,175,"Horizontal line: median");p.text(16,187,"Score = −ln(1 + reaction penalty)")
p.save()
