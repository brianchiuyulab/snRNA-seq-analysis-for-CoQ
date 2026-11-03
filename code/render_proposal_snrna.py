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

types=["Type I","Type II","Specialized MF","MuSC","FAP","EC","SMC","Adipocyte","Myeloid cell","Lymphocyte"]
palette=["#377eb8","#e41a1c","#984ea3","#a65628","#ff7f00","#4daf4a","#66c2a5","#e6ab02","#f781bf","#999999","#888888"]
p=Panel("5A","Nuclear annotations",165,100)
im=np.asarray(Image.open(root/"figures/Fig05_Celltype_annotation_UMAP.png").convert("RGB"))
dark=im.max(2)<120
ys=np.where(dark.mean(1)>.45)[0];xs=np.where(dark.mean(0)>.60)[0]
if not(len(ys) and len(xs)):raise ValueError("Could not locate source UMAP axes")
ax=p.ax(12,14,69,73);ax.imshow(im[ys[0]+2:ys[-1]-2,xs[0]+2:xs[-1]-2],aspect="equal");ax.set(xticks=[],yticks=[],xlabel="UMAP 1",ylabel="UMAP 2")
for i,(label,color) in enumerate(zip(types+["Unknown"],palette)):
    p.key(86,17+7.8*i,color,label)
p.save()
p=Panel("5B","Cell-type marker expression",308,114)
v=tab("celltype_marker_dotplot_values.tsv.gz")
genes=v.gene.drop_duplicates().tolist()
assert len(genes)==28 and set(v.cell_type)==set(types)
ax=p.ax(54,14,203,69)
sc=ax.scatter(v.gene.map(dict(zip(genes,range(len(genes))))),v.cell_type.map(dict(zip(types,range(10)))),
              s=1+15*v.pct_expressing,c=v.mean_z_by_gene,cmap="RdBu_r",vmin=-2,vmax=2,edgecolors=".5",linewidths=.15)
ax.set(xticks=range(len(genes)),yticks=range(10),xlim=(-.5,27.5),ylim=(9.5,-.5))
ax.set_xticklabels(genes,rotation=90);ax.set_yticklabels(types);ax.tick_params(length=0)
p.text(261,21,"Mean z-score");cb=p.fig.colorbar(sc,cax=p.ax(263,26,6,24),ticks=[-2,0,2]);cb.ax.tick_params(length=1)
p.text(261,61,"Expressing")
for i,pc in enumerate([10,50,90]):
    p.text(275,71+i*10,f"{pc}%")
    m=p.ax(261,64+i*10,10,10);m.scatter([0],[0],s=1+15*pc/100,facecolors="none",edgecolors=".3",lw=.5);m.axis("off")
p.save()
types=["MuSC","Type I","Type II","Specialized MF"]
short=["MuSC","Type I","Type II","Sp. MF"]
groups=["Young","Older, BI=100","Older, BI<100"];gc=["#3971ad","#d48a12","#b94444"]
stats=tab("COQ_statistics.tsv");pb=tab("COQ_donor_pseudobulk.tsv.gz")
p=Panel("5C","COQ pathway vs young donors",223,118)
v=stats[stats.cell_type.isin(types)].copy();genes=v.gene.drop_duplicates().tolist()
ax=p.ax(36,17,111,86);threshold=v.p_adjusted_bh_within_gene_celltype<.05
sig=-np.log10(v.p_adjusted_bh_within_gene_celltype.clip(lower=1e-12));lim=max(1,np.nanpercentile(abs(v.log2_fold_change),95))
xx=v.cell_type.map(dict(zip(types,np.arange(4)*2)))+v.comparison.str.contains("<").astype(int)
yy=v.gene.map(dict(zip(genes,range(len(genes)))))
sc=ax.scatter(xx,yy,s=3+8*sig.clip(upper=4),c=v.log2_fold_change,cmap="RdBu_r",vmin=-lim,vmax=lim,edgecolors=np.where(threshold,"#99005c","#777"),lw=np.where(threshold,.85,.2))
ax.set(xlim=(-.55,7.55),ylim=(12.5,-.5),xticks=np.arange(8),yticks=range(len(genes)));ax.set_xticklabels(["=","<"]*4);ax.set_yticklabels(genes);ax.tick_params(length=0)
for i,s in enumerate(short):p.text(36+(2*i+1)*111/8,116,s,ha="center")
p.text(147,21,"log2 fold change");p.fig.colorbar(sc,cax=p.ax(150,27,8,22),ticks=[-int(lim),0,int(lim)])
p.text(147,62,"Larger dot: lower q");p.text(147,74,"Pink ring: q < 0.05",color="#99005c");p.text(147,89,"=   older BI = 100");p.text(147,101,"<   older BI < 100");p.save()
p=Panel("5D","COQ8A expression by donor",249,118)
ax=p.ax(31,25,213,72)
for i,cell in enumerate(types):
    for j,g in enumerate(groups):
        v=pb[(pb.cell_type==cell)&(pb.group==g)&(pb.gene=="COQ8A")].sort_values("donor").log1p_cpm.dropna()
        if len(v)==0:continue
        x=i+(j-1)*.23
        ax.boxplot(v,positions=[x],widths=.18,patch_artist=True,showfliers=False,boxprops={"facecolor":gc[j],"alpha":.2,"linewidth":.5},medianprops={"color":"black","lw":.7},whiskerprops={"lw":.5},capprops={"lw":.5})
        ax.scatter(x+np.linspace(-.052,.052,len(v)),v,s=5,c=gc[j],linewidths=0)
ax.set(xticks=range(4),xlim=(-.5,3.55),ylim=(-.18,7.4),yticks=[0,2,4,6],ylabel="ln(CPM + 1)");ax.set_xticklabels(short)
for j,(s,c) in enumerate(zip(["Young","Older BI = 100","Older BI < 100"],gc)):p.key([25,81,160][j],20,c,s)
adj=stats[(stats.cell_type=="MuSC")&(stats.gene=="COQ8A")].p_adjusted_bh_within_gene_celltype
assert len(adj)==2 and np.allclose(adj,.033,atol=.001)
ax.text(-.32,6.55,"Both q = 0.033",fontsize=7);p.save()
