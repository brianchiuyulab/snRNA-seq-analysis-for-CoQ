"""Create a small offline entry page for the manuscript and figures."""
from pathlib import Path
import html,re
R=Path(__file__).resolve().parents[1]
cards=[]
names=[('Figure_1_age','Figure 1 · 年齡差異：Young 7 / Older 6'),('Figure_2_associations','Figure 2 · 分年齡 COQ8A 關聯：Young 7 / Older 14'),
       ('Figure_S1_high_low','Figure S1 · 年齡內高低分組'),('Figure_S2_network_CoQ','Figure S2 · 全模型與14個 CoQ 反應'),('Figure_S3_sensitivity','Figure S3 · 統計敏感度')]
names += [('Figure_global_reaction_ranks','全反應排名'),('Figure_global_pathways_1','全路徑 dot plot：前半'),('Figure_global_pathways_2','全路徑 dot plot：後半'),('Figure_global_pathway_heatmap','全路徑 donor heatmap')]
names += [('depth_control','資料量負對照：固定 donor 的反應分數變化')]
for name,label in names:
    cards.append(f'<section><h2>{label}</h2><a href="figures/{name}.png"><img src="figures/{name}.png" alt="{label}"></a><p><a href="figures/{name}.svg">SVG 向量圖</a></p></section>')
links=''.join(f'<li><a href="manuscript/{f}">{label}</a></li>' for f,label in [
 ('ANALYSIS_WORKFLOW.md','分析流程：從輸入到結果'),('REVIEW_STATUS.md','程式與結果核對'),('SUMMARY_ZH.md','中文結論摘要'),('FIGURE_SOURCE_MAP.md','圖表、資料與程式對照'),('DEPTH_CONTROL_RESULTS_ZH.md','資料量負對照結果'),('RESULTS_AND_DISCUSSION.md','Results & Discussion'),('METHODS.md','Methods'),('FIGURE_LEGENDS.md','Figure legends'),('CODE_AVAILABILITY.md','Code & data availability')])
page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>COQ8A · MuSC · COMPASS</title>
<style>body{max-width:960px;margin:36px auto;padding:0 20px;font:16px/1.7 Arial,"Microsoft JhengHei",sans-serif;color:#243341;background:#f4f6f8}h1{font-size:30px}h2{font-size:21px}a{color:#126980}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}code{background:#e6edf1;padding:3px 6px}small{color:#5c6972}</style>
<h1>COQ8A / MuSC / COMPASS</h1><p>探索性年齡比較、COQ8A 關聯與資料量敏感度。原始 counts 與 CPM 已核對；生物年齡差異與取樣量的貢獻尚待分離。</p>
<section><h2>論文與程式</h2><ul>'''+links+'''<li><a href="figures/COMPASS_figures.pdf">完整五頁圖稿 PDF</a></li><li><a href="figures/COMPASS_global_ranking.pdf">全反應與路徑圖稿 PDF</a></li><li><a href="README.md">重現方式與檔案索引</a></li></ul><p><code>python code/reproduce.py</code></p></section>'''+''.join(cards)+'''<small>每位 donor 為一個生物重複；統計定義與完整校正範圍見 Methods 及圖說。</small></html>'''
(R/'index.html').write_text(page,encoding='utf8')
