# 同 donor 輸入量負對照

固定原BI13與MuSC註記；只計算CoQ、ACLY、PDH、SDH。共 79 個輸入 profiles。

完整輸入13profiles；核數>=10的9donor，各隨機取5核3次，共27profiles；全部13donor各在原pseudobulk raw UMI做binomial(p=.1)稀釋3次，共39profiles。每個profile都重新轉CPM。重抽樣是技術重複，不當新增donor。

原完整資料重跑与原BI13四反應的最大分數差：0。

|條件|反應|donor平均分數下降人數|donor下降中位數|原Older−Young平均差|
|---|---|---:|---:|---:|
|nuclei5|ACLY|9/9|-0.2292|-0.4350|
|nuclei5|CoQ synthesis|9/9|-0.1975|-0.2948|
|nuclei5|PDH|9/9|-0.3269|-0.4207|
|nuclei5|SDH|9/9|-0.2328|-0.3015|
|umi10pct|ACLY|13/13|-0.1477|-0.4350|
|umi10pct|CoQ synthesis|13/13|-0.1823|-0.2948|
|umi10pct|PDH|13/13|-0.2144|-0.4207|
|umi10pct|SDH|13/13|-0.1581|-0.3015|

負對照顯示，同一 donor 在較少核數或 UMI 輸入下可出現較低 COMPASS 分數。生物年齡差異與取樣量的貢獻仍待分離；本測試評估輸入量敏感度，並未估計校正後的年齡效應。
