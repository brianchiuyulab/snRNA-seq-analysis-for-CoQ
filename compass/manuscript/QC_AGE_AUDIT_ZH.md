# BI13 年齡比較：資料處理與技術混雜核對

## 結論

原始counts到CPM的計算核對通過；但是目前年齡比較有嚴重的核數／總UMI／模型基因覆蓋度不平衡，不能把廣泛下降直接解讀成已排除技術影響的老化代謝下降。原有數值及p/q沒有改變；圖稿的可重現性通過不等於生物推論的穩健性通過。本輪不改固定MuSC註記、不改BI13名單，也不改先前COQ8A/Barthel Index RNA分析數值。

## 實際檢查

1. 直接以h5py讀取原v21 H5AD的raw integer counts CSR，逐一重加2,878核、21donor的每基因counts，驗證總UMI及全部31,398列輸出CPM。所有21人均符合，差異僅%.8g寫檔捨入。
2. 每donor的輸入CPM欄總和999,999.991至1,000,000.007，確實已做總library-size normalization；不是把不同donor的raw UMI直接餵給COMPASS，也沒有先log再當linear CPM輸入。
3. 固定庫去重、donor mapping及penalty轉score方向沿用已核對結果；原BI13未直接使用OM5/OM9重複library。
4. CPM調整總量比例，不會補回少核／少UMI造成未抽到的基因；也不等於cohort、肌肉部位或核內RNA品質校正。主年齡p/q先前是未調整協變數的比較。BH只處理多重檢定，不處理這些混雜。

## BI13實際差異（各donor指標的組內中位數）

|指標|Young7|Older6|
|---|---:|---:|
|MuSC核數|65|5|
|donor總UMI|123,011|10,361|
|可偵測RECON2基因數|833|177.5|
|每核UMI中位數，再取donor中位數|2,017.5|1,880|
|每核可偵測基因中位數，再取donor中位數|762.5|631.5|
|每核粒線體比例中位數，再取donor中位數|0.172%|1.643%|

不是每顆老年核的UMI都少一個數量級；主要不平衡出現在每donor可供彙總的核數與總覆蓋量。所有donor的詳細數字保留於tables/qc_age_audit/donor_qc.json。

## 是否真的全部下降

6,533可變反應按組內中位差（1e-9容差）分為：Older低6,232個、高58個、中位數相同243個。58個較高反應沒有任何原始p<.05或BH q<.05。3,722個q<.05反應全部Older較低。故不是每個反應都低，但方向嚴重不對稱。

## 深度敏感度

在同一BI13，CoQ分數與總UMI的Spearman係數0.989，與模型基因覆蓋度0.984；全模型平均rank index與覆蓋度0.995。這些是強烈混雜警訊，不單獨證明生物效應為零。

新補測OLS（HC3穩健SE、t參考分布）保留全部13位donor，逐一加入technical covariate。CoQ年齡效應在coverage調整後beta=-0.0471、p=.231；總UMI的log1p調整後beta=-0.0451、p=.427；cohort+coverage後p=.344。全模型rank index在coverage調整後p=.481。ACLY/PDH/SDH加入coverage後也未達原始p<.05。這些是新的探索性模型，不能把p當成原秩和p的直接重算；未套用舊反應q，也不宣稱已充分校正。

## 輸入量負對照與解讀

同 donor 的 79 個輸入量負對照已完成，見 DEPTH_CONTROL_RESULTS_ZH.md。5 核條件的 9 位 donor 與 10% UMI 條件的 13 位 donor，四個受測反應的平均分數皆降低。資料減量因此能造成反應分數下降；生物年齡效應與取樣量的貢獻仍待分離。上述 OLS 模型與負對照分別評估統計調整及輸入量敏感度。

計算方法與來源：code/qc_age_audit.py、code/depth_control.py、tables/qc_age_audit、tables/depth_control。
