import os, glob
import numpy as np
import anndata as ad

ROOT = r"C:\Users\User\Desktop\Single cell for CoQ\Data_raw\step1_out_v2\counts_h5ad"
fs = glob.glob(os.path.join(ROOT, "*.counts.h5ad"))

print("n_files =", len(fs))
show = fs  # 先檢查all

for f in show:
    a = ad.read_h5ad(f)  # 不用 backed，抽樣比較穩
    X = a.X

    # sparse matrix -> 用 X.data；dense -> flatten
    data = X.data if hasattr(X, "data") else np.ravel(X)

    if data.size == 0:
        print(os.path.basename(f), "EMPTY")
        continue

    # 抽樣看小數部分（只看非零）
    n = min(200000, data.size)
    idx = np.random.choice(data.size, size=n, replace=False)
    samp = data[idx]

    frac = np.abs(samp - np.round(samp))
    max_frac = float(frac.max())

    print(os.path.basename(f),
          "shape=", a.shape,
          "dtype=", samp.dtype,
          "max_frac=", max_frac)
