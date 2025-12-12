import os
import re
import tarfile
import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad


# ==============================================================================
# USER CONFIG
# ==============================================================================
ROOT = r"C:\Users\User\Desktop\Single cell for CoQ"
PROCESS_DIR = os.path.join(ROOT, "Process_version")
TABLE1_XLSX = os.path.join(ROOT, "Table1.xlsx")

OUT_ROOT = os.path.join(ROOT, "Data_raw", "step1_out_v2")
OUT_H5AD_DIR = os.path.join(OUT_ROOT, "counts_h5ad")
os.makedirs(OUT_H5AD_DIR, exist_ok=True)

MANIFEST_TSV = os.path.join(OUT_ROOT, "sample_manifest.tsv")
FAIL_TSV = os.path.join(OUT_ROOT, "failures.tsv")


# ==============================================================================
# HELPERS
# ==============================================================================
def read_table1_snRNA_subjects(xlsx_path: str) -> list[str]:
    """
    Table1.xlsx 的結構你那份是：
    - 第 0 行可能是 title / 空行
    - 第 1 行才是真正 header: Donor, Sex, Age, ..., Data
    - 第 2 行開始是資料
    我們只抓 Data 欄位包含 'snRNA-seq' 的 Donor，得到 22 位。
    """
    raw = pd.read_excel(xlsx_path, header=None)
    header = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy()
    df.columns = header
    df = df[~df["Donor"].isna()].copy()

    df["Donor"] = df["Donor"].astype(str)
    df["Data"] = df["Data"].astype(str)

    sn = df[df["Data"].str.contains("snRNA-seq", case=False, na=False)]
    donors = sorted(sn["Donor"].unique().tolist(), key=lambda x: (re.sub(r"\D", "", x) == "", x))
    return donors


def guess_subject_id_from_prefix(sample_prefix: str) -> str:
    """
    sample_prefix 會像：
    - om1_gm_snrna_seq
    - ym3_gc_snrna_seq_1
    - p21_gm_snrna_seq_6
    抓前綴 OM/YM/P + 數字
    """
    s = sample_prefix.lower()
    m = re.match(r"^(om\d+|ym\d+|p\d+)", s)
    if not m:
        return "UNKNOWN"
    return m.group(1).upper()


def guess_batch(subject_id: str) -> str:
    """
    你目前這批資料：OM/YM 是中國，P 是歐洲（依你檔名與資料包）
    """
    if subject_id.startswith("P"):
        return "Europe"
    if subject_id.startswith("OM") or subject_id.startswith("YM"):
        return "China"
    return "Unknown"


def guess_muscle_from_prefix(sample_prefix: str) -> str:
    """
    從檔名猜肌肉部位標記（僅作為 metadata，後續還可用 Table1/其他表修正）
    常見：GM, VL, TA, ST, GC
    """
    s = sample_prefix.lower()
    for tag in ["gm", "vl", "ta", "st", "gc"]:
        if f"_{tag}_" in s:
            return tag.upper()
    return "NA"


def list_triplets_in_tar(tar_path: str) -> dict[str, dict[str, str]]:
    """
    在 tar 裡掃描所有 members，用 basename 判斷 triplet：
    - *_matrix.mtx or *_matrix.mtx.gz
    - *_barcodes.tsv or *_barcodes.tsv.gz
    - *_features.tsv / *_features.tsv.gz 或 *_genes.tsv / *_genes.tsv.gz

    回傳：
    {
      "p21_gm_snrna_seq_1": {"mtx": "...member...", "barcodes": "...", "features": "..."},
      ...
    }
    """
    trip = {}
    with tarfile.open(tar_path, "r:*") as t:
        members = t.getmembers()
        for m in members:
            if not m.isfile():
                continue
            name = m.name.replace("\\", "/")
            base = name.split("/")[-1].lower()

            # matrix
            if base.endswith("_matrix.mtx") or base.endswith("_matrix.mtx.gz"):
                prefix = re.sub(r"_matrix\.mtx(\.gz)?$", "", base)
                trip.setdefault(prefix, {})["mtx"] = name

            # barcodes
            if base.endswith("_barcodes.tsv") or base.endswith("_barcodes.tsv.gz"):
                prefix = re.sub(r"_barcodes\.tsv(\.gz)?$", "", base)
                trip.setdefault(prefix, {})["barcodes"] = name

            # features / genes
            if (base.endswith("_features.tsv") or base.endswith("_features.tsv.gz")
                    or base.endswith("_genes.tsv") or base.endswith("_genes.tsv.gz")):
                prefix = re.sub(r"_(features|genes)\.tsv(\.gz)?$", "", base)
                trip.setdefault(prefix, {})["features"] = name

    # keep only complete triplets
    complete = {k: v for k, v in trip.items() if set(v.keys()) == {"mtx", "barcodes", "features"}}
    return complete


def extract_member_to_file(tar: tarfile.TarFile, member_name: str, out_path: str):
    """
    把 tar member 寫到 out_path（原樣，不解壓 gzip）
    """
    m = tar.getmember(member_name)
    f = tar.extractfile(m)
    if f is None:
        raise FileNotFoundError(f"Cannot extract member: {member_name}")
    with open(out_path, "wb") as w:
        shutil.copyfileobj(f, w)


def read_tsv_maybe_gz(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", header=None, compression="gzip")
    return pd.read_csv(path, sep="\t", header=None)


def read_mtx_maybe_gz(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            mat = scipy.io.mmread(f)
    else:
        mat = scipy.io.mmread(path)
    return mat


def load_10x_from_tempdir(tmpdir: str) -> ad.AnnData:
    """
    tmpdir 內必須有：
    - matrix.mtx 或 matrix.mtx.gz
    - barcodes.tsv 或 barcodes.tsv.gz
    - features.tsv 或 features.tsv.gz
    我們手動讀，避免 scanpy 對 features 欄位數的假設。
    """
    # locate files
    cand_mtx = [os.path.join(tmpdir, x) for x in ["matrix.mtx.gz", "matrix.mtx"] if os.path.exists(os.path.join(tmpdir, x))]
    cand_bar = [os.path.join(tmpdir, x) for x in ["barcodes.tsv.gz", "barcodes.tsv"] if os.path.exists(os.path.join(tmpdir, x))]
    cand_feat = [os.path.join(tmpdir, x) for x in ["features.tsv.gz", "features.tsv"] if os.path.exists(os.path.join(tmpdir, x))]

    if len(cand_mtx) != 1 or len(cand_bar) != 1 or len(cand_feat) != 1:
        raise FileNotFoundError(f"Missing standard 10x files in tmpdir={tmpdir}")

    mtx_path, bar_path, feat_path = cand_mtx[0], cand_bar[0], cand_feat[0]

    # read matrix (genes x cells)
    mat = read_mtx_maybe_gz(mtx_path)
    if not sp.issparse(mat):
        mat = sp.coo_matrix(mat)
    mat = mat.tocsr()

    # transpose to cells x genes
    X = mat.T.tocsr()

    # counts should be integer (mmread gives float)
    if X.dtype != np.int32 and X.dtype != np.int64:
        # 先轉成 int64 再視情況壓到 int32
        X = X.astype(np.int64)
        if X.max() < np.iinfo(np.int32).max:
            X = X.astype(np.int32)

    # barcodes
    bc = read_tsv_maybe_gz(bar_path)[0].astype(str).values

    # features
    feat = read_tsv_maybe_gz(feat_path)
    # 10x v3: gene_id, gene_symbol, feature_type ...
    if feat.shape[1] >= 2:
        gene_ids = feat[0].astype(str).values
        gene_symbols = feat[1].astype(str).values
    else:
        gene_ids = feat[0].astype(str).values
        gene_symbols = feat[0].astype(str).values

    var = pd.DataFrame(index=pd.Index(gene_symbols, name="gene_symbol"))
    var["gene_id"] = gene_ids
    var["gene_symbol"] = gene_symbols

    obs = pd.DataFrame(index=pd.Index(bc, name="barcode"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def ensure_unique_obsnames(adata: ad.AnnData, sample_id: str) -> None:
    adata.obs_names = [f"{sample_id}:{x}" for x in adata.obs_names]
    adata.obs_names_make_unique()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # 1) Table1 donors (snRNA-seq only)
    sn_subjects = read_table1_snRNA_subjects(TABLE1_XLSX)
    sn_subjects_set = set([x.upper() for x in sn_subjects])
    print(f"[INFO] Table1 snRNA subjects: {len(sn_subjects)} -> {sn_subjects}")

    # 2) tar list
    tar_files = sorted([str(p) for p in Path(PROCESS_DIR).glob("*.tar.gz")] + [str(p) for p in Path(PROCESS_DIR).glob("*.tar")])
    if len(tar_files) == 0:
        raise RuntimeError(f"No tar/tar.gz found under: {PROCESS_DIR}")

    manifest_rows = []
    fail_rows = []
    wrote = 0

    # 3) scan each tar
    for tar_path in tar_files:
        tar_name = os.path.basename(tar_path)
        print(f"\n[SCAN] {tar_name}")

        complete = list_triplets_in_tar(tar_path)
        print(f"  complete triplets found: {len(complete)}")

        # filter by Table1 subjects (保留 sample_prefix 對應的 subject_id 在 22 位內)
        kept = {}
        for sample_prefix in complete.keys():
            subj = guess_subject_id_from_prefix(sample_prefix)
            if subj in sn_subjects_set:
                kept[sample_prefix] = complete[sample_prefix]
        print(f"  kept after Table1 filter: {len(kept)}")

        if len(kept) == 0:
            continue

        # open tar once
        with tarfile.open(tar_path, "r:*") as t:
            for sample_prefix, mem in kept.items():
                sample_id = sample_prefix.lower()
                subject_id = guess_subject_id_from_prefix(sample_prefix)
                batch = guess_batch(subject_id)
                muscle = guess_muscle_from_prefix(sample_prefix)

                out_h5ad = os.path.join(OUT_H5AD_DIR, f"{sample_id}.counts.h5ad")

                # skip if already exists
                if os.path.exists(out_h5ad) and os.path.getsize(out_h5ad) > 0:
                    manifest_rows.append({
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "batch": batch,
                        "muscle_guess": muscle,
                        "n_cells": np.nan,
                        "n_genes": np.nan,
                        "out_h5ad": out_h5ad,
                        "source_tar": tar_name,
                        "member_mtx": mem["mtx"],
                        "member_barcodes": mem["barcodes"],
                        "member_features": mem["features"],
                        "status": "SKIP_EXISTS"
                    })
                    continue

                tmpdir = tempfile.mkdtemp(prefix="step1_10x_")
                try:
                    # decide standard filenames based on whether member endswith .gz
                    mtx_is_gz = mem["mtx"].lower().endswith(".gz")
                    bar_is_gz = mem["barcodes"].lower().endswith(".gz")
                    feat_is_gz = mem["features"].lower().endswith(".gz")

                    mtx_out = os.path.join(tmpdir, "matrix.mtx.gz" if mtx_is_gz else "matrix.mtx")
                    bar_out = os.path.join(tmpdir, "barcodes.tsv.gz" if bar_is_gz else "barcodes.tsv")
                    feat_out = os.path.join(tmpdir, "features.tsv.gz" if feat_is_gz else "features.tsv")

                    extract_member_to_file(t, mem["mtx"], mtx_out)
                    extract_member_to_file(t, mem["barcodes"], bar_out)
                    extract_member_to_file(t, mem["features"], feat_out)

                    # load
                    adata = load_10x_from_tempdir(tmpdir)
                    ensure_unique_obsnames(adata, sample_id)

                    # annotate
                    adata.obs["sample_id"] = sample_id
                    adata.obs["subject_id"] = subject_id
                    adata.obs["batch"] = batch
                    adata.obs["muscle_guess"] = muscle
                    adata.uns["source_tar"] = tar_name
                    adata.uns["member_paths"] = mem

                    # keep counts (do NOT normalize/log here)
                    # X already counts. To be explicit, also store in layer.
                    adata.layers["counts"] = adata.X.copy()

                    # write
                    adata.write_h5ad(out_h5ad, compression="gzip")

                    wrote += 1
                    print(f"  [WRITE] {sample_id}: cells={adata.n_obs}, genes={adata.n_vars} -> {os.path.basename(out_h5ad)}")

                    manifest_rows.append({
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "batch": batch,
                        "muscle_guess": muscle,
                        "n_cells": adata.n_obs,
                        "n_genes": adata.n_vars,
                        "out_h5ad": out_h5ad,
                        "source_tar": tar_name,
                        "member_mtx": mem["mtx"],
                        "member_barcodes": mem["barcodes"],
                        "member_features": mem["features"],
                        "status": "OK"
                    })

                except Exception as e:
                    # 記錄失敗原因，讓你回頭精準查 OM4 或特定 sample
                    print(f"  [FAIL] {sample_id}: {e}")
                    fail_rows.append({
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "source_tar": tar_name,
                        "member_mtx": mem.get("mtx", ""),
                        "member_barcodes": mem.get("barcodes", ""),
                        "member_features": mem.get("features", ""),
                        "error": str(e),
                    })
                    manifest_rows.append({
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "batch": batch,
                        "muscle_guess": muscle,
                        "n_cells": np.nan,
                        "n_genes": np.nan,
                        "out_h5ad": out_h5ad,
                        "source_tar": tar_name,
                        "member_mtx": mem.get("mtx", ""),
                        "member_barcodes": mem.get("barcodes", ""),
                        "member_features": mem.get("features", ""),
                        "status": "FAIL"
                    })
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

    # 4) write manifest / failures
    man = pd.DataFrame(manifest_rows)
    man.to_csv(MANIFEST_TSV, sep="\t", index=False)
    print(f"\n[OK] manifest -> {MANIFEST_TSV}")
    print(f"[OK] wrote h5ad: {wrote}")

    if len(fail_rows) > 0:
        fails = pd.DataFrame(fail_rows)
        fails.to_csv(FAIL_TSV, sep="\t", index=False)
        print(f"[WARN] failures -> {FAIL_TSV}  (n={len(fail_rows)})")


if __name__ == "__main__":
    main()
