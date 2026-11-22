#!/usr/bin/env python3
"""
Elliptic++ replication of the Elliptic2 "who sent it" finding.

Question: on a second real dataset, how much of illicit-transaction detection
is explained by WHO sent the transaction (its input wallets) rather than by the
transaction's own features / how the money moved?

Task   : binary transaction classification, illicit (class 1) vs licit (class 2);
         unknown (class 3) rows are dropped from train and test.
Sender : the set of input addresses of a transaction (AddrTx_edgelist.csv).

Feature sets
  full        : all transaction features in txs_features.csv (local + aggregate + augmented)
  local       : Local_feature_* only (no neighbourhood aggregates)
  sender      : mean / max of the input wallets' features at the tx's time step
  receiver    : same for output wallets (contrast)
  sender_prior: train-only, out-of-fold target encoding of the sender wallets
                (share of the sender's OTHER labelled train txs that are illicit) -
                the purest "who sent it" signal, no transaction content at all
  full+prior  : full + sender_prior

Splits
  random   : stratified 70/30
  temporal : train time steps 1-34, test 35-49 (standard Elliptic split)
  sender_disjoint : transactions grouped into sender clusters (union-find over
                shared input addresses); whole clusters go to train or test, so
                no test sender is ever seen in training

Diagnostic reported per split: share of test txs with >=1 input wallet seen in train.

wallets_classes.csv is NOT used as a feature: Elliptic++ wallet labels are
derived from the labels of the transactions they took part in, so using them
would leak the target.

Usage:
  python3 epp_sender_test.py --data "/path/to/Elliptic++ Dataset" --out results_epp
  (the folder must hold txs_features.csv, txs_classes.csv, AddrTx_edgelist.csv,
   TxAddr_edgelist.csv, wallets_features.csv; subfolders are searched too)
"""
import argparse, os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold

FILES = ["txs_features.csv", "txs_classes.csv", "AddrTx_edgelist.csv",
         "TxAddr_edgelist.csv", "wallets_features.csv"]


def find_files(root):
    found = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f in FILES and f not in found:
                found[f] = os.path.join(d, f)
    missing = [f for f in FILES if f not in found]
    if missing:
        sys.exit(f"Missing files under {root}: {missing}")
    return found


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- loading
def load(root):
    p = find_files(root)
    log("reading CSVs ...")
    txf = pd.read_csv(p["txs_features.csv"])
    txc = pd.read_csv(p["txs_classes.csv"])
    a2t = pd.read_csv(p["AddrTx_edgelist.csv"])   # input_address, txId
    t2a = pd.read_csv(p["TxAddr_edgelist.csv"])   # txId, output_address
    wf = pd.read_csv(p["wallets_features.csv"])   # address, Time step, ...
    tx = txf.merge(txc, on="txId", how="inner")
    tx = tx[tx["class"].isin([1, 2])].copy()
    tx["y"] = (tx["class"] == 1).astype(int)
    log(f"labelled txs: {len(tx):,}  illicit: {tx.y.sum():,} ({tx.y.mean():.2%})")
    return tx, a2t, t2a, wf


def wallet_agg(tx, edges, addr_col, wf, prefix):
    """mean and max of wallet features over a tx's input (or output) wallets,
    using the wallet's row at the tx's time step (falls back to wallet mean)."""
    wcols = [c for c in wf.columns if c not in ("address", "Time step")]
    w_ts = wf.groupby(["address", "Time step"], as_index=False)[wcols].mean()
    w_all = wf.groupby("address", as_index=False)[wcols].mean()
    e = edges[edges.txId.isin(tx.txId)][["txId", addr_col]].rename(columns={addr_col: "address"})
    e = e.merge(tx[["txId", "Time step"]], on="txId")
    m = e.merge(w_ts, on=["address", "Time step"], how="left")
    miss = m[wcols[0]].isna()
    if miss.any():
        fb = e[miss.values].merge(w_all, on="address", how="left")
        m.loc[miss.values, wcols] = fb[wcols].values
    g = m.groupby("txId")[wcols]
    out = pd.concat([g.mean().add_prefix(f"{prefix}_mean_"),
                     g.max().add_prefix(f"{prefix}_max_")], axis=1)
    out[f"{prefix}_n_wallets"] = e.groupby("txId").size()
    return out


# ---------------------------------------------------------------- sender clusters
def sender_clusters(tx, a2t):
    """union-find: txs sharing any input address end up in the same cluster."""
    e = a2t[a2t.txId.isin(tx.txId)]
    ids = {t: i for i, t in enumerate(tx.txId.values)}
    parent = np.arange(len(ids))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, grp in e.groupby("input_address")["txId"]:
        idx = [ids[t] for t in grp.values]
        r0 = find(idx[0])
        for j in idx[1:]:
            rj = find(j)
            if rj != r0:
                parent[rj] = r0
    return np.array([find(ids[t]) for t in tx.txId.values])


# ---------------------------------------------------------------- sender prior
def sender_prior(tx_ids, y, train_mask, a2t, n_folds=5, seed=0, alpha=1.0):
    """Smoothed share of illicit labelled TRAIN txs per input wallet, averaged
    (and maxed) over a tx's wallets, excluding the tx itself.
    Train rows: out-of-fold. Test rows: fitted on all train rows.
    Returns array (n, 3): mean prior, max prior, any-wallet-seen flag."""
    n = len(tx_ids)
    base = y[train_mask].mean()
    e = a2t[a2t.txId.isin(tx_ids)][["txId", "input_address"]]
    pos = pd.Series(np.arange(n), index=tx_ids)
    e = e.assign(row=pos.loc[e.txId].values)
    out = np.full((n, 3), np.nan)
    out[:, 2] = 0

    def fill(fit_rows, apply_rows):
        fit = e[e.row.isin(fit_rows)]
        stats = (fit.assign(yy=y[fit.row.values])
                 .groupby("input_address")["yy"].agg(["sum", "count"]))
        app = e[e.row.isin(apply_rows)].join(stats, on="input_address")
        app["seen"] = app["count"].notna().astype(int)
        app["p"] = (app["sum"].fillna(0) + alpha * base) / (app["count"].fillna(0) + alpha)
        agg = app.groupby("row").agg(pm=("p", "mean"), px=("p", "max"), s=("seen", "max"))
        out[agg.index.values, 0] = agg.pm.values
        out[agg.index.values, 1] = agg.px.values
        out[agg.index.values, 2] = agg.s.values

    tr = np.where(train_mask)[0]
    te = np.where(~train_mask)[0]
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for a, b in skf.split(tr, y[tr]):
        fill(tr[a], tr[b])
    fill(tr, te)
    out[np.isnan(out[:, 0]), 0] = base   # txs with no known input wallet
    out[np.isnan(out[:, 1]), 1] = base
    return out


# ---------------------------------------------------------------- evaluation
def fit_eval(Xtr, ytr, Xte, yte, seed):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=seed)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    # threshold chosen on train predictions to maximise F1 (no test peeking)
    ptr = clf.predict_proba(Xtr)[:, 1]
    ths = np.linspace(0.05, 0.95, 19)
    th = ths[np.argmax([f1_score(ytr, ptr >= t, zero_division=0) for t in ths])]
    return dict(auroc=roc_auc_score(yte, p), auprc=average_precision_score(yte, p),
                f1=f1_score(yte, p >= th, zero_division=0))


def make_splits(tx, clusters, seed):
    n = len(tx)
    y = tx.y.values
    idx = np.arange(n)
    tr, _ = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    m_rand = np.zeros(n, bool); m_rand[tr] = True
    m_temp = (tx["Time step"].values <= 34)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr_g, _ = next(gss.split(idx, y, groups=clusters))
    m_grp = np.zeros(n, bool); m_grp[tr_g] = True
    return {"random": m_rand, "temporal": m_temp, "sender_disjoint": m_grp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_epp")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tx, a2t, t2a, wf = load(a.data)
    tx = tx.reset_index(drop=True)
    full_cols = [c for c in tx.columns if c not in ("txId", "Time step", "class", "y")]
    local_cols = [c for c in full_cols if c.startswith("Local_feature")]

    log("aggregating sender / receiver wallet features ...")
    S = wallet_agg(tx, a2t, "input_address", wf, "snd")
    R = wallet_agg(tx, t2a, "output_address", wf, "rcv")
    tx = tx.join(S, on="txId").join(R, on="txId")
    snd_cols, rcv_cols = list(S.columns), list(R.columns)
    has_snd = tx["snd_n_wallets"].notna().mean()
    log(f"txs with >=1 known input wallet: {has_snd:.1%}")

    log("building sender clusters ...")
    clusters = sender_clusters(tx, a2t)
    cl_sizes = pd.Series(clusters).value_counts()
    log(f"sender clusters: {len(cl_sizes):,}  largest holds {cl_sizes.iloc[0]/len(tx):.1%} of txs")

    y = tx.y.values
    rows, diag = [], []
    for seed in range(a.seeds):
        for split, mtr in make_splits(tx, clusters, seed).items():
            # (temporal split is fixed; its seeds vary only the model and prior folds)
            pri = sender_prior(tx.txId.values, y, mtr, a2t, seed=seed)
            prior_cols = ["prior_mean", "prior_max", "prior_seen"]
            P = pd.DataFrame(pri, columns=prior_cols)
            seen_test = P.loc[~mtr, "prior_seen"].mean()
            diag.append(dict(seed=seed, split=split, n_train=int(mtr.sum()),
                             n_test=int((~mtr).sum()), test_illicit=y[~mtr].mean(),
                             test_sender_seen_in_train=seen_test))
            D = pd.concat([tx, P], axis=1)
            sets = {"full": full_cols, "local": local_cols, "sender": snd_cols,
                    "receiver": rcv_cols, "sender_prior": prior_cols,
                    "full+prior": full_cols + prior_cols}
            for name, cols in sets.items():
                r = fit_eval(D.loc[mtr, cols].values, y[mtr],
                             D.loc[~mtr, cols].values, y[~mtr], seed)
                rows.append(dict(seed=seed, split=split, features=name, **r))
                log(f"seed {seed} {split:15s} {name:13s} AUROC {r['auroc']:.4f} "
                    f"AUPRC {r['auprc']:.4f} F1 {r['f1']:.4f}")

    res = pd.DataFrame(rows); dg = pd.DataFrame(diag)
    res.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    dg.to_csv(os.path.join(a.out, "split_diagnostics.csv"), index=False)
    summ = (res.groupby(["split", "features"])[["auroc", "auprc", "f1"]]
            .agg(["mean", "std"]).round(4))
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    # share of full-feature AUPRC above base rate recovered by sender-only signal
    lines = []
    for split in ["random", "temporal", "sender_disjoint"]:
        base = dg[dg.split == split].test_illicit.mean()
        g = res[res.split == split].groupby("features").auprc.mean()
        for s in ["sender", "sender_prior"]:
            frac = (g[s] - base) / (g["full"] - base) if g["full"] > base else np.nan
            lines.append(dict(split=split, signal=s, auprc=g[s], full_auprc=g["full"],
                              base_rate=base, share_of_full_lift=frac))
    rec = pd.DataFrame(lines).round(4)
    rec.to_csv(os.path.join(a.out, "sender_share_of_lift.csv"), index=False)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(n_labelled=len(tx), illicit_rate=float(y.mean()),
                       share_with_sender=float(has_snd), n_clusters=int(len(cl_sizes)),
                       largest_cluster_share=float(cl_sizes.iloc[0] / len(tx))), f, indent=2)
    print("\n=== SUMMARY (mean / std over seeds) ===\n", summ.to_string())
    print("\n=== split diagnostics (mean) ===\n",
          dg.groupby("split")[["n_train", "n_test", "test_illicit",
                               "test_sender_seen_in_train"]].mean().round(4).to_string())
    print("\n=== share of full-model AUPRC lift recovered by sender-only signals ===\n",
          rec.to_string(index=False))
    log(f"done -> {a.out}/")


if __name__ == "__main__":
    main()
