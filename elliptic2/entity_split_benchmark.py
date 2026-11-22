"""
entity_split_benchmark.py  -  Does a LEARNED model's score survive an
entity-aware split on Elliptic2?

Compares, on 5 seeds each:
  split "random"  : standard random 80:10:10 over subgraphs (as in the papers)
  split "entity"  : subgraphs that share a NON-HUB sender are kept together
                    (group split 80:10:10). Hub senders (entities feeding many
                    subgraphs, e.g. exchanges) are excluded from grouping; the
                    hub cut-off is chosen automatically so no group exceeds 10%
                    of subgraphs.

Models (no entity IDs are ever used as features):
  lookup        : suspicious rate of the subgraph's senders in training
                  (memorisation reference, no learning)
  gb_all        : gradient boosting on sender + receiver + internal node
                  features + internal edge features + simple structure
  gb_no_sender  : same, minus all sender features
  gb_sender_only: sender features only

Metrics on the test split: PR-AUC, ROC-AUC, recall in the top 1% and 5% of
alerts (analyst review budgets).

Inputs: nodes.csv, edges.csv, connected_components.csv (in --data) and
edges_1hop.parquet, nodes_1hop.parquet (in --hood, from extract_1hop.py).

Usage (from the folder with the CSVs):
    python3 -m pip install --user scikit-learn
    caffeinate -i python3 entity_split_benchmark.py --data . --hood ./out
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = [0, 1, 2, 3, 4]
MAX_GROUP_SHARE = 0.10


# ---------------------------------------------------------------- structure
def senders_receivers(nodes, sub_e, e_ids):
    node_cc = nodes.set_index("clId")["ccId"]
    sub_e = sub_e.assign(cc1=sub_e["clId1"].map(node_cc), cc2=sub_e["clId2"].map(node_cc))
    internal = sub_e[sub_e["cc1"] == sub_e["cc2"]]
    nodes = nodes.copy()
    nodes["is_source"] = ~nodes["clId"].isin(set(internal["clId2"]))
    nodes["is_sink"] = ~nodes["clId"].isin(set(internal["clId1"]))
    no_src = nodes.groupby("ccId")["is_source"].sum() == 0
    no_sink = nodes.groupby("ccId")["is_sink"].sum() == 0
    cyc_src, cyc_sink = set(no_src[no_src].index), set(no_sink[no_sink].index)
    nodes.loc[nodes["ccId"].isin(cyc_src), "is_source"] = True
    nodes.loc[nodes["ccId"].isin(cyc_sink), "is_sink"] = True
    src_cc = nodes[nodes["is_source"]].set_index("clId")["ccId"]
    sink_cc = nodes[nodes["is_sink"]].set_index("clId")["ccId"]

    e = e_ids[["clId1", "clId2"]].copy()
    e["cc_a"] = e["clId1"].map(node_cc)
    e["cc_b"] = e["clId2"].map(node_cc)

    se = e[e["clId2"].isin(src_cc.index)].copy()
    se["cc"] = se["clId2"].map(src_cc)
    se = se[se["cc_a"] != se["cc"]]
    senders = se[["cc", "clId1"]].drop_duplicates().rename(columns={"clId1": "ent"})

    re_ = e[e["clId1"].isin(sink_cc.index)].copy()
    re_["cc"] = re_["clId1"].map(sink_cc)
    re_ = re_[re_["cc_b"] != re_["cc"]]
    receivers = re_[["cc", "clId2"]].drop_duplicates().rename(columns={"clId2": "ent"})
    return internal, senders, receivers, cyc_src, cyc_sink


# ---------------------------------------------------------------- features
def agg(df, key, feat_cols, prefix):
    g = df.groupby(key)[feat_cols]
    out = pd.concat([g.mean().add_suffix("_mean"), g.max().add_suffix("_max")], axis=1)
    return out.add_prefix(prefix)


def build_features(cc, nodes, internal, senders, receivers, node_feat, edge_feat,
                   cyc_src, cyc_sink):
    nf_cols = [c for c in node_feat.columns if c != "clId"]
    ef_cols = [c for c in edge_feat.columns if c not in ("clId1", "clId2", "txId")]

    snd = senders.merge(node_feat, left_on="ent", right_on="clId", how="left")
    rcv = receivers.merge(node_feat, left_on="ent", right_on="clId", how="left")
    inn = nodes.merge(node_feat, on="clId", how="left")
    # match internal edges on all three columns (txId alone is not unique)
    ie = internal[["clId1", "clId2", "txId", "cc1"]].merge(
        edge_feat, on=["clId1", "clId2", "txId"], how="left").rename(columns={"cc1": "cc"})

    # label-free hub degree: how many subgraphs each sender/receiver touches
    s_deg = senders.groupby("ent")["cc"].nunique()
    r_deg = receivers.groupby("ent")["cc"].nunique()

    X = pd.DataFrame(index=cc["ccId"].to_numpy())
    blocks = {
        "sender": agg(snd, "cc", nf_cols, "snd_"),
        "receiver": agg(rcv, "cc", nf_cols, "rcv_"),
        "internal_node": agg(inn, "ccId", nf_cols, "inn_"),
        "internal_edge": agg(ie, "cc", ef_cols, "ied_"),
    }
    struct = pd.DataFrame({
        "n_nodes": nodes.groupby("ccId").size(),
        "n_internal_edges": internal.groupby("cc1").size(),
        "n_senders": senders.groupby("cc").size(),
        "n_receivers": receivers.groupby("cc").size(),
        "snd_hubdeg_max": senders.assign(d=senders["ent"].map(s_deg)).groupby("cc")["d"].max(),
        "rcv_hubdeg_max": receivers.assign(d=receivers["ent"].map(r_deg)).groupby("cc")["d"].max(),
    })
    struct["cyclic_src"] = struct.index.isin(cyc_src).astype(int)
    struct["cyclic_sink"] = struct.index.isin(cyc_sink).astype(int)
    blocks["structure"] = struct

    for name, b in blocks.items():
        X = X.join(b, how="left")
    sender_cols = [c for c in X.columns if c.startswith("snd_")]
    return X.astype("float32"), sender_cols


# ---------------------------------------------------------------- splits
def random_split(ccs, seed):
    perm = np.random.default_rng(seed).permutation(len(ccs))
    a, b = int(0.8 * len(ccs)), int(0.9 * len(ccs))
    return ccs[perm[:a]], ccs[perm[a:b]], ccs[perm[b:]]


def sender_groups(ccs, senders, hub_cut, protect=frozenset()):
    # Hubs (degree >= hub_cut) are left out of grouping, EXCEPT protected
    # entities (strict mode: any sender that feeds a suspicious subgraph).
    deg = senders.groupby("ent")["cc"].nunique()
    keep = senders[(senders["ent"].map(deg) < hub_cut) | senders["ent"].isin(protect)]
    parent = {c: c for c in ccs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in keep.groupby("ent")["cc"]:
        v = grp.to_numpy()
        r = find(v[0])
        for u in v[1:]:
            ru = find(u)
            if ru != r:
                parent[ru] = r
    return pd.Series([find(c) for c in ccs], index=ccs)


def choose_hub_cut(ccs, senders, protect, max_share):
    deg = senders.groupby("ent")["cc"].nunique()
    excludable = deg[~deg.index.isin(protect)]
    tried = []
    for cut in [1000, 500, 200, 100, 50, 25, 10, 5, 3, 2]:
        g = sender_groups(ccs, senders, cut, protect)
        share = g.value_counts().iloc[0] / len(ccs)
        tried.append({"hub_cut": cut, "largest_group_share": round(float(share), 4),
                      "hub_entities_excluded": int((excludable >= cut).sum())})
        if share <= max_share:
            return cut, g, tried
    print("  WARNING: no cut-off met the group-size limit; using the last one tried.")
    return cut, g, tried


def entity_split(groups, seed, y_of):
    # Keep each group whole. Shuffle groups, then send each to the split that
    # is furthest below its 80/10/10 target -- measured on suspicious
    # subgraphs for groups that contain any, and on total size otherwise.
    # This keeps validation and test non-empty and roughly stratified.
    sizes = groups.value_counts()
    sus = pd.Series(y_of.loc[groups.index].to_numpy(), index=groups.to_numpy()).groupby(level=0).sum()
    order = np.random.default_rng(10_000 + seed).permutation(sizes.index.to_numpy())
    share = np.array([0.8, 0.1, 0.1])
    t_size, t_sus = share * len(groups), share * max(sus.sum(), 1)
    f_size, f_sus = np.zeros(3), np.zeros(3)
    assign = {}
    for g in order:
        if sus.get(g, 0) > 0:
            k = int(np.argmax((t_sus - f_sus) / t_sus))
        else:
            k = int(np.argmax((t_size - f_size) / t_size))
        assign[g] = k
        f_size[k] += sizes[g]
        f_sus[k] += sus.get(g, 0)
    lab = groups.map(assign).to_numpy()
    idx = groups.index.to_numpy()
    return idx[lab == 0], idx[lab == 1], idx[lab == 2]


# ---------------------------------------------------------------- eval
def metrics(y, s):
    order = np.argsort(-s, kind="mergesort")
    P = y.sum()
    if P == 0 or P == len(y):
        return {"pr_auc": np.nan, "roc_auc": np.nan,
                "recall_top1pct": np.nan, "recall_top5pct": np.nan}
    out = {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s)}
    for frac in (0.01, 0.05):
        k = max(1, int(round(frac * len(y))))
        out[f"recall_top{int(frac*100)}pct"] = y[order[:k]].sum() / P
    return out


def lookup_scores(senders, y_of, train, test, prior):
    tr = senders[senders["cc"].isin(train)].assign(y=lambda d: d["cc"].map(y_of))
    st = tr.groupby("ent")["y"].agg(["sum", "size"])
    rate = (st["sum"] + prior) / (st["size"] + 1)
    te = senders[senders["cc"].isin(test)]
    s = te.assign(r=te["ent"].map(rate)).groupby("cc")["r"].max()
    return s.reindex(test).fillna(prior).to_numpy()


def fit_gb(X, y_of, tr, va, te, cols, seed):
    # Fit once for up to 400 rounds; pick the round count with the best
    # validation PR-AUC (checked every 25 rounds); score test at that round.
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        early_stopping=False, class_weight="balanced", random_state=seed)
    Xtr, ytr = X.loc[tr, cols].to_numpy(), y_of.loc[tr].to_numpy()
    Xva, yva = X.loc[va, cols].to_numpy(), y_of.loc[va].to_numpy()
    Xte = X.loc[te, cols].to_numpy()
    m.fit(Xtr, ytr)
    best_ap, best_it = -1.0, 1
    for it, p in enumerate(m.staged_predict_proba(Xva), start=1):
        if it % 25 == 0 or it == m.n_iter_:
            ap = average_precision_score(yva, p[:, 1])
            if ap > best_ap:
                best_ap, best_it = ap, it
    for it, p in enumerate(m.staged_predict_proba(Xte), start=1):
        if it == best_it:
            return p[:, 1], best_it
    return m.predict_proba(Xte)[:, 1], m.n_iter_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hood", default="./out")
    ap.add_argument("--strict", action="store_true",
                    help="never exclude a sender that feeds any suspicious subgraph")
    ap.add_argument("--max-group", type=float, default=None,
                    help="largest allowed group share (default 0.10, strict 0.25)")
    ap.add_argument("--tag", default=None, help="suffix for output files")
    args = ap.parse_args()
    d, h = os.path.expanduser(args.data), os.path.expanduser(args.hood)
    max_share = args.max_group if args.max_group else (0.25 if args.strict else MAX_GROUP_SHARE)
    tag = args.tag or ("strict" if args.strict else "")
    sfx = f"_{tag}" if tag else ""
    t0 = time.time()

    nodes = pd.read_csv(os.path.join(d, "nodes.csv"))
    sub_e = pd.read_csv(os.path.join(d, "edges.csv"))
    cc = pd.read_csv(os.path.join(d, "connected_components.csv"))
    cc["y"] = (cc["ccLabel"] == "suspicious").astype(int)
    y_of = cc.set_index("ccId")["y"]
    ccs = cc["ccId"].to_numpy()
    prior = float(cc["y"].mean())

    print("Loading neighbourhood files ...")
    edge_feat = pq.read_table(os.path.join(h, "edges_1hop.parquet")).to_pandas()
    node_feat = pq.read_table(os.path.join(h, "nodes_1hop.parquet")).to_pandas()

    internal, senders, receivers, cyc_src, cyc_sink = senders_receivers(nodes, sub_e, edge_feat)
    print("Building features ...")
    X, sender_cols = build_features(cc, nodes, internal, senders, receivers,
                                    node_feat, edge_feat, cyc_src, cyc_sink)
    del edge_feat
    all_cols = list(X.columns)
    no_sender = [c for c in all_cols if c not in sender_cols]
    feature_sets = {"gb_all": all_cols, "gb_no_sender": no_sender, "gb_sender_only": sender_cols}
    print(f"Feature matrix: {X.shape[0]:,} subgraphs x {X.shape[1]} features "
          f"({len(sender_cols)} sender features)")

    sus_ccs = set(cc.loc[cc["y"] == 1, "ccId"])
    protect = (frozenset(senders.loc[senders["cc"].isin(sus_ccs), "ent"])
               if args.strict else frozenset())
    print(f"Choosing hub cut-off for the entity-aware split "
          f"(strict={args.strict}, protected senders={len(protect):,}, max group={max_share:.0%}) ...")
    hub_cut, groups, tried = choose_hub_cut(ccs, senders, protect, max_share)
    for t in tried:
        print(f"  hub_cut={t['hub_cut']:>5}: largest group {t['largest_group_share']:.2%}, "
              f"hub entities excluded {t['hub_entities_excluded']:,}")
    print(f"  -> using hub_cut={hub_cut}")

    rows = []
    for split in ("random", "entity"):
        for seed in SEEDS:
            tr, va, te = (random_split(ccs, seed) if split == "random"
                          else entity_split(groups, seed, y_of))
            yte = y_of.loc[te].to_numpy()
            # leakage diagnostic: suspicious test subgraphs with a sender seen in training
            tr_snd = senders[senders["cc"].isin(tr)]
            seen = set(tr_snd["ent"])
            # "informative": seen in training feeding at least one suspicious subgraph
            informative = set(tr_snd.loc[tr_snd["cc"].map(y_of) == 1, "ent"])
            sus_te = te[yte == 1]
            st = senders[senders["cc"].isin(sus_te)]
            hit = st.assign(s=st["ent"].isin(seen)).groupby("cc")["s"].any()
            hit_inf = st.assign(s=st["ent"].isin(informative)).groupby("cc")["s"].any()
            base = {"split": split, "seed": seed, "n_test": len(te),
                    "n_test_suspicious": int(yte.sum()),
                    "sus_test_sender_seen": float(hit.reindex(sus_te, fill_value=False).mean())
                    if len(sus_te) else np.nan,
                    "sus_test_informative_seen": float(
                        hit_inf.reindex(sus_te, fill_value=False).mean())
                    if len(sus_te) else np.nan}
            r = dict(base, model="lookup", **metrics(yte, lookup_scores(senders, y_of, tr, te, prior)))
            rows.append(r)
            for name, cols in feature_sets.items():
                s, it = fit_gb(X, y_of, tr, va, te, cols, seed)
                rows.append(dict(base, model=name, iters=it, **metrics(yte, s)))
            print(f"  done: split={split} seed={seed} ({time.time() - t0:,.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(h, f"entity_split_results{sfx}.csv"), index=False)
    cols = ["pr_auc", "roc_auc", "recall_top1pct", "recall_top5pct"]
    summ = res.groupby(["split", "model"])[cols].agg(["mean", "std"]).round(4)
    tests = res.groupby("split")[["n_test", "n_test_suspicious",
                                  "sus_test_sender_seen",
                                  "sus_test_informative_seen"]].mean().round(3)

    with open(os.path.join(h, f"entity_split_summary{sfx}.json"), "w") as f:
        json.dump({"strict": args.strict, "max_group_share": max_share,
                   "protected_senders": len(protect), "hub_cut": hub_cut, "hub_cut_search": tried,
                   "n_features": X.shape[1], "prior": prior,
                   "summary": {f"{s}|{m}|{c}|{st}": float(v)
                               for (s, m), row in summ.iterrows()
                               for (c, st), v in row.items()}}, f, indent=2)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print(f"strict={args.strict}   hub_cut={hub_cut}   features={X.shape[1]}   "
          f"chance PR-AUC={prior:.4f}")
    print("Average test size per split:\n" + tests.to_string())
    print("\nMean (std) over 5 seeds:")
    print(summ.to_string())
    print("=" * 90)
    print(f"Per-seed results saved to {os.path.join(h, f'entity_split_results{sfx}.csv')}. "
          "Send this output back.")


if __name__ == "__main__":
    main()
