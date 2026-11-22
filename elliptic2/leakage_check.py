"""
leakage_check.py  -  Step 7: do senders/receivers leak labels under a random split?

Uses the small files from extract_1hop.py plus nodes.csv, edges.csv and
connected_components.csv. Trains nothing; only counts and a lookup baseline.

Definitions follow Song et al. (2024, RevTrack):
  source = subgraph node with no incoming edge from inside its subgraph
  sink   = subgraph node with no outgoing edge to inside its subgraph
  sender   = node OUTSIDE the subgraph with an edge INTO one of its sources
  receiver = node OUTSIDE the subgraph that one of its sinks sends to
Assumes edge direction clId1 -> clId2 (funds flow). Confirm against the
RevTrack code before writing this into the paper.

Reports:
  A. sender/receiver coverage and how concentrated they are (hubs)
  B. under random 80:10:10 splits (5 seeds): share of test subgraphs whose
     senders/receivers were already seen in training
  C. a NO-LEARNING lookup baseline: score each test subgraph by the
     suspicious rate of its senders/receivers in training. If this already
     gets a high PR-AUC, published near-perfect scores partly reflect
     memorized entities, not learned laundering behaviour.
  D. feasibility of a sender-disjoint split (group size via shared senders)

Usage (from the folder with the CSVs):
    caffeinate -i python3 leakage_check.py --data . --hood ./out
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEEDS = [0, 1, 2, 3, 4]


def average_precision(y, s):
    """PR-AUC as average precision (same definition as sklearn)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    # handle ties: evaluate only at the last index of each distinct score
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    P = y.sum()
    if P == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / P
    recall_prev = np.r_[0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hood", default="./out")
    args = ap.parse_args()
    d = os.path.expanduser(args.data)
    h = os.path.expanduser(args.hood)

    nodes = pd.read_csv(os.path.join(d, "nodes.csv"))            # clId, ccId
    sub_e = pd.read_csv(os.path.join(d, "edges.csv"))            # clId1, clId2, txId
    cc = pd.read_csv(os.path.join(d, "connected_components.csv"))  # ccId, ccLabel
    cc["y"] = (cc["ccLabel"] == "suspicious").astype(int)
    y_of = cc.set_index("ccId")["y"]

    node_cc = nodes.set_index("clId")["ccId"]

    # ---- sources and sinks from internal edges
    sub_e = sub_e.assign(cc1=sub_e["clId1"].map(node_cc), cc2=sub_e["clId2"].map(node_cc))
    internal = sub_e[sub_e["cc1"] == sub_e["cc2"]]
    has_in = set(internal["clId2"])
    has_out = set(internal["clId1"])
    nodes["is_source"] = ~nodes["clId"].isin(has_in)
    nodes["is_sink"] = ~nodes["clId"].isin(has_out)
    no_src = nodes.groupby("ccId")["is_source"].sum() == 0
    no_sink = nodes.groupby("ccId")["is_sink"].sum() == 0
    # cyclic subgraphs (no source or no sink): fall back to all nodes, and count them
    cyc_src = set(no_src[no_src].index)
    cyc_sink = set(no_sink[no_sink].index)
    nodes.loc[nodes["ccId"].isin(cyc_src), "is_source"] = True
    nodes.loc[nodes["ccId"].isin(cyc_sink), "is_sink"] = True
    src_cc = nodes[nodes["is_source"]].set_index("clId")["ccId"]
    sink_cc = nodes[nodes["is_sink"]].set_index("clId")["ccId"]

    # ---- senders and receivers from the 1-hop edges
    e = pq.read_table(os.path.join(h, "edges_1hop.parquet"),
                      columns=["clId1", "clId2"]).to_pandas()
    e["cc_src_side"] = e["clId1"].map(node_cc)  # cc of the sending end (NaN if outside)
    e["cc_dst_side"] = e["clId2"].map(node_cc)

    # sender edges: outside node -> a source of subgraph c
    se = e[e["clId2"].isin(src_cc.index)].copy()
    se["cc"] = se["clId2"].map(src_cc)
    se = se[se["cc_src_side"] != se["cc"]]
    senders = se[["cc", "clId1"]].drop_duplicates().rename(columns={"clId1": "ent"})

    # receiver edges: a sink of subgraph c -> outside node
    re_ = e[e["clId1"].isin(sink_cc.index)].copy()
    re_["cc"] = re_["clId1"].map(sink_cc)
    re_ = re_[re_["cc_dst_side"] != re_["cc"]]
    receivers = re_[["cc", "clId2"]].drop_duplicates().rename(columns={"clId2": "ent"})

    out = {}
    n_cc = len(cc)
    out["A_subgraphs"] = n_cc
    out["A_cyclic_no_source"] = len(cyc_src)
    out["A_cyclic_no_sink"] = len(cyc_sink)
    out["A_share_with_any_sender"] = round(senders["cc"].nunique() / n_cc, 4)
    out["A_share_with_any_receiver"] = round(receivers["cc"].nunique() / n_cc, 4)
    out["A_distinct_senders"] = int(senders["ent"].nunique())
    out["A_distinct_receivers"] = int(receivers["ent"].nunique())

    for name, df in [("sender", senders), ("receiver", receivers)]:
        per_ent = df.groupby("ent")["cc"].nunique().sort_values(ascending=False)
        out[f"A_top10_{name}_subgraph_counts"] = per_ent.head(10).tolist()
        out[f"A_share_subgraphs_touching_top10_{name}s"] = round(
            df[df["ent"].isin(per_ent.head(10).index)]["cc"].nunique() / n_cc, 4)
        # label purity of the busiest entities
        lab = df.assign(y=df["cc"].map(y_of))
        purity = lab.groupby("ent")["y"].agg(["mean", "size"]).loc[per_ent.head(10).index]
        out[f"A_top10_{name}_suspicious_rate"] = purity["mean"].round(3).tolist()

    # ---- B + C: random splits
    ccs = cc["ccId"].to_numpy()
    y_all = y_of.loc[ccs].to_numpy()
    prior = y_all.mean()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ccs))
        n_tr = int(0.8 * len(ccs))
        n_va = int(0.1 * len(ccs))
        train = set(ccs[perm[:n_tr]])
        test = ccs[perm[n_tr + n_va:]]

        r = {"seed": seed}
        scores = {}
        for name, df in [("sender", senders), ("receiver", receivers)]:
            tr = df[df["cc"].isin(train)].assign(y=lambda x: x["cc"].map(y_of))
            ent_stats = tr.groupby("ent")["y"].agg(["sum", "size"])
            seen = set(ent_stats.index)
            te = df[df["cc"].isin(test)]
            te_any_seen = te.assign(s=te["ent"].isin(seen)).groupby("cc")["s"].any()
            r[f"test_share_any_{name}_seen_in_train"] = round(
                te_any_seen.reindex(test, fill_value=False).mean(), 4)
            sus_test = test[y_of.loc[test].to_numpy() == 1]
            r[f"suspicious_test_share_any_{name}_seen"] = round(
                te_any_seen.reindex(sus_test, fill_value=False).mean(), 4)
            # smoothed suspicious rate per entity, max over a subgraph's entities
            rate = (ent_stats["sum"] + prior) / (ent_stats["size"] + 1)
            s = te.assign(r=te["ent"].map(rate)).groupby("cc")["r"].max()
            scores[name] = s.reindex(test).fillna(prior).to_numpy()

        y_test = y_of.loc[test].to_numpy()
        r["lookup_PR_AUC_senders"] = round(average_precision(y_test, scores["sender"]), 4)
        r["lookup_PR_AUC_receivers"] = round(average_precision(y_test, scores["receiver"]), 4)
        r["lookup_PR_AUC_both_max"] = round(average_precision(
            y_test, np.maximum(scores["sender"], scores["receiver"])), 4)
        r["random_PR_AUC_expected"] = round(y_test.mean(), 4)
        rows.append(r)

    res = pd.DataFrame(rows)
    out["B_C_per_seed"] = rows
    out["B_C_mean"] = res.drop(columns="seed").mean().round(4).to_dict()

    # ---- D: sender-disjoint split feasibility (union-find over shared senders)
    parent = {c: c for c in ccs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in senders.groupby("ent")["cc"]:
        vals = grp.to_numpy()
        root = find(vals[0])
        for v in vals[1:]:
            rv = find(v)
            if rv != root:
                parent[rv] = root
    roots = pd.Series([find(c) for c in ccs])
    sizes = roots.value_counts()
    out["D_sender_groups"] = int(len(sizes))
    out["D_largest_group_share_of_subgraphs"] = round(sizes.iloc[0] / n_cc, 4)
    out["D_largest_group_suspicious"] = int(
        y_of.loc[ccs[(roots == sizes.index[0]).to_numpy()]].sum())
    out["D_top5_group_sizes"] = sizes.head(5).tolist()

    with open(os.path.join(h, "leakage_check.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("=" * 66)
    for k, v in out.items():
        if k != "B_C_per_seed":
            print(f"{k}: {v}")
    print("=" * 66)
    print(f"Saved to {os.path.join(h, 'leakage_check.json')}. Send this output back.")


if __name__ == "__main__":
    main()
