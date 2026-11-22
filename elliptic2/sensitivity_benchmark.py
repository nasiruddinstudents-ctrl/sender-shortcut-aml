"""
sensitivity_benchmark.py  -  Sensitivity checks for the Elliptic2 leakage paper.

Configurations (each compared with the main run: hub cut-off 100, cyclic
subgraphs kept with the all-nodes fallback):
  cut50   entity-aware split with hub cut-off 50  (more hubs left ungrouped)
  cut200  entity-aware split with hub cut-off 200 (fewer hubs left ungrouped)
  nocyc   cyclic subgraphs (no source or no sink) removed entirely;
          random and entity-aware splits, cut-off chosen as in the main run

Models: lookup, novelty, gb_all, gb_sender, gb_intrinsic (same definitions as
robustness_benchmark.py). Protection threshold 50%.

Needs entity_split_benchmark.py and robustness_benchmark.py in the same folder.

Usage (from the folder with the CSVs):
    caffeinate -i python3 -u sensitivity_benchmark.py --data . --hood ./out > run_sensitivity.txt 2>&1
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import entity_split_benchmark as esb
from robustness_benchmark import novelty_scores

warnings.filterwarnings("ignore", category=UserWarning)

CONFIGS = {
    "cut50": {"cut": 50, "drop_cyclic": False, "splits": ["entity"]},
    "cut200": {"cut": 200, "drop_cyclic": False, "splits": ["entity"]},
    "nocyc": {"cut": None, "drop_cyclic": True, "splits": ["random", "entity"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hood", default="./out")
    ap.add_argument("--configs", default="cut50,cut200,nocyc")
    ap.add_argument("--entity-seeds", type=int, default=10)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--protect-rate", type=float, default=0.5)
    ap.add_argument("--max-group", type=float, default=0.10)
    args = ap.parse_args()
    d, h = os.path.expanduser(args.data), os.path.expanduser(args.hood)
    t0 = time.time()

    nodes = pd.read_csv(os.path.join(d, "nodes.csv"))
    sub_e = pd.read_csv(os.path.join(d, "edges.csv"))
    cc = pd.read_csv(os.path.join(d, "connected_components.csv"))
    cc["y"] = (cc["ccLabel"] == "suspicious").astype(int)
    y_of = cc.set_index("ccId")["y"]
    all_ccs = cc["ccId"].to_numpy()

    print("Loading neighbourhood files ...", flush=True)
    edge_feat = pq.read_table(os.path.join(h, "edges_1hop.parquet")).to_pandas()
    node_feat = pq.read_table(os.path.join(h, "nodes_1hop.parquet")).to_pandas()
    internal, senders_all, receivers, cyc_src, cyc_sink = esb.senders_receivers(nodes, sub_e, edge_feat)
    print("Building features ...", flush=True)
    X, _ = esb.build_features(cc, nodes, internal, senders_all, receivers,
                              node_feat, edge_feat, cyc_src, cyc_sink)
    del edge_feat
    cols = list(X.columns)
    degree = ["snd_hubdeg_max", "rcv_hubdeg_max", "n_senders", "n_receivers"]
    sender = [c for c in cols if c.startswith("snd_") and c not in degree]
    receiver = [c for c in cols if c.startswith("rcv_") and c not in degree]
    internal_cols = [c for c in cols if c not in set(sender + receiver + degree)]
    feature_sets = {"gb_all": cols, "gb_sender": sender, "gb_intrinsic": internal_cols + receiver}
    cyclic = set(cyc_src) | set(cyc_sink)

    rows = []
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        cfg = CONFIGS[name]
        ccs = all_ccs[~np.isin(all_ccs, list(cyclic))] if cfg["drop_cyclic"] else all_ccs
        senders = senders_all[senders_all["cc"].isin(ccs)]
        prior = float(y_of.loc[ccs].mean())
        rate = senders.assign(yy=senders["cc"].map(y_of)).groupby("ent")["yy"].mean()
        protect = frozenset(rate[rate >= args.protect_rate].index)
        if cfg["cut"] is None:
            cut, groups, _ = esb.choose_hub_cut(ccs, senders, protect, args.max_group)
        else:
            cut = cfg["cut"]
            groups = esb.sender_groups(ccs, senders, cut, protect)
        largest = groups.value_counts().iloc[0] / len(ccs)
        print(f"\n[{name}] subgraphs={len(ccs):,} (suspicious {int(y_of.loc[ccs].sum()):,}), "
              f"hub_cut={cut}, largest group={largest:.2%}", flush=True)
        if "entity" in cfg["splits"] and largest > 0.5:
            print(f"  SKIPPED entity split for {name}: one group holds {largest:.0%} of subgraphs, "
                  "so a balanced 80/10/10 split is impossible. Report this as a finding.", flush=True)
            cfg = dict(cfg, splits=[x for x in cfg["splits"] if x != "entity"])

        for split in cfg["splits"]:
            n_seeds = args.entity_seeds if split == "entity" else args.random_seeds
            for seed in range(n_seeds):
                tr, va, te = (esb.random_split(ccs, seed) if split == "random"
                              else esb.entity_split(groups, seed, y_of))
                yte = y_of.loc[te].to_numpy()
                if min(len(tr), len(va), len(te)) == 0 or yte.sum() == 0 or y_of.loc[va].sum() == 0:
                    print(f"  skipped: {name} split={split} seed={seed} (empty split or no positives)",
                          flush=True)
                    continue
                tr_snd = senders[senders["cc"].isin(tr)]
                informative = set(tr_snd.loc[tr_snd["cc"].map(y_of) == 1, "ent"])
                sus_te = te[yte == 1]
                st = senders[senders["cc"].isin(sus_te)]
                inf = st.assign(s=st["ent"].isin(informative)).groupby("cc")["s"].any()
                base = {"config": name, "split": split, "seed": seed, "hub_cut": cut,
                        "largest_group": largest, "prevalence": yte.mean(),
                        "sus_informative_seen": float(inf.reindex(sus_te, fill_value=False).mean())
                        if len(sus_te) else np.nan}
                scores = {"lookup": esb.lookup_scores(senders, y_of, tr, te, prior),
                          "novelty": novelty_scores(senders, tr, te)}
                for m, fc in feature_sets.items():
                    scores[m], _ = esb.fit_gb(X, y_of, tr, va, te, fc, seed)
                for m, s in scores.items():
                    rows.append(dict(base, model=m, **esb.metrics(yte, s)))
                print(f"  done: {name} split={split} seed={seed} ({time.time() - t0:,.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        print("No configuration produced results.")
        return
    res.to_csv(os.path.join(h, "sensitivity_results.csv"), index=False)
    g = res.groupby(["config", "split", "model"])
    summ = pd.DataFrame({
        "pr_auc_mean": g["pr_auc"].mean(), "pr_auc_std": g["pr_auc"].std(),
        "pr_auc_median": g["pr_auc"].median(), "roc_auc_mean": g["roc_auc"].mean(),
        "prevalence": g["prevalence"].mean(),
    })
    summ["lift_ratio_of_means"] = summ["pr_auc_mean"] / summ["prevalence"]
    diag = res.groupby(["config", "split"])[["hub_cut", "largest_group", "prevalence",
                                              "sus_informative_seen"]].mean()
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("Split diagnostics (means over seeds):\n" + diag.round(4).to_string())
    print("\nPR-AUC by configuration:\n" + summ.round(4).to_string())
    print("=" * 90)
    print("Saved sensitivity_results.csv in the out folder. Send this output back.")


if __name__ == "__main__":
    main()
