"""
robustness_benchmark.py  -  Robustness run for the Elliptic2 leakage finding.

Reuses the tested pieces of entity_split_benchmark.py (keep both files in the
same folder) and changes three things:

1. Balanced protection: in the entity-aware split, only senders whose
   suspicious rate is >= --protect-rate (default 0.5) are always kept in
   grouping. Licit hubs that touched one or two suspicious subgraphs no longer
   glue most of the data into one group.
2. Feature blocks are separated:
     sender    - sender node features (no degree)
     degree    - how many subgraphs a sender / receiver touches (label-free)
     receiver  - receiver node features
     internal  - internal node + internal edge features + subgraph structure
3. 95% stratified bootstrap intervals for every model's PR-AUC, and paired
   bootstrap intervals for key differences on the same test set.

Models (no entity IDs as features):
  lookup        suspicious rate of the subgraph's senders in training
  novelty       share of the subgraph's senders never seen in training
  gb_all        all blocks
  gb_sender     sender block only
  gb_degree     degree block only
  gb_intrinsic  internal + receiver blocks (no sender, no degree)
  gb_internal   internal block only

Usage (from the folder with the CSVs):
    caffeinate -i python3 robustness_benchmark.py --data . --hood ./out
"""

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

import entity_split_benchmark as esb

SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 300


def stratified_boot_idx(y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def novelty_scores(senders, train, test):
    seen = set(senders.loc[senders["cc"].isin(train), "ent"])
    te = senders[senders["cc"].isin(test)]
    s = te.assign(u=~te["ent"].isin(seen)).groupby("cc")["u"].mean()
    # subgraphs with no sender at all get the neutral middle value
    return s.reindex(test).fillna(0.5).to_numpy().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hood", default="./out")
    ap.add_argument("--protect-rate", type=float, default=0.5)
    ap.add_argument("--max-group", type=float, default=0.10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--splits", default="random,entity",
                    help="comma list: random, entity")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    sfx = f"_{args.tag}" if args.tag else ""
    d, h = os.path.expanduser(args.data), os.path.expanduser(args.hood)
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
    internal, senders, receivers, cyc_src, cyc_sink = esb.senders_receivers(nodes, sub_e, edge_feat)

    print("Building features ...")
    X, _ = esb.build_features(cc, nodes, internal, senders, receivers,
                              node_feat, edge_feat, cyc_src, cyc_sink)
    del edge_feat
    cols = list(X.columns)
    degree = ["snd_hubdeg_max", "rcv_hubdeg_max", "n_senders", "n_receivers"]
    sender = [c for c in cols if c.startswith("snd_") and c not in degree]
    receiver = [c for c in cols if c.startswith("rcv_") and c not in degree]
    internal_cols = [c for c in cols if c not in set(sender + receiver + degree)]
    feature_sets = {
        "gb_all": cols,
        "gb_sender": sender,
        "gb_degree": degree,
        "gb_intrinsic": internal_cols + receiver,
        "gb_internal": internal_cols,
    }
    for k, v in feature_sets.items():
        print(f"  {k:13s}: {len(v)} features")

    # ---- balanced protection: senders with a high suspicious rate
    rate = senders.assign(y=senders["cc"].map(y_of)).groupby("ent")["y"].mean()
    protect = frozenset(rate[rate >= args.protect_rate].index)
    print(f"Protected senders (suspicious rate >= {args.protect_rate:.0%}): {len(protect):,}")
    hub_cut, groups, tried = esb.choose_hub_cut(ccs, senders, protect, args.max_group)
    for t in tried:
        print(f"  hub_cut={t['hub_cut']:>5}: largest group {t['largest_group_share']:.2%}, "
              f"hub entities excluded {t['hub_entities_excluded']:,}")
    print(f"  -> using hub_cut={hub_cut}")

    rows, boot_rows = [], []
    for split in splits:
        for seed in seeds:
            tr, va, te = (esb.random_split(ccs, seed) if split == "random"
                          else esb.entity_split(groups, seed, y_of))
            yte = y_of.loc[te].to_numpy()

            tr_snd = senders[senders["cc"].isin(tr)]
            informative = set(tr_snd.loc[tr_snd["cc"].map(y_of) == 1, "ent"])
            sus_te = te[yte == 1]
            st = senders[senders["cc"].isin(sus_te)]
            inf_seen = st.assign(s=st["ent"].isin(informative)).groupby("cc")["s"].any()

            base = {"split": split, "seed": seed, "n_test": len(te),
                    "n_test_suspicious": int(yte.sum()),
                    "sus_test_informative_seen": float(
                        inf_seen.reindex(sus_te, fill_value=False).mean()) if len(sus_te) else np.nan}

            scores = {
                "lookup": esb.lookup_scores(senders, y_of, tr, te, prior),
                "novelty": novelty_scores(senders, tr, te),
            }
            for name, fcols in feature_sets.items():
                scores[name], _ = esb.fit_gb(X, y_of, tr, va, te, fcols, seed)

            for name, s in scores.items():
                rows.append(dict(base, model=name, **esb.metrics(yte, s)))

            # bootstrap: per-model CI and paired differences
            rng = np.random.default_rng(100 + seed)
            boots = {name: [] for name in scores}
            for _ in range(N_BOOT):
                ix = stratified_boot_idx(yte, rng)
                for name, s in scores.items():
                    boots[name].append(average_precision_score(yte[ix], s[ix]))
            boots = {k: np.array(v) for k, v in boots.items()}
            for name, b in boots.items():
                boot_rows.append({"split": split, "seed": seed, "what": name,
                                  "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
            for a, b in [("gb_all", "gb_intrinsic"), ("gb_sender", "gb_intrinsic"),
                         ("gb_intrinsic", "gb_internal"), ("gb_all", "lookup")]:
                diff = boots[a] - boots[b]
                boot_rows.append({"split": split, "seed": seed, "what": f"{a} - {b}",
                                  "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5)})
            print(f"  done: split={split} seed={seed} ({time.time() - t0:,.0f}s)")

    res = pd.DataFrame(rows)
    bres = pd.DataFrame(boot_rows)
    res.to_csv(os.path.join(h, f"robustness_results{sfx}.csv"), index=False)
    bres.to_csv(os.path.join(h, f"robustness_bootstrap{sfx}.csv"), index=False)

    mcols = ["pr_auc", "roc_auc", "recall_top5pct"]
    summ = res.groupby(["split", "model"])[mcols].agg(["mean", "std"]).round(4)
    ci = bres.groupby(["split", "what"])[["lo", "hi"]].mean().round(4)
    tests = res.groupby("split")[["n_test", "n_test_suspicious",
                                  "sus_test_informative_seen"]].mean().round(3)

    with open(os.path.join(h, f"robustness_summary{sfx}.json"), "w") as f:
        json.dump({"protect_rate": args.protect_rate, "n_seeds": args.n_seeds,
                   "splits": splits, "protected_senders": len(protect),
                   "hub_cut": hub_cut, "hub_cut_search": tried, "prior": prior,
                   "n_boot": N_BOOT}, f, indent=2)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print(f"protect_rate={args.protect_rate}  protected={len(protect):,}  hub_cut={hub_cut}  "
          f"seeds={args.n_seeds}  overall prevalence={prior:.4f}")
    lift = res.assign(prev=res["n_test_suspicious"] / res["n_test"],
                      lift=lambda q: q["pr_auc"] / q["prev"])
    lift_tbl = lift.groupby(["split", "model"])["lift"].agg(["mean", "min", "max"]).round(2)
    print("Average test size per split:\n" + tests.to_string())
    print(f"\nMean (std) over {args.n_seeds} seeds:\n" + summ.to_string())
    print("\nPR-AUC as times above chance (test prevalence), mean/min/max over seeds:\n"
          + lift_tbl.to_string())
    print("\nBootstrap 95% intervals for PR-AUC (averaged over seeds); "
          "rows with ' - ' are paired differences:\n" + ci.to_string())
    print("=" * 90)
    print(f"Saved robustness_results{sfx}.csv and robustness_bootstrap{sfx}.csv "
          "in the out folder. Send this output back.")


if __name__ == "__main__":
    main()
