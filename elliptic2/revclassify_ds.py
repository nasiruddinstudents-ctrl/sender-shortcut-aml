"""
revclassify_ds.py  -  Reimplementation of RevClassify (DeepSets version) from
Song et al. (2024), evaluated under the random and entity-aware splits.

A subgraph is represented only by the node features of its SENDERS and its
RECEIVERS. Each set goes through its own small network (phi), is mean-pooled,
and the two pooled vectors plus log set sizes go through a classifier (rho).

Differences from the paper, to state in the manuscript:
  * inputs are the released binned node features (RevTrack also used node
    embeddings it serves separately);
  * sender/receiver extraction is our own (their preprocessing is unpublished);
  * mean pooling + log set size instead of plain sum pooling, for stability
    with hub-sized sets.

Variants:
  ds_full           senders + receivers (the RevClassify setting)
  ds_receiver_only  receivers only (no sender information)

Needs entity_split_benchmark.py in the same folder. CPU is fine.

Usage (from the folder with the CSVs):
    python3 -m pip install --user torch
    caffeinate -i python3 -u revclassify_ds.py --data . --hood ./out
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

import entity_split_benchmark as esb

warnings.filterwarnings("ignore", category=UserWarning)


class SetIndex:
    """CSR-style index: for each subgraph position, its member node rows."""

    def __init__(self, pairs, cc_pos, node_row, n_cc):
        cp = pairs["cc"].map(cc_pos).to_numpy()
        nr = pairs["ent"].map(node_row).to_numpy()
        ok = ~(np.isnan(cp.astype(float)) | np.isnan(nr.astype(float)))
        cp, nr = cp[ok].astype(np.int64), nr[ok].astype(np.int64)
        order = np.argsort(cp, kind="stable")
        self.members = nr[order]
        counts = np.bincount(cp, minlength=n_cc)
        self.start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.count = counts

    def gather(self, batch_pos):
        c = self.count[batch_pos]
        seg = np.repeat(np.arange(len(batch_pos)), c)
        offs = np.repeat(self.start[batch_pos], c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
        return self.members[offs], seg, c


class DeepSetsClassifier(nn.Module):
    def __init__(self, d_in, use_senders=True, hidden=128, dropout=0.2):
        super().__init__()
        self.use_senders = use_senders
        mk = lambda: nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.phi_r = mk()
        self.phi_s = mk() if use_senders else None
        d_rho = hidden * (2 if use_senders else 1) + (2 if use_senders else 1)
        self.rho = nn.Sequential(nn.Linear(d_rho, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    @staticmethod
    def pool(h, seg, n, counts):
        out = torch.zeros(n, h.shape[1])
        out.index_add_(0, seg, h)
        return out / counts.clamp(min=1).unsqueeze(1)

    def forward(self, feats, batch):
        n = batch["n"]
        parts, sizes = [], []
        r_idx, r_seg, r_cnt = batch["r"]
        parts.append(self.pool(self.phi_r(feats[r_idx]), r_seg, n, r_cnt))
        sizes.append(torch.log1p(r_cnt.float()).unsqueeze(1))
        if self.use_senders:
            s_idx, s_seg, s_cnt = batch["s"]
            parts.append(self.pool(self.phi_s(feats[s_idx]), s_seg, n, s_cnt))
            sizes.append(torch.log1p(s_cnt.float()).unsqueeze(1))
        return self.rho(torch.cat(parts + sizes, dim=1)).squeeze(1)


def make_batch(pos, s_index, r_index):
    def t(ix):
        idx, seg, cnt = ix.gather(pos)
        return (torch.from_numpy(idx), torch.from_numpy(seg), torch.from_numpy(cnt).float())
    return {"n": len(pos), "s": t(s_index), "r": t(r_index)}


def predict(model, feats, pos, s_index, r_index, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pos), bs):
            out.append(torch.sigmoid(model(feats, make_batch(pos[i:i + bs], s_index, r_index))).numpy())
    return np.concatenate(out) if out else np.array([])


def train_eval(feats, y, tr, va, te, s_index, r_index, use_senders, seed,
               epochs=40, patience=6, bs=512, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DeepSetsClassifier(feats.shape[1], use_senders=use_senders)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_w = torch.tensor((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_ap, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(feats, make_batch(b, s_index, r_index)),
                           torch.from_numpy(y[b]).float())
            loss.backward()
            opt.step()
        yv = y[va]
        ap = (average_precision_score(yv, predict(model, feats, va, s_index, r_index))
              if 0 < yv.sum() < len(yv) else 0.0)
        if ap > best_ap:
            best_ap, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return predict(model, feats, te, s_index, r_index), ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hood", default="./out")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--protect-rate", type=float, default=0.5)
    ap.add_argument("--max-group", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d, h = os.path.expanduser(args.data), os.path.expanduser(args.hood)
    t0 = time.time()

    nodes = pd.read_csv(os.path.join(d, "nodes.csv"))
    sub_e = pd.read_csv(os.path.join(d, "edges.csv"))
    cc = pd.read_csv(os.path.join(d, "connected_components.csv"))
    cc["y"] = (cc["ccLabel"] == "suspicious").astype(int)
    y_of = cc.set_index("ccId")["y"]
    ccs = cc["ccId"].to_numpy()
    cc_pos = pd.Series(np.arange(len(ccs)), index=ccs)
    y = cc["y"].to_numpy()

    print("Loading neighbourhood files ...", flush=True)
    e_ids = pq.read_table(os.path.join(h, "edges_1hop.parquet"),
                          columns=["clId1", "clId2", "txId"]).to_pandas()
    node_feat = pq.read_table(os.path.join(h, "nodes_1hop.parquet")).to_pandas()
    _, senders, receivers, _, _ = esb.senders_receivers(nodes, sub_e, e_ids)

    # node features: log1p then z-score (label-free, over the neighbourhood)
    fcols = [c for c in node_feat.columns if c != "clId"]
    Xn = np.log1p(np.clip(node_feat[fcols].to_numpy(dtype=np.float32), 0, None))
    Xn = np.nan_to_num((Xn - np.nanmean(Xn, 0)) / (np.nanstd(Xn, 0) + 1e-6))
    feats = torch.from_numpy(Xn.astype(np.float32))
    node_row = pd.Series(np.arange(len(node_feat)), index=node_feat["clId"].to_numpy())

    s_index = SetIndex(senders, cc_pos, node_row, len(ccs))
    r_index = SetIndex(receivers, cc_pos, node_row, len(ccs))
    print(f"Sets built: {s_index.members.size:,} sender links, "
          f"{r_index.members.size:,} receiver links ({time.time() - t0:,.0f}s)", flush=True)

    rate = senders.assign(yy=senders["cc"].map(y_of)).groupby("ent")["yy"].mean()
    protect = frozenset(rate[rate >= args.protect_rate].index)
    hub_cut, groups, _ = esb.choose_hub_cut(ccs, senders, protect, args.max_group)
    print(f"Entity split: protect_rate={args.protect_rate}, hub_cut={hub_cut}", flush=True)

    rows = []
    for split in ("random", "entity"):
        for seed in range(args.n_seeds):
            tr, va, te = (esb.random_split(ccs, seed) if split == "random"
                          else esb.entity_split(groups, seed, y_of))
            tr, va, te = (cc_pos.loc[x].to_numpy() for x in (tr, va, te))
            prev = y[te].mean()
            for name, use_s in (("ds_full", True), ("ds_receiver_only", False)):
                s, ep = train_eval(feats, y, tr, va, te, s_index, r_index, use_s, seed)
                m = esb.metrics(y[te], s)
                rows.append({"split": split, "seed": seed, "model": name, "epochs": ep,
                             "prevalence": prev, "lift": m["pr_auc"] / prev, **m})
            print(f"  done: split={split} seed={seed} ({time.time() - t0:,.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(h, "revclassify_ds_results.csv"), index=False)
    summ = res.groupby(["split", "model"])[["pr_auc", "roc_auc", "recall_top5pct", "lift"]] \
              .agg(["mean", "std"]).round(4)
    with open(os.path.join(h, "revclassify_ds_summary.json"), "w") as f:
        json.dump({"protect_rate": args.protect_rate, "hub_cut": hub_cut,
                   "n_seeds": args.n_seeds}, f, indent=2)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print(f"RevClassify-DS reimplementation, {args.n_seeds} seeds")
    print(summ.to_string())
    print("=" * 90)
    print("Saved revclassify_ds_results.csv in the out folder. Send this output back.")


if __name__ == "__main__":
    main()
