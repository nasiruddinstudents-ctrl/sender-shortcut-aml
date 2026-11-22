"""
inspect_elliptic2.py  -  Step 4 gate check for the Elliptic2 dataset.

Reads every file in chunks (safe on a laptop), then reports:
  * file size, column names and row count for each CSV
  * label balance and subgraph sizes (from the small labeled files)
  * how large a 1-hop (and optionally 2-hop) neighbourhood around the
    labeled subgraphs would be, plus a rough RAM estimate

Nothing here trains a model or changes the data.

Usage:
    pip install pandas numpy
    python inspect_elliptic2.py --data ~/elliptic2/dataset
    python inspect_elliptic2.py --data ~/elliptic2/dataset --two-hop   # slower

Send the full printed output back for the gate decision.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

FILES = [
    "nodes.csv",
    "edges.csv",
    "connected_components.csv",
    "background_nodes.csv",
    "background_edges.csv",
]
SMALL_FILES = {"nodes.csv", "edges.csv", "connected_components.csv"}
CHUNK = 5_000_000


def human(nbytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} PB"


def header(path: str) -> list:
    return list(pd.read_csv(path, nrows=0).columns)


def count_rows(path: str) -> int:
    n = 0
    for chunk in pd.read_csv(path, usecols=[0], chunksize=CHUNK):
        n += len(chunk)
    return n


def bytes_per_row(path: str) -> float:
    sample = pd.read_csv(path, nrows=100_000)
    return sample.memory_usage(deep=True).sum() / max(len(sample), 1)


def find_col(cols, keywords):
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the 5 CSV files")
    ap.add_argument("--two-hop", action="store_true", help="also estimate 2-hop size")
    args = ap.parse_args()
    d = os.path.expanduser(args.data)

    # ------------------------------------------------------------------ 1
    section("1. FILES: size, columns, rows, estimated in-memory size")
    info = {}
    for f in FILES:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print(f"\n[MISSING] {f}")
            continue
        t = time.time()
        cols = header(p)
        rows = count_rows(p)
        bpr = bytes_per_row(p)
        info[f] = {"cols": cols, "rows": rows}
        print(f"\n{f}")
        print(f"  disk size      : {human(os.path.getsize(p))}")
        print(f"  rows           : {rows:,}")
        print(f"  columns ({len(cols):>3})  : {cols[:12]}{' ...' if len(cols) > 12 else ''}")
        print(f"  est. RAM (pandas, all columns): {human(bpr * rows)}")
        print(f"  ({time.time() - t:,.0f}s)")

    # ------------------------------------------------------------------ 2
    section("2. LABELED SUBGRAPHS")
    labeled_ids = None
    cc_path = os.path.join(d, "connected_components.csv")
    nodes_path = os.path.join(d, "nodes.csv")

    if os.path.exists(cc_path):
        cc = pd.read_csv(cc_path)
        print(f"connected_components.csv columns: {list(cc.columns)}")
        lab = find_col(cc.columns, ["label", "class", "illicit", "suspicious"])
        if lab:
            vc = cc[lab].value_counts()
            print(f"\nLabel counts ({lab}):\n{vc.to_string()}")
            print(f"Share of each label:\n{(vc / vc.sum()).round(4).to_string()}")
        else:
            print("No label-like column found; check the column list above.")

    if os.path.exists(nodes_path):
        nodes = pd.read_csv(nodes_path)
        print(f"\nnodes.csv columns: {list(nodes.columns)}")
        id_col = nodes.columns[0]
        comp_col = find_col(nodes.columns, ["cc", "component", "subgraph"])
        labeled_ids = np.unique(nodes[id_col].to_numpy())
        print(f"Unique nodes inside labeled subgraphs ({id_col}): {len(labeled_ids):,}")
        if comp_col and comp_col != id_col:
            sizes = nodes.groupby(comp_col).size()
            print(f"\nSubgraph size (nodes per {comp_col}):")
            print(sizes.describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())
            dup = nodes[id_col].duplicated().sum()
            print(f"Node appearances in more than one subgraph: {dup:,}")

    # ------------------------------------------------------------------ 3
    section("3. NEIGHBOURHOOD SIZE AROUND LABELED SUBGRAPHS")
    be_path = os.path.join(d, "background_edges.csv")
    if labeled_ids is None or not os.path.exists(be_path):
        print("Skipped: need nodes.csv and background_edges.csv.")
        return

    be_cols = info.get("background_edges.csv", {}).get("cols") or header(be_path)
    src, dst = be_cols[0], be_cols[1]
    print(f"Assuming edge endpoints are columns '{src}' -> '{dst}'. "
          "Correct this if the header above says otherwise.")

    def hop(seed: np.ndarray):
        """One streaming pass: edges touching seed, and the expanded node set."""
        n_edges = 0
        found = [seed]
        for ch in pd.read_csv(be_path, usecols=[src, dst], chunksize=CHUNK):
            s = ch[src].to_numpy()
            t = ch[dst].to_numpy()
            ms = np.isin(s, seed, assume_unique=False)
            mt = np.isin(t, seed, assume_unique=False)
            m = ms | mt
            n_edges += int(m.sum())
            found.append(s[m])
            found.append(t[m])
        return n_edges, np.unique(np.concatenate(found))

    bpr_edge = bytes_per_row(be_path)
    bn_path = os.path.join(d, "background_nodes.csv")
    bpr_node = bytes_per_row(bn_path) if os.path.exists(bn_path) else 0

    t = time.time()
    e1, n1 = hop(labeled_ids)
    print(f"\n1-hop: {len(n1):,} nodes, {e1:,} edges  ({time.time() - t:,.0f}s)")
    print(f"  est. RAM with all features: "
          f"{human(len(n1) * bpr_node + e1 * bpr_edge)}")

    if args.two_hop:
        t = time.time()
        e2, n2 = hop(n1)
        print(f"\n2-hop: {len(n2):,} nodes, {e2:,} edges  ({time.time() - t:,.0f}s)")
        print(f"  est. RAM with all features: "
              f"{human(len(n2) * bpr_node + e2 * bpr_edge)}")
    else:
        print("\n(Run again with --two-hop for the 2-hop estimate.)")

    section("DONE - send this whole output back")


if __name__ == "__main__":
    main()
