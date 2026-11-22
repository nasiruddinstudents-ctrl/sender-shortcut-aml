"""
extract_1hop.py  -  Step 6 prep: pull the 1-hop neighbourhood of the labeled
subgraphs out of the full Elliptic2 files, once, into small Parquet files.

Streams the 77 GB edge file and the 5 GB node file in blocks (low memory),
keeps only rows that touch a labeled-subgraph node, and writes:

  out/edges_1hop.parquet   background edges touching labeled nodes (all 95 features)
  out/nodes_1hop.parquet   background-node rows for every node in that neighbourhood
  out/extract_summary.json counts and checks

Nothing is modified in the original CSVs. Expected run time on a laptop:
roughly 20-60 minutes, mostly reading the 77 GB file.

Usage (from the folder with the CSVs):
    python3 -m pip install --user pyarrow
    caffeinate -i python3 extract_1hop.py --data . --out ./out
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

BLOCK = 256 << 20  # 256 MB read blocks


def header(path):
    return list(pd.read_csv(path, nrows=0).columns)


def col_types(cols, id_cols):
    # IDs as int64; binned ordinal features as float32 (exact for small ints,
    # tolerates any blanks, halves memory vs float64).
    return {c: (pa.int64() if c in id_cols else pa.float32()) for c in cols}


def stream_filter(path, id_cols, keep_mask_fn, out_path, label):
    cols = header(path)
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=BLOCK),
        convert_options=pacsv.ConvertOptions(column_types=col_types(cols, id_cols)),
    )
    writer = None
    kept = scanned = 0
    t = time.time()
    for batch in reader:
        scanned += batch.num_rows
        mask = keep_mask_fn(batch)
        sub = batch.filter(mask)
        if sub.num_rows:
            tbl = pa.Table.from_batches([sub])
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            kept += sub.num_rows
        print(f"\r  {label}: scanned {scanned:,} rows, kept {kept:,}  "
              f"({time.time() - t:,.0f}s)", end="", flush=True)
    if writer:
        writer.close()
    print()
    return scanned, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()
    d = os.path.expanduser(args.data)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    nodes = pd.read_csv(os.path.join(d, "nodes.csv"))
    sub_edges = pd.read_csv(os.path.join(d, "edges.csv"))
    labeled = pa.array(np.unique(nodes["clId"].to_numpy()), type=pa.int64())
    print(f"Labeled-subgraph nodes: {len(labeled):,}")

    # ---- 1. edges touching any labeled node
    print("\n[1/2] Filtering background_edges.csv ...")
    e_path = os.path.join(out, "edges_1hop.parquet")
    e_scanned, e_kept = stream_filter(
        os.path.join(d, "background_edges.csv"),
        id_cols={"clId1", "clId2", "txId"},
        keep_mask_fn=lambda b: pc.or_(
            pc.is_in(b.column("clId1"), value_set=labeled),
            pc.is_in(b.column("clId2"), value_set=labeled),
        ),
        out_path=e_path,
        label="edges",
    )

    # ---- node set of the neighbourhood
    e_ids = pq.read_table(e_path, columns=["clId1", "clId2", "txId"]).to_pandas()
    hood = np.unique(np.concatenate([
        e_ids["clId1"].to_numpy(), e_ids["clId2"].to_numpy(), labeled.to_numpy()
    ]))
    hood_arr = pa.array(hood, type=pa.int64())
    print(f"Neighbourhood nodes: {len(hood):,}")

    # ---- 2. node features for the neighbourhood
    print("\n[2/2] Filtering background_nodes.csv ...")
    n_path = os.path.join(out, "nodes_1hop.parquet")
    n_scanned, n_kept = stream_filter(
        os.path.join(d, "background_nodes.csv"),
        id_cols={"clId"},
        keep_mask_fn=lambda b: pc.is_in(b.column("clId"), value_set=hood_arr),
        out_path=n_path,
        label="nodes",
    )

    # ---- checks
    sub_tx_found = sub_edges["txId"].isin(e_ids["txId"]).mean()
    summary = {
        "labeled_nodes": int(len(labeled)),
        "background_edges_scanned": int(e_scanned),
        "edges_kept": int(e_kept),
        "neighbourhood_nodes": int(len(hood)),
        "background_nodes_scanned": int(n_scanned),
        "node_rows_kept": int(n_kept),
        "nodes_missing_features": int(len(hood) - n_kept),
        "share_of_subgraph_edges_found_by_txId": round(float(sub_tx_found), 6),
        "edge_txId_unique": bool(e_ids["txId"].is_unique),
    }
    with open(os.path.join(out, "extract_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"{k:42s}: {v}")
    print("=" * 60)
    print(f"Files written to {out}. Send this summary back.")


if __name__ == "__main__":
    main()
