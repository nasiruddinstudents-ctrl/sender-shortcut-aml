# Elliptic2 audit

Scripts for the Elliptic2 part of the paper (Sections 4.1–4.5 and 5). Download Elliptic2 from Elliptic on Kaggle first; no data is included here.

Run in this order:

| Step | Script | Purpose | Paper |
|---|---|---|---|
| 1 | `inspect_elliptic2.py` | Verify dataset properties from the released files | Table 1 |
| 2 | `extract_1hop.py` | Stream the background files and extract the one-hop neighborhood of all labeled subgraphs | Section 4.1 |
| 3 | `leakage_check.py` | Leakage diagnostic: share of suspicious test subgraphs with a sender that fed a suspicious subgraph in training | Section 4.3 |
| 4 | `entity_split_benchmark.py` | Random and entity-aware splits; lookup, novelty and gradient-boosting baselines | Table 2, Sections 5.1–5.4 |
| 5 | `revclassify_ds.py` | RevClassify-DS reimplementation and receivers-only variant | Table 2, Section 5.5 |
| 6 | `sensitivity_benchmark.py` | Protection threshold and hub cut-off sensitivity | Table 3, Section 5.6 |
| 7 | `robustness_benchmark.py` | Removal of cyclic subgraphs and other robustness checks | Table 3, Section 5.6 |

Paths to the downloaded data are set at the top of each script; edit them before running.

The full background edge file needs about 143 GB of memory if loaded with features, which is why step 2 streams it instead. Steps 3–7 work on the extracted neighborhood (about 1.6 GB).
