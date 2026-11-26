# Who Sent It or How It Moved? Sender Shortcuts in Cryptocurrency AML Benchmarks

Companion code for:

> M. N. Uddin, "Who Sent It or How It Moved? Sender Shortcuts in Cryptocurrency
> Anti-Money-Laundering Benchmarks, Audited on Elliptic2 and Elliptic++,"
> arXiv:2611.29744, 2026.

The scripts audit whether illicit-activity labels in two public cryptocurrency
AML benchmarks can be predicted from the identity of the sending entity alone,
rather than from transaction-graph structure, and report how much of the
benchmark performance this sender shortcut explains.

## Repository layout

- `elliptic2/` — audit scripts for the Elliptic2 subgraph-classification benchmark
- `elliptic_pp/` — audit script for the Elliptic++ benchmark
- `requirements.txt` — Python dependencies

## Data

The datasets are **not** redistributed here. Download them from their official sources
and accept their terms of use:

- **Elliptic2** — released by Elliptic (Bellei et al., 2024)
- **Elliptic++** — released by Elmougy and Liu (KDD 2023): https://github.com/git-disl/EllipticPlusPlus

Set the data paths at the top of each script to your local copies before running.

## Setup

```bash
pip install -r requirements.txt
```

Then run the scripts in `elliptic2/` and `elliptic_pp/`. Each script prints the
results reported in the corresponding tables of the paper.

## Contact

Mohammad Nasir Uddin — ORCID 0009-0009-0990-4616
